
大家好，今天我们继续实现 Makemore。在上节课中，我们实现了二元语言模型，既使用了计数方法，也使用了一个超级简单的神经网络，该网络只有一个线性层。这是我们上节课构建的 Jupyter 笔记本，我们看到我们的方法是只查看前一个字符，然后预测序列中下一个字符的分布，我们通过计数并将它们归一化为概率来实现这一点，因此这里的每一行总和为一。

现在，如果你只有一个字符的上下文背景，这种方法确实可行且易于理解。但问题在于，由于模型只考虑了一个字符的上下文，其预测结果并不理想，生成的内容听起来不太像名字。然而，这种方法更大的问题在于，如果我们想在预测序列中的下一个字符时考虑更多的上下文，情况很快就会变得复杂起来。这个表格的大小会迅速膨胀，实际上它会随着上下文长度的增加而呈指数级增长。

因为如果我们每次只取一个字符，上下文的可能性只有 27 种。但如果取前两个字符来预测第三个字符，这个矩阵的行数（可以这样理解）就会突然变成 27 乘以 27，也就是有 729 种可能的上下文组合。如果取三个字符作为上下文，那么上下文的可能性会激增至 20,000 种。因此，这个矩阵的行数实在太多了，每种可能性的计数又太少，整个模型就会崩溃，效果很差。

所以今天我们就要转向这个要点，我们将实现一个多层感知机模型来预测序列中的下一个字符。我们采用的这种建模方法遵循了Bengio 等人在 2003 年发表的这篇论文。所以我这里打开了这篇论文。虽然这不是最早提出使用多层感知机或神经网络来预测序列中下一个字符或标记的论文，但它无疑是当时非常有影响力的一篇。它经常被引用作为这一思想的代表，我认为这是一篇非常出色的文章。

因此，这就是我们要先研究再实施的论文。这篇论文共有 19 页，我们没有时间深入探讨所有细节，但我建议大家去阅读它。文章通俗易懂、引人入胜，里面还包含了许多有趣的见解。

在引言部分，他们描述的问题与我刚才所述完全一致。为了解决这个问题，他们提出了以下模型。请注意，我们正在构建的是字符级语言模型，因此我们的工作是基于字符层面的。而在这篇论文中，他们使用了 17,000个可能单词的词汇表，构建的却是单词级语言模型。

但我们仍会坚持使用这些字符，只是采用相同的建模方法。现在，他们的做法基本上是建议将这 17,000 个单词中的每一个单词都关联到一个 30 维的特征向量上。因此，每个单词现在都被嵌入到一个 30 维的空间中。你可以这样理解。我们有 17,000 个点或向量在一个 30 维的空间里，你可能会觉得，这非常拥挤。对于这么小的空间来说，点的数量太多了。

最初，这些单词的向量是完全随机初始化的，因此它们在空间中随机分布。但接下来，我们将通过反向传播来调整这些单词的嵌入向量。因此，在这个神经网络的训练过程中，这些点或者说向量基本上会在这个空间中移动。你可能会想象，例如，那些意义非常相似或实际上是同义词的单词最终会出现在空间中非常相近的位置。相反，意义截然不同的单词则会出现在空间中的其他地方。除此之外，他们的建模方法与我们的完全相同。

他们使用多层神经网络，根据前面的单词预测下一个单词。为了训练神经网络，他们像我们一样，最大化训练数据的对数似然。因此，建模方法本身是相同的。

现在，他们有了一个关于这种直觉的具体例子。为什么它能奏效？基本上，假设你正在尝试预测“一只狗在空白处奔跑”这样的句子。现在，假设训练数据中从未出现过“一只狗在空白处”这个确切的短语。而现在，当模型部署到某个地方进行测试时，它试图生成一个句子。比如它说："一只狗在\_\_里奔跑"。由于它在训练集中从未遇到过这个确切的短语，就像我们所说的，你超出了分布范围。比如，你根本没有任何理由去怀疑接下来会发生什么。但这种方法实际上可以让你绕过这个问题。因为也许你没有看到确切的短语，一只狗正在某个地方奔跑。

但也许你见过类似的短语。也许你见过这样的句子：“狗在空白处奔跑”。而你的神经网络可能已经学会了 “a” 和 “the” 经常可以互换使用。因此，或许它提取了 "a" 的词嵌入和 "the" 的词嵌入，实际上让它们在向量空间中彼此靠近。这样你就能通过词嵌入传递知识，并实现这种方式的泛化。

同样地，神经网络可以知道猫和狗都是动物，它们经常出现在许多非常相似的上下文中。因此，即使你没有见过这个确切的短语，或者没有见过具体的行走或奔跑的动作，你仍然可以通过嵌入空间来传递知识。这样你就能将知识推广到新的场景中。

那么现在让我们向下滚动到神经网络的图表。这里有一个很好的图表。在这个例子中，我们采用了前三个单词。我们正在尝试预测一个序列中的第四个单词。正如我提到的，这三个前面的单词来自一个包含 17,000 个可能单词的词汇表。因此，每一个基本上都是输入单词的索引。

由于有 17000 个单词，所以这是一个介于 0 到 16999 之间的整数。现在，他们还有一个称为 $C$ 的查找表。这个查找表是一个 17000 乘以 30 的矩阵。基本上，我们在这里所做的是将其视为一个查找表。

因此，每个索引都会从这个嵌入矩阵中提取一行，这样每个索引都被转换为对应的 30 维词嵌入向量。于是，我们为三个单词设置了 30 个神经元的输入层，总共构成了 90 个神经元。这里还提到，这个矩阵 $C$ 在所有单词之间是共享的。所以我们一直在为每一个单词反复索引同一个矩阵 $C$。

接下来是这个神经网络中的隐藏层。这个隐藏层的大小是这个神经网络的一个超参数。所以我们用“超参数”这个词来表示神经网络设计者可以自由决定的设计选择。它的大小可以随心所欲地设定，可大可小。例如，这个尺寸可以设为 100。

我们将探讨隐藏层大小的多种选择，并评估它们的效果。假设这里有 100 个神经元，它们都将与构成这三个单词的 90 个单词或 90 个数字完全连接。因此，这是一个全连接层。接着是一个 tanh。然后是这个输出层。由于接下来可能出现 17,000 个单词，这一层拥有 17,000 个神经元，它们全都与隐藏层中的所有神经元完全连接。所以这里有很多参数，因为有很多单词。大部分计算都在这里。这是最耗时的层。

这里有 17000 个逻辑值。在此基础上，我们有一个 softmax 层，这个我们在之前的视频中也见过。每一个逻辑值都会被指数化，然后所有值会被归一化，使其总和为 1，这样我们就得到了序列中下一个单词的一个良好的概率分布。

当然，在训练过程中，我们实际上拥有标签。我们知道序列中下一个词的真实身份。我们会用这个词或其索引来提取该词对应的概率值，然后针对神经网络的所有参数，最大化这个词出现的概率。

所以参数就是这个输出层的权重和偏置、隐藏层的权重和偏置，以及嵌入查找表 C。所有这些都通过反向传播进行优化。至于这些虚线箭头，可以忽略它们。那代表了一种我们在这个视频中不会探讨的神经网络变体。

这就是设置步骤，现在我们来实现它。好的，我为本节课创建了一个全新的笔记本。我们正在导入PyTorch，同时也在导入 Matplotlib 以便创建图表。然后我将所有名字读入一个单词列表，就像我之前做的那样，并在这里展示前八个。请记住，我们总共有 32,000 个名字。这些只是前八个。

然后，我在这里构建字符词汇表以及从字符串形式的字符到整数的所有映射关系，反之亦然。现在，我们首先要做的是为神经网络编译数据集。我不得不重写这段代码。我马上给你看看它是什么样子的。这是我为数据集创建编写的代码。我先运行一下，然后简单解释一下它的工作原理。

首先，我们要定义一个称为"块大小"的概念。这本质上是指用于预测下一个字符的上下文长度。在这个例子中，我们使用三个字符来预测第四个字符，因此我们的块大小为三。

这就是支撑预测的块的大小。然后在这里，我构建了 x 和 y。x 是神经网络的输入，y 是 x 中每个样本的标签。接着，我遍历前五个单词。为了在开发所有代码时提高效率，我只处理前五个，但之后我们会回来修改这里，以便使用整个训练集。

所以在这里我打印了单词 Emma，基本上是在展示我们可以生成的示例，即从单个单词 Emma 中生成的五个示例。当我们只给出点点的上下文时，序列中的第一个字符是 e。在这种上下文中，标签是 m。当上下文是这样时，标签是 m，以此类推。我构建这个的方法是首先从一个填充的零标记上下文开始。

然后我遍历所有字符。我获取序列中的字符，基本上构建出当前字符的数组 y，以及存储当前运行上下文的数组 x。接着在这里，你看，我打印所有内容，然后在这里裁剪上下文并将新字符按顺序输入。

或者也可以是，比如说，十个，那么它看起来就会像这样。我们取十个字符来预测第十一个，并且总是用点来填充。让我把它调回到三个，这样我们就和论文里展示的一致了。最后，数据集目前如下所示。从这五个单词中，我们创建了一个包含 32 个样本的数据集，每个输入神经网络的样本是三个整数，我们还有一个同样为整数的标签 y。所以 x 看起来是这样的。这些是各个样本。

然后 y 是标签。基于此，我们现在来编写一个神经网络，接收这些 x 并预测 y。首先，我们构建嵌入查找表 C。我们有 27 个可能的字符，并将它们嵌入到低维空间中。

在这篇论文中，他们使用了 17,000 个单词，并将它们嵌入到维度低至 30 的空间中。也就是说，他们将 17,000 个单词压缩进了 30维 空间。而在我们的案例中，只有 27 个可能的字符，所以让我们尝试将它们压缩进更小的空间，比如一开始可以尝试二维空间。

所以这个查找表会是随机数字，我们将有 27 行和两列。对吧？所以 27 个字符中的每一个都会有一个二维嵌入。这就是我们初始时随机初始化的嵌入矩阵 C。

现在，在我们使用这个查找表 C 将所有整数嵌入输入 x 之前，让我先尝试嵌入一个单独的整数，比如5，这样我们就能理解这个过程是如何运作的。当然，一种方法是直接取 C，然后索引到第 5 行，这样就得到了一个向量，即 C 的第 5 行。这是一种方法。我在上一讲中介绍的另一种方法看起来不同，但实际上是一样的。

所以在上一讲中，我们做的是把这些整数拿出来，先用独热编码对它们进行编码。所以我们用 f.one-hot，想对整数 5 进行编码，并且告诉它类别数是 27。这样就会得到一个 26 维的全零向量，只有第五位是 1。现在，这实际上行不通。原因是这个输入必须是一个张量。我故意制造了一些这样的错误，就是为了让你们看到一些错误以及如何修复它们。

所以这必须是一个张量，而不是整数，修复起来相当简单。我们得到一个独热向量，第五维度为 1，其形状为 27。现在请注意，正如我在之前的视频中简要提到的，如果我们拿这个独热向量乘以 C，那么你会期待什么结果呢？首先，你可能会预期会报错，因为期望的是长整型标量类型，但实际找到的是浮点型。

所以有点让人困惑，但问题在于独热编码的数据类型是长整型。它是一个 6 4位整数，但这是一个浮点张量。PyTorch 不知道如何将整数与浮点数相乘，这就是为什么我们必须显式地将其转换为浮点数，以便能够进行乘法运算。

现在，这里的输出实际上是相同的，之所以相同是因为这里的矩阵乘法的工作原理。我们有一个独热向量乘以 C 的列，由于所有的零，它们实际上会屏蔽掉 C 中除第五行以外的所有内容，而第五行被提取出来。因此，我们实际上得到了相同的结果。

这说明我们可以这样理解第一个部分——即整数的嵌入表示：既可以将其视为通过查找表 C 进行整数索引的过程，也可以等效地将其视作这个大神经网络的第一层。这一层的神经元没有非线性激活函数（如 tanh），它们只是线性神经元，其权重矩阵就是 C。然后我们将整数编码为独热向量输入神经网络，这第一层本质上完成了嵌入操作。因此这两种方式是实现同一功能的等效方法。

我们决定采用索引方式，因为它速度要快得多。我们将摒弃神经网络中独热编码输入的解读方式，直接索引整数并使用嵌入表。现在，嵌入单个整数（比如数字 5）非常简单——只需让 PyTorch 检索矩阵 C 的第五行即可。但如何同时嵌入存储在数组 x 中的这 32×3 个整数呢？幸运的是，PyTorch 的索引功能相当灵活且强大。所以，不能像这样直接获取单个元素5。实际上，你可以使用列表来索引。例如，我们可以获取第 5、6、7 行，操作方式如下。

我们可以用列表进行索引。不仅限于列表，实际上也可以用整数张量来索引。这里有一个整数张量 5、6、7，同样可以正常工作。

事实上，我们也可以重复第 7 行并多次检索它，这样同一个索引就会在这里多次嵌入。所以这里我们用一个一维的整数张量进行索引，但实际上你也可以用多维的整数张量进行索引。这里我们有一个二维的整数张量。

因此，我们只需在 x 处执行 C 操作即可，这完全可行。其形状为 32×3，即原始形状。现在，对于每一个 32×3 的整数，我们都在这里检索到了对应的嵌入向量。

简单来说，我们以这个为例。第 13 个，或者说索引 13，第二个维度，例如是整数 1。所以在这里，如果我们计算 C(x)，得到那个数组，然后我们通过该数组的 13 索引 2，就能得到这里的嵌入。

你可以验证 C 在 1 处的值，也就是该位置的整数，确实等于这个。你看它们是相等的。所以长话短说，PyTorch 的索引功能非常棒，要同时嵌入 X 中的所有整数，我们只需简单地执行 C(x)，那就是我们的嵌入，而且这样就能正常工作。

现在让我们构建这一层，即隐藏层。我们将随机初始化这些权重，我称之为 w1。该层的输入数量将是 3 乘以 2，对吧？因为我们有二维嵌入，共三个，所以输入数量是 6。而这一层的神经元数量则由我们自行决定。

我们以100个神经元为例。然后偏置项也会随机初始化，我们只需要100个。现在的问题是，我们不能简单地...通常我们会获取输入，在这里就是嵌入向量，我们希望将其与这些权重相乘，然后再加上偏置项。

这大致就是我们想要做的。但这里的问题是，这些嵌入被堆叠在这个输入张量的维度中，所以这行不通，这个矩阵乘法，因为这是一个32乘3乘2的形状，而我无法将其乘以6乘100。因此，我们需要以某种方式将这些输入在这里连接起来，以便我们可以按照这些思路进行操作，但目前这不可行。

那么我们如何将这个32×3×2的张量转换为32×6的张量，以便能够真正执行这里的乘法运算呢？我想告诉大家，在Torch中通常有多种方法可以实现你的目标，其中一些方法会更快、更好、更简洁等等。这是因为Torch是一个非常庞大的库，包含了大量的函数。如果我们查看文档并点击Torch部分，你会发现这里的滚动条非常小，这正是因为你可以调用如此之多的函数来转换张量、创建张量、对张量进行乘法、加法等各种不同的操作。

And so this is kind of like the space of possibility, if you will. Now one of the things that you can do is if we can Ctrl-F for concatenate, and we see that there's a function Torch.cat, short for concatenate. And this concatenates a given sequence of tensors in a given dimension, and these tensors must have the same shape, etc. 

So we can use the concatenate operation to, in a naive way, concatenate these three embeddings for each input. So in this case, we have mp of this shape. And really what we want to do is we want to retrieve these three parts and concatenate them. 

So we want to grab all the examples. We want to grab first the zeroth index, and then all of this. So this plucks out the 32 by 2 embeddings of just the first word here. 

And so basically we want this guy, we want the first dimension, and we want the second dimension. And these are the three pieces individually. And then we want to treat this as a sequence, and we want to Torch.cat on that sequence. 

So this is the list. Torch.cat takes a sequence of tensors, and then we have to tell it along which dimension to concatenate. So in this case, all of these are 32 by 2, and we want to concatenate not across dimension 0, but across dimension 1. So passing in 1 gives us the result that the shape of this is 32 by 6, exactly as we'd like. 

So that basically took 32 and squashed these by concatenating them into 32 by 6. Now this is kind of ugly because this code would not generalize if we want to later change the block size. Right now we have three inputs, three words, but what if we had five? Then here we would have to change the code because I'm indexing directly. Well, Torch comes to rescue again because there turns out to be a function called unbind, and it removes a tensor dimension. 

So it removes the tensor dimension, returns a tuple of all slices along a given dimension without it. So this is exactly what we need, and basically when we call Torch.unbind of m and pass in dimension 1, index 1, this gives us a list of tensors exactly equivalent to this. So running this gives us a line 3, and it's exactly this list. 

So we can call Torch.cat on it and along the first dimension, and this works, and this shape is the same. But now it doesn't matter if we have block size 3 or 5 or 10, this will just work. So this is one way to do it, but it turns out that in this case there's actually a significantly better and more efficient way, and this gives me an opportunity to hint at some of the internals of Torch.tensor. So let's create an array here of elements from 0 to 17, and the shape of this is just 18. 

It's a single vector of 18 numbers. It turns out that we can very quickly re-represent this as different sized n-dimensional tensors. We do this by calling a view, and we can say that actually this is not a single vector of 18, this is a 2 by 9 tensor, or alternatively this is a 9 by 2 tensor, or this is actually a 3 by 3 by 2 tensor. 

As long as the total number of elements here multiply to be the same, this will just work. And in PyTorch, this operation calling that view is extremely efficient, and the reason for that is that in each tensor there's something called the underlying storage. And the storage is just the numbers always as a one-dimensional vector, and this is how this tensor is represented in the computer memory. 

It's always a one-dimensional vector. But when we call that view, we are manipulating some of the attributes of that tensor that dictate how this one-dimensional sequence is interpreted to be an n-dimensional tensor. And so what's happening here is that no memory is being changed, copied, moved, or created when we call that view. 

The storage is identical, but when you call that view, some of the internal attributes of the view of this tensor are being manipulated and changed. In particular, there's something called storage offset, strides, and shapes, and those are manipulated so that this one-dimensional sequence of bytes is seen as different n-dimensional arrays. There's a blog post here from Eric called PyTorch Internals where he goes into some of this with respect to tensor and how the view of a tensor is represented.

And this is really just like a logical construct of representing the physical memory. And so this is a pretty good blog post that you can go into. I might also create an entire video on the internals of TorchTensor and how this works. 

For here, we just note that this is an extremely efficient operation. And if I delete this and come back to our EMB, we see that the shape of our EMB is 32 by 3 by 2. But we can simply ask for PyTorch to view this instead as a 32 by 6. And the way this gets flattened into a 32 by 6 array just happens that these two get stacked up in a single row. And so that's basically the concatenation operation that we're after. 

And you can verify that this actually gives the exact same result as what we had before. So this is an element y equals, and you can see that all the elements of these two tensors are the same. And so we get the exact same result. 

So long story short, we can actually just come here. And if we just view this as a 32 by 6 instead, then this multiplication will work and give us the hidden states that we're after. So if this is h, then h-shaped is now the 100 dimensional activations for every one of our 32 examples. 

And this gives the desired result. Let me do two things here. Number one, let's not use 32. 

We can, for example, do something like EMB.shape at 0 so that we don't hard code these numbers. And this would work for any size of this EMB. Or alternatively, we can also do negative 1. When we do negative 1, PyTorch will infer what this should be. 

Because the number of elements must be the same, we're saying that this is 6, PyTorch will derive that this must be 32, or whatever else it is if EMB is of different size. The other thing is here, one more thing I'd like to point out is here when we do the concatenation, this actually is much less efficient because this concatenation would create a whole new tensor with a whole new storage. So new memory is being created because there's no way to concatenate tensors just by manipulating the view attributes. 

So this is inefficient and creates all kinds of new memory. So let me delete this now. We don't need this. 

And here to calculate h, we want to also dot 10h of this to get our h. So these are now numbers between negative 1 and 1 because of the 10h. And we have that the shape is 32 by 100. And that is basically this hidden layer of activations here for every one of our 32 examples. 

Now there's one more thing I've lost over that we have to be very careful with, and that's this plus here. In particular, we want to make sure that the broadcasting will do what we like. The shape of this is 32 by 100, and B1's shape is 100.

So we see that the addition here will broadcast these two. And in particular, we have 32 by 100 broadcasting to 100. So broadcasting will align on the right, create a fake dimension here.

So this will become a 1 by 100 row vector. And then it will copy vertically for every one of these rows of 32 and do an element-wise addition. So in this case, the correct thing will be because the same bias vector will be added to all the rows of this matrix. 

So that is correct. That's what we'd like. And it's always good practice to just make sure so that you don't shoot yourself in the foot. 

And finally, let's create the final layer here. So let's create W2 and B2. The input now is 100, and the output number of neurons will be for us 27, because we have 27 possible characters that come next. 

So the biases will be 27 as well. So therefore, the logits, which are the outputs of this neural net, are going to be H multiplied by W2 plus B2. Logits.shape is 32 by 27, and the logits look good.

Now, exactly as we saw in the previous video, we want to take these logits, and we want to first exponentiate them to get our field.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)