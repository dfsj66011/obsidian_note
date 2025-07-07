
Paper: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)


#### 问题引入

这里要讨论的问题是 *序列建模*。到目前为止，深度学习中有两种主要的序列建模方法。一种叫做 Transformer，另一种叫做循环神经网络。还有一些混合变体，将注意力机制与循环神经网络等相结合。

对于循环神经网络来说，我们无法并行计算每个位置的输出，必须一步一步地进行。Transformer 的优势在于其可并行性，只需增加 GPU 数量就能训练大规模模型。循环神经网络的问题实际上有两个。首先，它是不可并行的；其次是隐藏状态的大小是固定的，可以按照你的需求任意扩大，但一旦选定了架构，它就是固定的。

另一方面，当我们使用 Transformer 模型时，语言模型所处理的输入大小是在不断增长的。因此， Transformer 的隐藏状态，也就是它的记忆，即我们输入 Transformer 以预测下一个标记的内容，实际上在不断增长。这也是另一个问题。

因此，在进行非常长的序列建模时，我们需要两样东西。

1. 我们希望语言模型能够利用迄今为止接收到的所有输入信息。这一点通过 Transformer 架构可以轻松实现。但问题在于，Transformer 需要持续保存所有输入标记才能获取完整上下文，这会导致内存占用不断增长。
2. 如果我们内存有限，可以使用循环神经网络，但它们在训练过程中无法并行化。第二个问题是它们的内存是固定的。固定内存还存在另一个问题，因为它无法调整。我们无法选择内在的内容。因此，有时语言模型能看到一些信息，有时则无法看到某些信息。

此外，我们已经看到许多架构试图改进循环神经网络的这种记忆能力。例如，在 Mamba 模型中，他们采用了一种名为 “hypometrics” 的特殊矩阵结构，旨在更高效地存储信息。然而在实际应用中，其表现并不如预期理想。

-------

#### 混合架构模式

对于 Transformer，我们有一个不断增长的记忆，一个叫做 KV 缓存的东西，它包含了所有过去的标记。因此，Transformer 始终可以利用所有过去的标记来预测下一个标记。而在循环神经网络中，我们有一个过去的记忆，它将所有过去的标记压缩成一个固定大小的记忆。然而，这也有其自身的问题，有时信息会丢失，因为它是固定的，而你试图塞入很多东西。所以我们无法决定里面有什么。我们只是希望网络能学会保留最重要的信息，而忘记不太重要的信息。问题在于，当我们训练语言模型时，我们会输入大量数据。

想象我们有一个混合模型，即一个结合了 Transformer 和循环神经网络的模型。假设这里是一个注意力层，也就是 Transformer 层。我们称之为注意力层。这是一个循环神经网络。假设这是一种新型的、可以并行化的循环网络。实际上，现在已经有一些可以并行化的新架构了。

但问题在于，这里的RNN（循环神经网络）会生成一个固定大小的记忆体。如果你输入1000个标记，它输出的记忆体（将被注意力机制利用）并不会包含 1000 个标记，而是更少——因为循环神经网络的核心目标就是将信息压缩成固定大小的记忆体，供这个 Transformer 模型（具体指这里的注意力层）使用。不过值得注意的是，这个注意力层非常擅长高效利用输入的数据。但这些数据并非完整的序列，因为我们已用循环神经网络对其进行了压缩。我们希望注意力机制能够利用循环神经网络压缩的信息来完成预测下一个标记的任务。

如果我们采用这种方式，想象一下我们拥有这种架构，即注意力机制与循环神经网络相结合的混合架构。这种架构的问题在于，当你进行训练时，由于我们采用深度学习的方式，我们会迫使模型学习我们设定的任何目标。它将被迫学习这种循环神经网络，以某种方式压缩信息，以便注意力机制能够利用这些信息。而注意力机制也将被迫从循环神经网络生成的这种压缩状态中提取任何可用的信息。这是有益的。所以当你训练它时，实际上损失会降低，而且你会发现它的表现相当不错。

----

#### 混合架构模式的问题

然而，当你在实践中使用它时，你提供给模型的问题可能不是语言模型以前见过的内容。所以我们也许可以称之为分布外数据。因此，模型可能不知道如何有效地压缩数据，不清楚该保留哪些信息、舍弃哪些信息。在这种情况下，循环神经网络就无法完成数据压缩的任务。由于预测下一个标记所需的数据没有被很好地压缩，注意力层也就无法利用这些数据来预测下一个标记。

因此在训练时，我们发现这种混合架构效果非常好，但在测试时，也就是实际使用时，我们发现它们的效果并不理想。这就是原因之一。

例如，如果我有一段很长的Python源代码，我不应该把注意力集中在那些可能是重复的注释上，而应该专注于代码本身；或者当我看到一些C#或C代码时，我不应该过分关注那些括号，因为它们可能只是冗余的，而应该把重点放在表达式等等上。因此，它实际上学会了压缩信息，但仅限于它在训练时见过的信息。现在，我们终于可以讨论这篇论文了。

-----------

#### 论文观点



这篇论文的观点是，我们需要为这些模型配备某种记忆机制。在Transformer模型中，我们采用了这种KV缓存机制。但问题在于，这个KV缓存会不断增长。因此，不断增长的KV缓存带来的问题是它需要大量内存。实际上，大多数模型并不受限于此。当前模型无法拥有非常大的上下文窗口，实际上是因为该模型的推理成本过高。

因此，推理成本确实非常高昂，因为我们需要保留KV缓存（键值缓存），而KV缓存是逐层存在的。对于大型模型来说，它们的层数非常多。这意味着在预测每个token时，都必须为模型的每一层保留所有token的数据。所以这个过程极其耗费资源。

那么，解决这种无限增长内存的方案就是采用压缩内存，但这种压缩内存仅在训练时效果很好。因此，问题是，我们能否拥有一个在测试时训练的内存模块？这就是为什么我们要讨论在测试时学习记忆，这种记忆在检索时是有效的，因为内存的目标是检索模型所需的显著信息，这些信息在测试时被准确输入，而不仅仅是训练时见过的信息。这就是我们试图通过Titans解决的问题。

现在，他们的做法是这样的。他们说，好吧，想象我们有一个模型，想象我们有一个模块，我们称之为M，这个模块，我们可以把它看作是模型中的一个层。那么，好吧，让我画一下。我觉得如果我们能画出来会容易得多。让我们加一张新纸，一个新页面。那么，好的。

假设我们有一个很长的序列。我们已经知道，循环神经网络的任务是将这个很长的序列压缩，以便Transformer可以使用它。现在让我们用泰坦们来试试看。这有什么不同？然后我们会检查所有细节。我们有了这个输入。让我们再到这里来一次。

所以我们有这个输入。我们将其转换为嵌入。然后，我会稍微不同地绘制一下，稍后我会解释为什么这样做。我们有一些，假设我们再次采用了一种混合架构，结合了Transformer和循环层，但我不会画出循环层。所以这是第一层，我觉得太大了。好吧，我们称之为L1。

第一层带注意力，第二层带注意力，第三层带注意力。然后我们得到输出，也就是逻辑值。好的。我认为现在更清晰了，对吧？好的。那么想象一下，在这个架构中我们有另一个模块，我们称之为记忆模块。让我们叫它神经记忆，因为这里就是这样称呼它的。

那就让我们称之为神经记忆吧，我会把它画成一个外部模块——神经记忆。现在我想展示一下神经记忆是如何运作的，然后我们再详细研究它实际上是如何训练的。我们通常训练模型的方式是这样的，想象一下，好吧，让我们先退一步说。

我们该如何训练这个模型？我们会输入一个序列。想象一下有一百万个标记。所以想象一个非常大的序列。假设有一百万个标记。你将这一百万个标记序列转换为嵌入向量。然后，在循环神经网络中处理这些嵌入向量，将这一百万个标记压缩成大约一千个标记，因为它的目标就是压缩信息，对吧？这样，输入到注意力机制的序列就变短了，因为注意力机制的问题是它的计算复杂度是平方级的。

因此，输入规模越小，计算效果越好。我们将这一千个压缩后的标记输入注意力机制，然后强制它仅利用这一千个压缩标记来预测下一个标记。虽然我们输入了一百万个标记，但要求注意力层仅利用少得多的信息来预测下一个标记。

所以我们希望循环神经网络擅长选择正确的标记来保留，并丢弃那些不保留的标记。实际上，这并不是真正的标记剪枝机制，而是一种标记压缩机制，但你可以把它想象成标记剪枝，就像它被输入了一百万个标记，但只保留了对于预测下一个标记最重要的一千个。而且这是在训练时完成的。

因此，在训练时，我们输入这一百万个标记，计算输出结果。由于训练时我们知道下一个标记应该是什么，我们会强制计算与预期下一个标记之间的损失，然后通过反向传播更新模型参数，并对所有序列重复这一过程。而标题的处理方式则有所不同。假设你再次有一百万个标记，你需要分两步进行。

首先我们要做的是，好吧，我们有这个输入，我们将其转换为嵌入。在训练循环中，想象我们正在训练这个泰坦架构，我们首先训练这个神经模块，让它学会记住我们的一百万个标记，然后我们要求它检索预测下一个标记所需的信息，并将其输入到注意力层。所以这个，我们称之为注意力层。

所以这是一个注意力层。这是一个注意力层，这也是一个注意力层。看看这里的区别。在我们有输入之前，我们预测输出，计算损失，反向传播并更新模块的所有参数。这里我们将做一些不同的事情。我们有一个输入，即一百万个标记。

我们将它们转化为嵌入向量，然后如此这般。我们在这里训练这个模块，它是独立的，论文中称之为训练的内循环。我们训练这个神经记忆模块，稍后我们会看到如何专门训练它，目的是让这个神经记忆模块能够全面掌握这些数据，以便在需要时轻松检索这些数据。

所以我们获取这一百万个标记，将它们转换为嵌入向量，在一个内部循环中训练这个神经记忆。然后我们取出这个经过训练以记忆这些数据的神经记忆，要求它从所见内容中检索出任何重要的信息，并将其作为注意力层的输入，这样注意力层就可以利用这个压缩的记忆来生成输出并预测下一个标记。这一过程不仅在训练时进行，在测试时也同样适用。

因此，当我们使用注意力机制与混合架构（例如注意力机制加循环神经网络）在测试时，也就是推理时，通常我们有一个提示词。想象一下这个提示词非常大，比如你要求Chargepd分析一个非常大的GitHub仓库的整个代码库。这时，这一百万个标记会被输入到循环神经网络中，而此时的网络是固定的，也就是说我们正在使用模型，不再改变其参数。

循环神经网络的任务是压缩数据，因此它会将这些标记压缩成一个更小的序列，然后输入到注意力层，并产生输出逻辑值。然而，可能我们输入到这些循环神经网络的信息有些偏离分布，循环神经网络从未见过类似的情况，因此在压缩这些数据时可能会表现得很糟糕。由于它在压缩这些数据时表现不佳，因为它不知道应该保留什么和不保留什么，注意力层将无法利用最重要的信息，从而无法很好地预测下一个标记，最终导致输出结果不佳。

而对于Titans来说，即使在测试阶段，也就是推理阶段，我们实际上也在训练一个模型，现在我来展示具体是如何实现的。想象一下，我们有一个GitHub代码库，它非常庞大，包含了一百万个token，我们希望语言模型能够分析这些内容。我们首先将这些token转换为嵌入向量，然后实时训练一个神经记忆模块，它的任务就是尽可能多地学习这一百万个token的信息，并提取其中最显著的部分——因为神经记忆的核心功能就是信息压缩。在这个内部循环训练完成后，我们提取这些信息并将其输入到注意力层中，这样注意力层就能充分利用神经记忆所检索到的信息。

所以，关于Titans，我们不仅仅有一个RNN（这是我们的记忆，在训练时被训练，之后就不再训练了，每当它看到从未见过的东西时就会失控）。我们有一个神经记忆，可以在推理时即时训练，其唯一目的是压缩信息。由于我们是在推理时训练它，我们希望它在从未见过的数据上也能表现得更好。现在，根据他们在论文中发布的基准测试结果（但实际上这在所有论文中都会发生，所以你永远不要相信基准测试，事情就是这样）。

干得好。现在让我们来看看细节。我想提醒你，我们要解决的问题是长上下文建模。长上下文建模存在一个问题，即使用Transformer进行长上下文推理的成本非常高。而RNN的问题在于，虽然我们在某些数据上对它们进行了训练，但当它们遇到从未见过的内容时，它们不知道如何压缩信息、该保留什么、不该保留什么，因此会变得混乱。由于这种混乱，它们无法很好地完成这项任务。注意力层无法有效利用这些信息，最终导致输出结果非常糟糕。

我们希望在使用神经网络记忆时，能在模型推理过程中动态训练记忆功能，专注于对输入数据进行压缩处理。现在我们可以深入细节了。好的，这里他们先对"什么是记忆"或"什么是线性注意力"等概念做了初步概述——不过目前我们暂时不需要关注这些内容。

他们说，想象一下我们有一个只有两种操作的内存模块：一种是写入操作，一种是读取操作。我们希望在推理时和训练时都能对这个内存进行读写。那么如何训练这个内存呢？首先，这种神经记忆本身就是一个神经网络，你可以把它看作是一个独立于其他架构的外部神经网络，而其他架构会使用这个神经记忆。所以你需要想象有一个像Transformer这样的模型正在利用这个神经记忆。

那么如何在推理时训练这个神经记忆呢？因为这是我们的问题所在——在训练时我们知道怎么做，只需输入数据、计算输出、反向传播，问题就解决了。但在推理时该如何操作呢？这就是他们在此探讨的重点。他们说：好吧，假设我们拥有这个记忆模块，首先需要考虑如何更新其中的信息（他们想更新记忆信息）——让我们再退一步思考：我们希望这个记忆模块实现什么功能？我们希望它能学会提取需要记忆的任何信息。为此，他们采用了一种非常特殊的损失函数，本质上属于重构损失的类型。

想象一下，我们拥有这样一个记忆系统。当我们要求它记住某个输入序列时——暂且称之为X，即这里的Xt。我们会通过两个线性投影（分别称为Wk和Wv）对其进行映射，这两个投影本质上与我们注意力机制中使用的相同。这个记忆系统只有在学会重建它所见过的数据时，才能出色地完成工作。而这里展示的损失函数（即L2损失）正是为了让它学会记住同一数据在"键投影"和"值投影"之间的映射关系。

所以它某种程度上学会了重现相同的数据，这就是记忆的作用。也就是说，如果我存入一些东西，我应该能够检索出相同的东西，所以我应该能够尽可能多地从存入的东西中获取信息。那么如何训练它呢？训练方法是这样的：假设我有一个记忆，我想通过某种梯度下降来更新这个记忆。梯度下降的工作原理是这样的：想象我们有一个神经网络，梯度下降的基本版本如下操作。我们有一个带有一些参数的神经网络，我们称这些参数为θ。假设在时间i时的参数为θ。

因此在训练的第i步，模型的参数会根据前一步的参数进行更新，具体方式是用前一步的参数减去一个我们称为γ的学习率，再乘以损失函数对模型参数的梯度。这个梯度告诉我们该如何调整参数才能最大化损失，但我们实际上是沿着梯度的反方向移动，这就是为什么你会看到一个减号——我们是在朝着与最大化损失相反的方向更新参数。

所以我们更新参数以减少损失，这就是我们在这里所做的。我们说，我们希望以某种方式更新我们的记忆，以最小化这里的损失，也就是记忆损失，也就是我们之前看到的重构损失。这个损失告诉我们，如果我要求记忆检索一些信息（即数据的键投影），它应该重新创建这些数据。在论文中，他们将这个记忆建模为一个线性层，所以线性层就是一个权重矩阵的矩阵乘法。因此，这里的记忆模块M只不过是一个线性层的权重矩阵。

所以我们正在修改这个权重矩阵a，因此神经记忆只是一个矩阵w。我们以这样一种方式修改这个w，以减少数据的重构损失，就像我们训练神经网络一样。所以我们用参数训练神经网络以减少损失，这些参数的计算方式使得它们能够产生尽可能小的损失。同样地，我们正在更新这个w矩阵，这将作为我们的记忆，以这样一种方式使得它能够产生尽可能小的信息损失，因为这是我们优化的损失，也就是重构损失。当他们称之为“意外”时。

所以这个呃，关于W矩阵（也就是我们的记忆）的梯度，相对于损失函数关于这个记忆W的梯度，他们称之为“惊讶度”。因为损失越大，模型在重构数据时遇到的困难就越大，这意味着模型看到这些数据时会感到“惊讶”。呃，这就是为什么他们称之为“惊讶度”。如果你曾经研究过优化器的工作原理，你会记得在深度学习中有一个叫做“动量”的概念。通常我们不会简单地这样更新模型参数，因为比如说，有时候我们希望保留——首先，我们不想——好吧，首先，损失是通过小批量梯度下降计算的，这意味着我们不是在整个输入数据集上计算损失，而是在数据的实例上计算。

就像一小批数据，而这个梯度的方向实际上是随机的，这意味着它不是梯度的真实方向，也就是说它会来回波动。想象一下梯度的真实方向在这里，但如果我们用第一批数据训练，可能方向是这个方向；下一批数据可能是那个方向；再下一批又可能是另一个方向，等等。平均来看，它会指向梯度的正确方向，但每一步都会带有噪声，因为我们不想在训练的每一步都过于自信地迈步。我们加入了这个动量项，动量项基本上是对所有梯度进行指数移动平均。

这样我们也能保留一些关于过去计算出的梯度的信息，以平滑权重的变化，避免调整幅度过大。也就是说，我们不会对每一步都赋予相同的权重。他们引入惊喜的动机是这样的：他们认为，如果训练记忆模型去重现数据，那么在遇到新数据后可能会错过这些新数据。也许有些新数据是模型应该记住的，但梯度在一段时间后会逐渐消失，导致模型错过它。

为了避免这种机制，他们采用了类似于我们在模型训练中使用的动量方法，并将其称为“过去惊喜”。这个“过去惊喜”实际上就是我们优化器（比如Adam优化器）中动量项的过去梯度。而“瞬时惊喜”则是指相对于当前输入的梯度。让我们回顾一下目前所讲的内容：我们有一个记忆模块，它只是一个我们希望优化的权重矩阵W。我们希望随着接收到的每个标记不断调整这个W，使其能够封装输入中的所有信息。我们如何知道它是否捕获了输入中的所有信息呢？因为我们要求它最小化输入的重构损失。

现在的问题是，我们不仅希望在训练期间对这个神经模型进行训练，还希望在推理时也进行训练。因为如果我们只在训练时进行，那么在推理时每次遇到从未见过的新信息时，它可能在压缩方面表现不佳，从而无法正常工作。那么如何在推理时实现这一点呢？我们实际要做的步骤如下：在推理时，我们假设有输入数据。第一个输入——让我把这些公式都写下来，以便我们可以在这里引用它们。这个公式我粘贴在这里，然后我们还要复制这个损失函数。

好的，让我们来了解一下推理时它是如何工作的。想象一下，我们有一百万个标记，实际上，假设我们想生成大量标记，而我们一开始只有一个标记，也就是说提示只有一个标记。接下来会发生的是，我们有了这一个标记，我们暂且称之为“一个标记”，我们将其作为输入提供给模型，模型会将其转换为嵌入表示，也就是只有一个嵌入向量。我们希望在这个单一标记上训练我们的神经记忆，让它学会重新生成这一个标记。具体操作方法是：首先，我们获取这个单一的嵌入向量，通过将其与一个名为wk的矩阵和另一个名为wb的矩阵进行矩阵乘法运算，将其投影为键和值。然后我们进行计算，这一步被称为记忆检索。

因为记忆仅被建模为一个线性层的权重矩阵w，所以从这个记忆中检索信息实际上就是w乘以输入，而输入他们称之为qt，所以它是输入通过wq矩阵的另一个投影。这个kt来自wk乘以x，这个vt来自wb乘以x，这个qt来自wq乘以x。这里的w是记忆的w，所以这是记忆的参数，也是记忆本身。我们想要更新这个w，那么该怎么做呢？我们用wv投影单个标记的信息，用wk投影它，计算这里的这个项，它只是w乘以这里的这个项，我们计算这里的这个损失，并计算它的梯度。这个损失的梯度可以用以下公式计算，他们实际上在后面有详细说明。

我可以实际演示如何推导它，这里有一个关于梯度的公式，这就是我们计算损失梯度的方法。如何计算这个公式呢？或者说如何推导它？让我们来讨论一下，不过好吧，一步一步来。他们计算的是这个损失相对于模型参数的梯度，模型的参数是什么呢？是w。所以他们计算的是这个损失相对于w的梯度，然后我们需要更新w。如何更新w呢？我们需要计算这里的这个st项，这个st项导致了帕塞价格，但我们没有任何帕塞价格。

那么假设现在这个值为零，乘以学习率，再乘以这个θ。θ_t就是学习率乘以我们计算出的这个梯度。然后我们用这个项st来更新w。现在我们已经更新了记忆，接下来要从这个记忆中检索信息。如何检索信息呢？我们直接取这个w，然后乘以x（即我们的单个标记），并用另一个称为wq的矩阵进行投影，使其变成qt。我们将qt乘以w，现在就能检索到信息。这些信息随后作为压缩的过往信息发送到模型的第一层，接着是第二层、第三层等等，以预测输出。模型会生成第一个输出标记，通常我们会将这个输出标记重新放回提示中，以生成下一个标记。

因为我们讨论的不仅仅是一个Transformer模型，而是一种混合架构，它包含注意力层和神经记忆。我们需要用这个新输入的标记来更新我们的神经记忆。这个新输入的标记将再次用于更新记忆，记忆将用新标记的信息进行更新，而不会被仅替换为这个新标记。因此，我们希望新的记忆能够包含我们之前输入的第一个标记和当前标记的信息。实际操作中，我们会获取模型输出的这个新标记，通过Wv进行投影，得到Vt；通过Wk进行投影，得到Kt。我们计算这个损失项，计算这个损失的梯度，然后像之前一样更新我们的神经记忆。但这次我们有了过去的惊喜。

因为我们不只是这样，还拥有之前的记忆，所以现在正在更新这个 w，希望它能包含关于第二个 token 和之前输入的第一个 token 的信息。正如你所看到的，由于我们是在测试时训练神经记忆，因为现在正在进行模型推理，我们希望它的表现会比仅在训练时训练的神经记忆更好。因为在训练时，某些模型可能——由于在每次更新步骤中，神经记忆实际上是在尝试最小化针对这一特定数据的损失，而不仅仅是在训练期间见过的数据，而是针对此时此刻看到的这一特定数据。我知道我向你灌输了很多信息，但我希望现在对于“内循环”和“外循环”在实际中意味着什么，应该会更清楚一些了。



so when we train the model we update the parameters of this big model to leverage whatever the memory creates and what the memory does not learn to compress information only at training time but also at inference time exactly on the data that you feed it at inference time now let's talk about the problems of this memory so the problem of this memory is that every time as you can see every time we need to run a gradient descent on each single token so this looks like it takes you need to come if when you need to train the model you have a very list big list of tokens and you want to train it as fast as possible but if you need to update the memory one token at a time it's very slow but fortunately in the paper they also propose a an algorithm to parallelize this training and this training can be parallelized actually not on the full sequence but only chunk by chunk which is still better than doing one token at a time so imagine you have one million token if we cannot parallelize it it means okay first take the first token update the memory then take the second token update the memory then third token so we need to do one million times this and we we cannot exploit our gpus because we have to do one operation at a time what they propose in the paper is a hybrid algorithm so it's not fully parallelizable on this entire sequence but chunk by chunk which is a good compromise it means that if you choose we have one million tokens and you you choose a chunk size of let's say 1000 you can parallelize the first 1000 tokens then you take the next 1000 token and you parallelize this one so in total you will compute 1000 steps not one million steps if you choose a chunk size of 1000 over a sequence length of one million they also say okay how to leverage this neural memory module you can use it as a contextual memory means that if you have a hybrid architecture in which you have attention and this neural memory so the one like the one we draw before what we can do is we take the sequence that is input by the user because the neural memory it's dropped off the neural memory is just to compress information we retrieve whatever is in the memory we append it to the sequence prepend it to the sequence along with some other persistent okay we can even not talk about the persistent memory tokens because i believe they just overdid all this stuff i mean the system could work even without the persistent memory tokens so we take our sequence we pretend whatever information is in the memory we feed it to the attention module and we use the output of the attention to update the memory and to produce the output so let's go to our architecture in this case basically it would mean imagine we have fed already 10 tokens to this memory and now we are trying to predict the 11th token what it would mean is that i would take the this 11th token i would input convert it into embeddings i would retrieve whatever is inside the neural memory so imagine the neural memory gives me in total because its job is compressing right even if i fed it 10 token it doesn't have to return me 10 tokens it has to return me a compressed version of this 10 token suppose the ratio is like suppose that the compressed state is five tokens so i would take these five tokens prepend it to my single token it will become six token i fed it to the first attention layer take the output of the attention update it and combine it with the attention output of the attention to get the output of this layer and feed it to the next one this is the neural memory as context usage the other usage is a memory as a gate which is this architecture here so in this case i have a our 11th token uh don't think about a persistent memory i believe i said it's over over it's just an overdoing you you don't have to use a persistent memory to to make this mechanism work they uh they take this 11th token they put it in the memory so now we update first the memory and they also feed it to the attention and then they combine the output of the neural memory which contains 11 token but when we retrieve it only gives us five token and then the output of the attention which we only fed one token and it's combined to produce the output or you can only use the memory as a module without any attention which means that basically you skip all this part so you take your input which could be one token one million token whatever you update the memory continuously you take the compressed version of the memory and you feed it directly to the linear layer that will produce the logits this is uh what they refer to as a memory as layer honestly you can create 1 million variants of this architecture the point is not how you use it the point is how it works so i want to punctualize how it how it works so we are training a module at test time which is different from what we do with recurrent neural networks so recurrent neural networks are trained at training time and their job is to compress data but because they do very well the job of compressing the data they have seen they may not function very well during inference because they may see some data that they have never seen however by having a memory like this that you can train at inference time and with an algorithm that is supposedly parallelizable we can avoid hopefully this problem because the only job of the memory is to be a memory to be able to retrieve so i actually like this paper because i believe that um it's a novel idea that i didn't think about before and i think it's okay this is part of a bigger um actually okay i've been researching a little bit about this area for a it's called the test time training um but this particular architecture was a little bit of innovative in this field um what else do we need to know to read this paper i think now you should have the information to read this paper because we have talked about how to update this memory and what is this memory this memory is just a linear layer in the paper they also say that okay this memory doesn't have to be just a linear layer it can be a multi-layer perceptron so it can be for example two layers with an activation in between and it will work in the same way and the algorithm that they have devised that is a parallelizable would work also with this multi-layer memory um well we didn't talk about persistent memory but the persistent memory are just the tokens that are prepended to the input so that uh and the they don't belong to the neural memory they belong to the outer loop as they call it here the outer loop is just this model here and this is the inner loop um but okay the system can work without persistent tokens this is my claim if you look at the benchmark it looks like that compared to the other architectures that are like mamba and the current neural networks it performs better if you check the average score over all these benchmarks i believe okay this is a promising area of research i will probably be looking forward to the code which has not been released yet but thank you guys for spending time with me i hope i gave you enough at least intuitions into how it is happening and i'm also really eager to look at the code because i think the best way to learn about a new architecture is actually to look at the code so have a good night

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)