
[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

络存在两个主要问题：首先它是不可并行化的（无法并行处理），其次它有一个固定大小的隐藏状态（也叫循环状态）。这个隐藏状态的大小可以任意设定（比如 1MB 或 1GB），但一旦确定架构就固定不变了。

相比之下，Transformer 模型的输入大小是不断增长的。比如当你给 ChatGPT 输入提示时：假设它是在"我喜欢吃披萨"这个句子上训练的，如果你只输入第一个词"我"，它只能基于"我"来预测；然后它会将"喜欢"这个词重新输入，这时模型看到的是"我喜欢"，就能预测下一个词；接着再把"吃"输入，模型看到"我喜欢吃"三个词后就能预测"披萨"了。

所以 Transformer 的记忆（即我们输入给它的所有 token）是在不断增长的。这带来一个问题：在做长序列建模时，我们既希望语言模型能利用所有历史输入（Transformer可以做到），但这样会导致内存不断增长；如果改用循环神经网络，虽然内存固定了，但又无法并行训练，而且固定内存还有个问题——它无法选择记住什么信息。就像让一个人记住 3000 本书是不可能的，因为人脑容量有限，循环网络也是这样。虽然有些架构（比如使用特殊 HIPPO 矩阵的 Mamba）试图改进循环网络的记忆能力，但实际效果并不理想。

现在来说说语言模型的训练过程：通常我们有一些输入 token，把它们转换成嵌入向量，然后输入到 Transformer 的各个层（第一层、第二层等），最终得到输出 logits。

关键区别在于：

- Transformer 通过键值缓存（KV cache）保存所有历史 token
- 循环网络则把所有历史压缩到一个固定大小的记忆中，这会导致信息丢失，因为我们无法控制它记住什么，只能希望网络自己学会保留重要信息、忘记次要信息。问题在于训练语言模型时需要输入海量数据。

所以我们，比如说，我们在整个维基百科上训练语言模型，我们也在整个网络和大量书籍上训练它，等等等等。因此，模型似乎看到了这个世界上存在的所有可能数据。然而，假设我们有一个混合模型，比如一个变换器（Transformer），但也有递归神经网络（RNN）。假设这是一个注意力层，也就是一个变换器层，我们称之为注意力层，而这个是一个递归神经网络。假设这是一个新的、可以并行化的递归网络架构。

问题在于，这个递归神经网络将产生一个固定大小的记忆。因此，如果你输入1000个标记，这个网络会输出一个记忆，但它不会是1000个标记，而是更少，因为递归网络的目标是将信息压缩成一个固定大小的记忆，以便变换器模型（也就是这个注意力层）可以利用它。注意力层非常擅长利用它被提供的数据，但这些数据并不是整个序列，因为我们用递归神经网络压缩了它。

我们希望注意力层可以利用递归神经网络压缩的信息来预测下一个标记。如果我们这样做，想象一下我们有这种架构，也就是注意力加上递归网络的混合架构。这个架构的问题在于，当你训练它时，由于深度学习的特性，我们强迫模型学习任何目标，它将被迫以某种方式压缩信息，以便注意力层可以使用它，而注意力层将被迫从递归神经网络压缩的状态中提取任何信息。

这很好，所以当你训练它时，损失降低，你会发现它表现得相当好。然而，当你在实际使用中输入提示时，模型可能没有在过去见过这样的数据，我们称之为分布外数据。模型可能不知道如何很好地压缩它，不知道该保留什么，不该保留什么。在这种情况下，递归网络将在压缩数据的任务中失败，因为预测下一个标记所需的数据没有被很好地压缩，注意力层将无法利用这些数据来预测下一个标记。

因此，在训练时，我们看到这种混合架构表现得很好，但在测试时，也就是在实际使用中，我们发现它们并不那么有效。这是原因之一：它们只学会压缩在训练时见过的信息。比如，如果它看到一段长的Python源代码，它会知道不应该关注一些可能重复的注释，而应该关注代码本身，或者在看到C代码时，不应该关注可能只是冗余的括号，而应该关注表达式等等。

它实际上学会了压缩信息，但仅限于在训练时见过的信息。现在我们可以谈谈这篇论文。论文声称，我们有这些需要某种记忆的模型。在变换器模型中，我们有这个K缓存。K缓存的问题在于它会增长。K缓存增长的问题在于它需要大量内存。实际上，大多数模型的限制在于我们无法在当前模型中拥有非常大的上下文窗口，因为这些模型的推理成本非常高。

这些模型非常昂贵，因为我们需要保留每一层的K缓存，而更大的模型有很多层，所以你需要为模型的每一层保留所有标记，以预测每个标记。因此，它非常昂贵。解决这个不断增长的无限记忆的方法是拥有一个压缩记忆，但这种压缩记忆仅在训练时效果很好。

论文的主张是，我们能否拥有一个在测试时训练的记忆模块，并且在检索时有效，因为记忆的目标是在测试时检索出模型需要的重要信息，而不仅仅是在训练时见过的信息。这就是我们试图用 Titans 解决的问题。

他们的做法如下：

他们说，好吧，想象一下我们有一个模块，我们称之为M模块。把它想象成模块中的一层。让我画一下，实际上画出来会更容易理解。假设我们有一个非常长的序列。我们已经看到递归网络的任务是压缩这个非常长的序列，以便变换器可以使用它。现在让我们看看Titans是如何不同的，然后我们会检查所有的细节。

我们有这个输入，再次来看一下，我们将其转换为嵌入。然后，我会以不同的方式画出来，稍后我会解释为什么。假设我们再次有一个变换器和递归层的混合架构，但我不会画递归层。这是第一层，我想这是第一层，我们称之为L1。第一层有注意力，第二层有注意力，第三层有注意力，然后我们有输出，即logits。我想现在更清楚了，对吧？

假设在这个架构中，我们有另一个模块，我们称之为记忆模块。我们称之为神经记忆，因为他们在这里是这样称呼的。我会把它画成一个外部模块——神经记忆。现在我想向你展示它如何与神经记忆一起工作，然后我们检查它是如何实际训练的。

通常我们如何训练模型呢？假设我们给它一个序列，想象一下100万个标记。假设一个非常大的序列，比如说100万个标记。你将这个标记序列转换为嵌入，然后在神经网络中运行这些嵌入，递归神经网络会将这100万个标记压缩成可能1000个标记，因为它的目标是压缩信息。

因为注意力的问题是它的计算复杂度是平方的，所以更小的输入会带来更好的计算效果。我们将这1000个压缩标记输入到注意力层，然后强制它仅利用这1000个压缩标记来预测下一个标记。我们输入100万个标记，但我们强制注意力层仅利用更少的信息来预测下一个标记。我们希望递归网络擅长选择保留哪些标记，而丢弃哪些标记。

实际上，这不是一个标记修剪机制，而是一个标记压缩机制。不过，你可以把它想象成一个标记修剪过程：它接收100万个标记，只保留最重要的1000个标记用于预测下一个标记。这是在训练时进行的。我们输入100万个标记，在训练时计算输出，我们知道下一个标记应该是什么，因为在训练时我们知道下一个标记是什么。我们计算损失与我们认为应该是下一个标记的差异，然后反向传播以更新模型的参数，并对我们拥有的所有序列重复这个过程。

对于Titans，它会以不同的方式工作。想象一下，你再次有100万个标记，你会进行两个步骤。首先，我们有这个输入，将其转换为嵌入。在训练循环中，想象我们正在训练这个Titans架构，我们首先训练这个神经模块来学习记忆我们的100万个标记，然后要求它检索预测下一个标记所需的信息，并将其提供给注意力层。

所以这是一个注意力层，这是一个注意力层，这是一个注意力层。看看这里的区别：之前我们有一个输入，预测输出，计算损失，反向传播并更新模型的所有参数。在这里，我们会做一些不同的事情。我们有一个输入，即100万个标记，将它们转换为嵌入，等等。

例如，我们在整个维基百科、整个网络、大量书籍等上训练语言模型。模型几乎见过世界上所有可能的数据。然而，我们希望当我们有一个混合模型时，比如一个结合了 Transformer 和递归神经网络的模型，

假设这里有一个注意力层（Transformer 层），我们称之为注意力层，而这里是一个递归神经网络。假设这是一个可以并行化的新型递归网络架构，但问题在于，递归神经网络会产生一个固定大小的记忆。如果输入 1000 个标记，它会输出一个记忆，这个记忆会被注意力层利用，但不会是 1000 个标记，因为递归网络的目标是将信息压缩成固定大小的记忆，以便 Transformer 模型（即这里的注意力层）利用。

注意力层非常擅长利用输入的数据，但这些数据并不是整个序列，因为我们用递归神经网络压缩了它。我们希望注意力层能利用递归网络压缩的信息来预测下一个标记。如果我们这样做，想象一下我们有一个注意力加递归网络的混合架构，这种架构的问题在于，当你训练它时，由于深度学习的特性，我们强迫模型学习任何目标，它会被迫学习以某种方式压缩信息，以便注意力层可以使用它，而注意力层会被迫提取递归神经网络压缩状态中的信息。

这很好，所以当你训练它时，损失会减少，你会发现它表现得相当好。然而，当你在实际使用中输入提示时，可能不是语言模型过去见过的数据，我们称之为分布外数据。模型可能不知道如何很好地压缩它，哪些信息该保留，哪些不该保留。在这种情况下，递归网络在压缩数据的任务上会失败，因为预测下一个标记所需的数据没有被很好地压缩，注意力层将无法利用这些数据来预测下一个标记。因此，在训练时我们看到这种混合架构效果很好，但在测试时（即实际使用时），我们发现它们效果不佳。这是其中一个原因：它们学会了很好地压缩它们见过的数据。

例如，如果它看到一段长的 Python 源代码，它知道不应该关注可能重复的注释，而应该关注代码；或者当它看到 C 代码时，不应该关注括号，因为它们只是冗余的，而应该关注表达式等等。因此，它实际上学会了压缩信息，但仅限于训练时见过的信息。

现在我们可以谈谈这篇论文。论文声称我们有这些需要某种记忆的模型。在 Transformer 模型中，我们有这个 K 缓存，问题是 K 缓存会增长。K 缓存增长的问题在于它需要大量内存。实际上，大多数模型的限制在于我们无法在当前模型中拥有很大的上下文窗口，因为这些模型的推理成本非常高。因为我们需要保留 K 缓存，K 缓存是每一层都有的，对于较大的模型，它们有很多层，所以你需要为模型的每一层保留所有标记，以预测每个标记。这非常昂贵。解决这个无限增长的记忆问题的方法是使用压缩记忆，但这种压缩记忆仅在训练时效果很好。

所以这个观点是，我们能不能有一个在测试时训练的记忆模块，这就是为什么我们要讨论在测试时学习记忆，这个模块要能高效检索信息，因为记忆的目标就是检索那些重要的、当前模块需要的信息，它要能有效检索那些在测试时实时输入的信息，而不仅仅是训练时见过的数据。这就是我们试图用Titans解决的问题。  

他们的做法是这样的：想象我们有一个模块，我们叫它M，这个模块可以看作是网络中的某一层。让我画出来可能更清楚，我们新建一页来画。假设我们有一个很长的序列，我们知道循环神经网络的任务就是压缩这个长序列，这样Transformer才能处理它。现在看看Titans有什么不同，然后再看细节。  

我们有这个输入，转换成嵌入向量，然后我稍微换个方式画。假设我们有一个混合架构，又是Transformer和循环层，但我不画循环层。这是Titans的第一层，叫它L1，第一层带注意力，第二层带注意力，第三层带注意力，然后输出logits。这样应该更清晰了。  

现在想象在这个架构里还有另一个模块，我们叫它记忆模块，或者神经记忆，因为论文里是这么叫的。我把它画成一个外部模块——神经记忆。现在我要展示它如何工作，然后再看具体是怎么训练的。  

通常我们训练模型的方式是：假设输入一个很长的序列，比如100万token，把它转换成嵌入向量，然后用循环神经网络压缩，比如压缩到1000个token，因为它的目标就是压缩信息。这样注意力层处理的序列就更短，因为注意力机制的计算复杂度是平方级的，输入越小计算效率越高。  

然后我们强迫模型只用这1000个压缩后的token去预测下一个token。输入是100万token，但注意力层只能基于这1000个token做预测。我们希望循环网络能学会保留重要的token，丢弃不重要的。其实它更像是一种token压缩机制，但你可以理解为token剪枝，比如从100万token里选出最重要的1000个。  

这是在训练时做的：输入100万token，计算输出，因为训练时我们知道下一个token应该是什么，所以我们可以计算损失，然后反向传播更新模型参数，对所有序列都这样训练。  

Titans的做法不同：假设还是100万token，我们分两步走。首先，输入转换成嵌入向量，然后在训练循环中做两件事


所以想象一下我们在训练这个Titans架构 我们首先训练这个Neal模块让它学会记住我们的100万个token 然后我们要求它检索预测下一个token所需的信息 并将其输入到注意力层 所以这个我们称之为注意力层 这是一个注意力层 这是一个注意力层 这是一个注意力层 看看这里的区别 之前我们有一个输入 我们预测输出 计算损失 反向传播 然后更新模型的所有参数 这里我们会做一些不同的事情 我们有一个100万个token的输入 我们将它们转换成嵌入向量等等 我们在这里训练这个单独的模型 在论文中他们称之为训练的内循环 我们训练这个神经记忆 稍后我们会看到如何训练它 唯一目的就是让这个神经记忆学习关于这些数据的所有信息 这样它就能在需要时轻松检索这些数据 所以我们获取这100万个token 将它们转换成嵌入向量 我们在内循环中训练这个神经记忆 然后我们获取这个已经训练好记住这些数据的神经记忆 然后我们要求它从它见过的所有信息中检索出任何重要的信息 并将其作为注意力层的输入 这样注意力层就可以利用这个压缩的记忆来产生输出并预测下一个token 这不仅仅是在训练时 在测试时也是如此 所以当我们使用混合架构的注意力时 比如注意力加循环神经网络 在测试时也就是推理时 我们通常有一个提示 想象这个提示非常大 因为你要求比如ChatGPT分析一个非常大的GitHub仓库的整个代码库 会发生的情况是 这100万个token会被输入到现在已经固定的循环神经网络中 所以我们是在使用模型 不再改变它的参数了 循环神经网络的工作是压缩数据 所以它会将这些token压缩成一个更短的序列 我们会将其输入到注意力层 然后它会输出logits 然而可能我们输入到这个循环神经网络的信息有些超出分布 循环神经网络从未见过类似的东西 它可能会在压缩这些数据时表现非常糟糕 因为它不知道该保留什么不该保留什么 注意力层就无法利用最重要的信息 然后它就无法很好地预测下一个token 所以会导致糟糕的输出 而使用Titans 即使在测试时也就是推理时 我们实际上是在训练一个模型 现在我来展示具体做法 想象现在我们又有一个GitHub仓库 它非常大 导致我们想让语言模型分析的100万个token 我们将其转换成嵌入向量 然后我们获取这100万个token 即时训练这个神经记忆 它的工作就是尽可能多地学习关于这100万个token的信息 检索最重要的信息 因为记忆的工作就是压缩信息 所以现在在我们在这个内循环中训练它之后 我们检索这些信息 将其输入到注意力层 然后注意力层应该能够利用神经记忆检索到的信息 所以基本上使用Titans 我们不仅仅有一个在训练时训练好之后就再也不训练的RNN作为我们的记忆 每次它看到从未见过的东西就会表现失常 我们有一个可以在推理时即时训练的神经记忆 唯一目的就是压缩东西 因为我们在推理时训练它 我们希望它即使在从未见过的数据上也能表现更好 现在根据他们在论文中发布的基准测试 不过这在所有论文中都会发生 所以你永远不要相信基准测试 看起来它现在做得不错 现在让我们看看细节 我想提醒你们我们正在解决的问题是长上下文建模 长上下文建模有一个问题 就是使用Transformer时 对长上下文进行推理非常昂贵 使用RNN时 我们遇到的问题是我们在一些数据上训练它们 但是当你用它们处理从未见过的东西时 它们不知道如何压缩 不知道该保留什么不该保留什么 所以它们会表现失常 因为它们表现失常 它们做不好这项工作 注意力层无法利用这些信息 所以它们只会产生非常糟糕的输出 使用神经记忆 我们想在推理模型的同时即时训练一个记忆 唯一目的就是压缩输入给它的任何数据 现在我们可以看看细节

好的 嗯 好的 这里他们先做了些初步的 呃 怎么说 嗯 关于记忆或线性注意力等的概述 我们现在暂时不关心这个 他们说 好吧 假设我们有个只有两种操作的内存模块 一个是写入操作 一个是读取操作 嗯 我们想在推理时和训练时都对这个内存进行读写 该如何训练这个内存呢 首先 这个神经记忆本身是个神经网络 也就是说你可以把它看作独立于其他架构的外部神经网络 嗯 那个架构会使用这个 呃 神经记忆 所以你得想象有个Transformer模型正在利用这个 呃 神经记忆 嗯 现在问题来了 怎么在推理时训练这个神经记忆 因为训练时我们知道怎么做 我们只要 呃 计算输出 反向传播 就搞定了 但在推理时该怎么做 这就是他们这里探讨的 他们说 好吧 假设我们有这个内存 呃 首先 我们想怎么更新它的信息 他们想更新内存里的信息 呃 好吧 再退一步 我们想让这个内存做什么 我们想让这个内存学会 呃 提取它该记住的任何信息 为此他们用了非常特殊的损失函数 就是那种重建损失 假设我们有这个内存 如果我们要求它记住 呃 好吧 假设我们有个输入序列 就叫它X吧 这个XT XT这里 我们用两个线性投影WK和W来映射它 基本上就相当于注意力机制里用的那种 这个 呃 内存怎样才能很好地工作呢 只有当它学会重建见过的数据时才行 呃 这就是你在这里看到的损失函数 就是这里的L2损失 嗯 它学习记住键投影和V投影之间的映射关系 所以算是学会重建相同的数据 嗯 这就是内存的职责 所以如果我存入某些东西 就应该能取出同样的东西 应该能尽可能还原存入的内容 怎么训练它呢 他们说 好吧 我有这个内存 想通过类似梯度下降的方法来更新 梯度下降怎么运作的 呃 想象我们有个普通网络 梯度下降的基本版本 呃 是这样运作的 我们有个带参数的网络 参数就叫θ吧 比如θ 嗯 在训练第i步时的参数θ 是用模型前一步的参数来更新的 所以是前一个时刻的参数 减去我们称为γ的学习率 乘以损失函数对模型参数的梯度 这个梯度告诉我们该如何改变 呃 参数才能最大化损失 但我们朝着梯度相反方向更新 所以你会看到减号 我们朝着最小化损失的方向更新参数 这里就是这么做的 我们说想这样更新内存 使得最小化这个损失 就是前面看到的记忆损失 也就是重建损失 这个损失表示 如果我让内存通过数据的键投影来检索信息 它就该重建这个数据 嗯 在论文里他们把这个内存建模成线性层




so in the linear layer is just a matrix multiplication with a weight Matrix so this memory module so M here uh it's nothing more than just a weight Matrix
              
                  35:54
                  uh of a linear layer so we are modifying ing this weight Matrix of a so the neural memory is just a matrix w we are modifying this W in such a way that it reduces the Reconstruction loss of the data um just the way we train a neural network so we train the neural network with parameters to reduce a loss and these parameters are calculated in such a way that they will result in the smallest loss possible in the same way we are doing updating this W Matrix which is which be our memory in such a way that it will result in the minimum
              
                  36:39
                  loss information possible because that's the the loss against which we are optimizing it which is the Reconstruction loss when we um and they call it the surprise so this uh gradient of the of the W Matrix which is our memory with respect to to the uh the gradient of the loss with respect to the W of this memory um they call it the surprise because the bigger the loss the bigger difficulty the model had in reconstructing its data so it means that the model is a surprised to see this data so um that's why they call it
              
                  37:20
                  surprise um if you have ever studied um um how optimizers work okay you will remember that deep learning we have this thing called momentum so usually we don't update the parameters of the model naively like this because for example sometimes we want to uh retain the uh we want to first of all we don't want the okay first of all the loss is computed with mini botch gradient descent uh um and it means that we don't compute it over all the input data set but over instances of data so like a small batch of
              
                  38:03
                  data and the direction of this gradient is actually stochastic which means that it is not the true direction of the gradient which means that it oscillates from what it um from it it oscillates so it is not indicating the true Direction imagine the true direction of the gradient is here but if we train it on the first badge maybe it's is it's in this direction maybe on the next B is in this direction maybe on the next B on this direction Etc on average it will point to the correct direction of the gradient but it will be noisy in each
              
                  38:35
                  step because we don't want to take um steps too confidently in each step of training we add this momentum term and the momentum term basically kind of creates a exponentially moving average of all the gradients so that we also keep some information about the past gradient that we have computed computed to smooth out the the the the change of the weights so that we don't take too much so it's not like we we we don't um wait each step in the same way and the the idea for them to introduce the surprise is as follows
              
                  39:15
                  they said okay if I train my um if I train my memory to recreate the data um then um it can Miss uh this um uh new data after um it sees some novel data so maybe uh there is some new data that the model should memorize but the the gradient kind of disappears after a while so the model will miss it so in order to avoid this mechanism they use the momentum just like we do when uh doing U model training um and they call it the past surprise and this past surprise is nothing more than the the term past gradient in the momentum in
              
                  40:02
                  [Music] the uh optimizers that we use for example theam Optimizer and then the momentary surprise which is the gradient with respect to the current input so rehearse what we have said so far we have this memory which is just a w Matrix that we want to optimize to uh in such a way that so we want to change this W continuous ly with every token that we receive in such a way that it's encapsulates all the information that it are in this input and we can um how do we know it can it captures all the information in this input because we ask
              
                  40:44
                  it to minimize the loss the Reconstruction loss of the input now the problem is we don't want to do this training of this Neal model just during training but we also want to do it during inference time because if we do it only during training what happens is that during inference time every time it will see some new information that it has never seen probably it will do a bad job at compressing so it will not work so how to do that at inference time we what we will do practically is as follows so at inference time we imagine we have um
              
                  41:21
                  inputs so the first input let me write some uh all these formulas actually so that we can refer to them uh here this one and I paste it here and then we also copy the loss this one okay let's learn how it would work at inference Time Imagine we have uh 1 million tokens and okay actually you know imagine we want to generate a lot of tokens and we start with one token only so the prompt is only one token what will happen is we have this one token so let's call it a one token we feed it to the model as input
              
                  42:14
                  which will be converted into embeddings which will be only one embedding and we want to train our Neal memory on this one single token so it should learn to recreate this one single token how we will do that in practice we take the uh memory first of all we take this one embedding and we project it into key and value by doing a matrix multiplication of this single token with a matrix called WK and another called WV then we um compute this this is called the retrieval of the memory and the retrieval because the memory is modeled
              
                  42:55
                  only as a w Matrix of a linear layer the retrieval of information from this memory will just be uh W multiplied by uh the input and the input actually they call it QT so it's another projection of the input through the WQ Matrix so this KT comes from w k * by X and this VT comes from W uh V mtip by X and this QT comes from WQ multi by X this W here is the W of the memory so this is the memory parameters and this is the memory so it's the parameters of the memory but it is also the memory itself we want to update this W okay so
              
                  43:43
                  how to do that so we project the information of single token with WV we project it with WK we compute this term here which is just W multiply by this term here we compute this loss here and we compute its gradient the gradient of this loss can be computed uh with the following formula they they actually specify later I can show you also how to derive it actually so there is a Formula here for the gradient this is the how we compute the gradient of the loss um how to compute this formula well how to derive it let's talk about it but
              
                  44:26
                  okay okay one St at that so uh they computed the the the gradient of this loss with respect to the parameters of the model mod what are the parameters of the model W okay so the comput the gradient of the loss of this loss with respect to W and then the we need to update W how to update w we need to compute this St term here this St term um uh results in the passer price but we don't have any passer price so let's suppose this is zero for now multiplied by a learn learning rate multiplied by the this Theta Theta T is
              
                  45:03
                  is the learning rate multiply by this gradient that we have computed and then we update this W using this term St now we have updated our memory then we retrieve information from this memory how to retrieve information this memory we just take this W and we multiply it by the we take X so our single token we project it with Matrix called the WQ so that it becomes a QT we multiply it by W and now we retrieve information this information is then sent to the first layer of the model as compressed past information then to the second to the
              
                  45:44
                  third etc etc to predict the output the model will produce the first output token then usually we put this output token back into the prompt to generate the next token here because we are not talking about just a Transformer model we are talking about a hybrid architecture that has attention layers plus neural memory we need to update our neural memory with this new incoming token so this new incoming token will again be used to update the memory the memory will be updated with the information of the new token it will not
              
                  46:19
                  be replaced with only this new token so it we hope that the new memory will encapsulate information about the first token that we've had before and the current token what we will do practically we will take this new token that was output by the model we will project it through WV and it will become VT we will project it through WK and it will become KT uh we compute this loss term we compute the gradient of this loss and we update our Neal memory like before but we have the past Surpise this time so because we are not just and we
              
                  46:56
                  also have the previous memory so we are updating this W and hopefully this will contain information about the token number two and the token number one that we've had before now as you can see because we are training the neural memory at test time because now we are inferencing the model we hope that it will perform better than a neural memory that has only be trained at com uh training time because at training time maybe some model the because um at each step of this update uh the neural memory is actually trying
              
                  47:30
                  to minimize the loss against this particular data not only the data that it has seen during training but only exactly on this particular data that is seeing exactly in this moment I know that I fed you with a lot of information but I hope now it should be a little more clear on practically what is what it means to have an inner loop and an outer loop so when we train the model we update the parameters of this big model to leverage whatever the memory creates and what the memory does not learn to compress information only at
              
                  48:06
                  training time but also at inference time exactly on the data that you feed it at inference time uh now let's talk about the problems of this memory so the problem of this memory is that every time as you can see every time we need to run a gradient descent on each single token so this looks like it takes um you need to come when you need to train the model you have a very list big list of tokens and you want to train it as fast as possible but if you need to update the memory one token at a time it's very
              
                  48:36
                  slow but fortunately in the paper they also propose a uh an algorithm to parallelize this training and this training can be parallelized actually not uh on the full sequence but only chunk by chunk which is still better than doing one token at a time so imagine you have 1 million token uh if we cannot par ize it it means okay first take the first token update the memory then take the second token update the memory then third token update the so we need to do 1 million time this and we we cannot exploit our gpus because we have
              
                  49:10
                  to do one operation at a time uh what they propose in the paper is a a a hybrid algorithm so it's not fully parallelizable on this entire sequence but chunk by chunk which is a good compromise it means that if you choose you have 1 million tokens and um uh you you choose a chunk size of let's say 1,000 um the you can parallelize the first 1,000 tokens then you take the next 1,000 token and you paraliz this one so in total you will compute 1,000 steps not 1 million steps if you choose a chunk size of 1,000 over a sequence
              
                  49:48
                  length of 1 million uh they also say okay how to leverage this neural memory module you can use it as a contextual memory means that if you have a hybrid architecture in which you have attention and um this Neal memory so the one like the one we draw before uh what we can do is we take the sequence that is uh input by the user uh because the neural memory it's jobed off the neural memory is just to uh compress information uh we retrieve whatever is in the memory we append it to the sequence prepend it to the sequence
              
                  50:23
                  along with some other persistent okay we can even not talk about the persistent memory tokens because I believe they just overdid all this stuff I mean the system could work even without the persistent memory tokens uh so um we take our sequence we prepend whatever information is in the memory we feed it to the attention module and we use the output of the attention to update the memory and to produce the output um so uh let's go to over architecture in in this case basically it would mean imagine we have fed already 10 to tokens
              
                  51:02
                  to this memory and now we are trying to predict the 11th token what it would mean is that I would take the um this 11th token I would input convert it into embeddings I would um retrieve whatever is inside the neural memory so imagine the neural memory gives me in TOT because it's job is compressing right even if I F it 10 token it doesn't have to return me 10 tokens it has to determine a compressed version of this T tokens suppose the ratio is like um suppose that the compressed state is five tokens so I would take these five
              
                  51:36
                  tokens prepend it to my single token it will become six token I fed it to the first attention layer take the output of the attention update it and combine it with the attention output of the attention to get the output of this layer and feed it to the next one this is the uh neural memory as context usage the other usage is uh memory as gate which is this architecture here so in this case I have a our 11th token uh don't think about the per persistent memory I believe I said it's over over it's just overdoing you you you don't
              
                  52:17
                  have to use persistent memory to to make this mechanism work they uh they take this 11th token they put it in the memory so now we update first the memory and they also feed it to the attention and then they combine the output of the Neal memory which contains 11 token but when we retrieve it only gives us five token and then the output of the attention which we only fed one token and it's combined to produce the output or you can only use the memory as a module without any attention which means that basically you skip all this
              
                  52:52
                  part so you take your input which could be one token 1 million talking whatever you update the memory continuously you take the compressed version of the memory and you feed it directly to the linear layer that will produce the logits this is uh what they refer to as memory as layer uh honestly you can create 1 million variants of this architecture the point is not how you use it the point is how it works so I want to punctual ASE how it to how it works so we are training a module at test time which is different from what we do with
              
                  53:29
                  recurrent neuron networks so recurrent neon networks are trained at training time and their job is to compress data uh but because they do very well the job of compressing the data they have seen they may not function very well during inference because they may see some data that they have never seen however by having a memory like this that you can train at inference time and with an algorithm that is supposedly parallelizable we can avoid hopefully this problem because the only job of the memory is to be a memory is to be able
              
                  54:03
                  to retrieve so I actually like this paper because I believe that um it's a novel idea that I didn't think about before uh and I think it's okay this is part of um bigger um actually okay I've been researching a little bit about this area for a while it's called the test time training um but this particular uh architecture uh um was um little bit of innovative in this field um what else do we need to know to read this paper uh I think now you should have the information to read this paper because we have talked about how
              
                  54:41
                  to update this memory and what is this memory this memory is just a linear layer in the paper they also say that okay this memory doesn't have to be just a linear layer it can be a multi-layer perceptron so it can be for example two layers with any activation in between and it will work in the same way and the algorithm that they have deviced that is parallelizable would work also with this multi-layer memory um well we didn't talk about persistent memory but the persistent memory are just the tokens that are
              
                  55:12
                  prepended to the input so that uh and they don't belong to the Neal memory they belong to the outer loop as they call it here the outer loop is just this model here and this is the inner Loop um but okay the the system can work without persistent tokens this is my claim if you look at the Benchmark it looks like that compared to the other architectures that are like Mamba and um the current networks it performs better if you check the average score over all these benchmarks um I believe okay this is a promising area of research I will
              
                  55:53
                  probably be looking forward to the code which has not been released yet but um thank you guys for spending time with me I hope I gave you enough at least intuitions into how it is happening and I'm also really eager to look at the code because I think the best way to learn about a new architecture is actually to look at the code so have a good night
              
            