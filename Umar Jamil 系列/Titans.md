
[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

好的，各位！今天我们要讨论这篇论文。在这篇论文中，我们会先探讨要解决的核心问题，然后分析提出的解决方案，最后评价其优缺点。我个人讲解论文的风格，是希望教会你们自主理解论文的方法，而不是逐字逐句复述——毕竟那种阅读你们自己也能完成。  

我更倾向于先介绍必要的背景知识，带大家打好基础，再切入问题本身，最后解析解决方案。那么，先来说说问题背景：当前深度学习中，序列建模主要有两种方式——Transformer 和 RNN。当然也存在混合变体，比如结合注意力机制与 RNN 的模型等等。  

我们具体看看这两种方式如何处理序列建模。以熟悉的语言建模为例：想象我们要训练一个语言模型，语言模型的训练通常是如何工作的呢？我们有一个序列，我们想要教会语言模型预测下一个 token。假设这是我们的 token 序列：第一个 token 是 "I"，第二个是 "like"——我总是假装一个 token 就是一个单词（虽然实际情况并非如此，但为了简单起见我们这么认为）。比如 "I like to eat pizza"。

想象我们要训练一个语言模型来生成这个确切短语，我们需要一个模型（可能是 Transformer，也可能是 RNN），我们强制它预测下一个 token。这就是序列建模的工作：我们有一个输入序列（称为input），试图将其映射到输出。我们通常训练的语言模型被称为自回归语言模型，意味着它在预测时可以使用所有过去的单词来预测下一个单词。也就是说，模型应该能准确预测这个句子——当输入 "I" 时它应该输出 "like"；输入 "I like" 时应该预测 "to"；输入 "I like to" 时应该预测 "eat"......你可以看到这个模式。当输入整个句子时，它应该输出 "EOS（end of sentence）"（这是一个表示结束的特殊 token）。

好的，生成过程到此结束。这就是我们训练语言模型的方式，我们会选取一些句子（可能来自文档、网页等任何内容），将单词逐个向后移位，并强制语言模型预测下一个 token。目前主要有两种模型可以实现这一目标：  

第一种是 Transformer。假设我们在这里引入一个 Transformer 模型，它通过注意力机制来实现语言建模。Transformer 的特点是——用于计算损失（模型训练依据）的语言模型输出可以并行处理。这也是当今大多数语言模型基于 Transformer 架构的原因：我们希望能充分利用 GPU 的并行计算能力。如果能将某些操作并行化，效率会更高。  

另一方面，我们还有 RNN（稍后会讨论 Transformer 和 RNN 各自的问题，现在先聚焦这一部分）。 Transformer 的优势在于可并行化，而另一种范式——顺便提一下，训练时移位的序列被称为目标序列（target sequence）——需要将 Transformer 的实际输出与目标序列对比，计算损失后，根据梯度反向传播以更新模型参数，因此，模型被强制学习在给定输入的情况下生成目标输出，这就是我们训练模型的方式。

我们可以把这个 Transformer 替换成 RNN，但 RNN 的问题在于它无法并行化——至少其基本形式不行。最近确实出现了一些通过并行扫描（parallel scan）技术实现并行化的 RNN 变体，但目前为止这些方法尚未投入实际应用。

关于这两种模型的区别：Transformer 的注意力机制我就不赘述了（假设大家已经了解，其实这里也不需要深入理解），只需要记住关键点—— Transformer 具有并行化能力，而基础形式的循环神经网络不具备这种特性。

RNN 的工作原理是这样的：无论是训练阶段还是推理阶段，当我们进行序列建模时（比如训练语言模型学会生成 "I like to eat pizza"这个句子），其工作流程如下：

1. 首先输入第一个 token（单词"I"）到 RNN
2. RNN 会产生一个输出（具体内容暂时未知）
3. 但我们强制要求它学习目标输出：当输入是 "I" 时，它应该预测出 "like"

因此，RNN 需要根据实际输出与目标输出（此处应为 "like"）之间的差异进行反向传播。这里的关键在于：循环神经网络不仅会生成输出 token，还会产生一个隐藏状态 —— 这个状态封装了模型迄今为止处理过的所有输入信息，相当于 RNN 的"记忆"。

让我们用更直观的方式描述这个过程：

1. 初始输入单词 "I" 输入 RNN 后，会生成新的隐藏状态（我们称为时间步 1 的隐藏状态 $H_1$）
2. 将 $H_1$ 与下一个 token "like" 一起输入 RNN 时，模型需要预测 "to"
3. 虽然当前直接输入只有 "like"，但通过前一个时间步的隐藏状态 $H_1$（其中包含 "I" 的信息），模型就能建立 "I like → to" 的关联
4. 同理，在第三个时间步：
   - 使用隐藏状态 $H_2$（包含 "I" 和 "like" 的历史信息）
   - 输入 token "to"
   - 强制模型预测 "eat"

与 Transformer 的对比：
- Transformer 预测特定 token（如 "pizza"）时，可以同时利用所有先前输入：
  * 训练时：这些输入作为键值对(key-value)并行处理
  * 推理时：形成上下文记忆(context memory)
- 这种全局可见性正是 Transformer 可并行化的根本原因——任何时候预测 token 都能看到完整序列

因此，我们将整个序列输入Transformer来预测每个位置的输出。由于Transformer能同时看到整个序列，它可以并行计算每个位置的输出。而循环神经网络(RNN)则无法并行计算每个位置的输出，必须逐步按时间步处理，因此不具备并行性。

Transformer 的核心优势就在于这种并行能力——只需增加 GPU 数量就能训练超大规模模型。而 RNN 的局限性在于：
1. 必须采用串行处理（类似 for 循环）：
   - 先计算第一个时间步
   - 然后第二个时间步
   - 依此类推...
2. 反向传播也需按时间步顺序进行
3. 这种顺序依赖性严重制约了训练效率

循环神经网络存在两个主要问题：首先它是不可并行化的（无法并行处理），其次它有一个固定大小的隐藏状态（也叫循环状态）。这个隐藏状态的大小可以任意设定（比如 1MB 或 1GB），但一旦确定架构就固定不变了。

相比之下，Transformer 模型的输入大小是不断增长的。比如当你给 ChatGPT 输入提示时：假设它是在"我喜欢吃披萨"这个句子上训练的，如果你只输入第一个词"我"，它只能基于"我"来预测；然后它会将"喜欢"这个词重新输入，这时模型看到的是"我喜欢"，就能预测下一个词；接着再把"吃"输入，模型看到"我喜欢吃"三个词后就能预测"披萨"了。

所以 Transformer 的记忆（即我们输入给它的所有 token）是在不断增长的。这带来一个问题：在做长序列建模时，我们既希望语言模型能利用所有历史输入（Transformer可以做到），但这样会导致内存不断增长；如果改用循环神经网络，虽然内存固定了，但又无法并行训练，而且固定内存还有个问题——它无法选择记住什么信息。就像让一个人记住 3000 本书是不可能的，因为人脑容量有限，循环网络也是这样。虽然有些架构（比如使用特殊 HIPPO 矩阵的 Mamba）试图改进循环网络的记忆能力，但实际效果并不理想。

现在来说说语言模型的训练过程：通常我们有一些输入 token，把它们转换成嵌入向量，然后输入到 Transformer 的各个层（第一层、第二层等），最终得到输出 logits。

关键区别在于：

- Transformer 通过键值缓存（KV cache）保存所有历史 token
- 循环网络则把所有历史压缩到一个固定大小的记忆中，这会导致信息丢失，因为我们无法控制它记住什么，只能希望网络自己学会保留重要信息、忘记次要信息。问题在于训练语言模型时需要输入海量数据。


例如，我们在整个维基百科、整个网络、大量书籍等上训练语言模型。模型几乎见过世界上所有可能的数据。然而，我们希望当我们有一个混合模型时，比如一个结合了 Transformer 和递归神经网络的模型，

假设这里有一个注意力层（Transformer 层），我们称之为注意力层，而这里是一个递归神经网络。假设这是一个可以并行化的新型递归网络架构，但问题在于，递归神经网络会产生一个固定大小的记忆。如果输入 1000 个标记，它会输出一个记忆，这个记忆会被注意力层利用，但不会是 1000 个标记，因为递归网络的目标是将信息压缩成固定大小的记忆，以便 Transformer 模型（即这里的注意力层）利用。

注意力层非常擅长利用输入的数据，但这些数据并不是整个序列，因为我们用递归神经网络压缩了它。我们希望注意力层能利用递归网络压缩的信息来预测下一个标记。如果我们这样做，想象一下我们有一个注意力加递归网络的混合架构，这种架构的问题在于，当你训练它时，由于深度学习的特性，我们强迫模型学习任何目标，它会被迫学习以某种方式压缩信息，以便注意力层可以使用它，而注意力层会被迫提取递归神经网络压缩状态中的信息。

这很好，所以当你训练它时，损失会减少，你会发现它表现得相当好。然而，当你在实际使用中输入提示时，可能不是语言模型过去见过的数据，我们称之为分布外数据。模型可能不知道如何很好地压缩它，哪些信息该保留，哪些不该保留。在这种情况下，递归网络在压缩数据的任务上会失败，因为预测下一个标记所需的数据没有被很好地压缩，注意力层将无法利用这些数据来预测下一个标记。因此，在训练时我们看到这种混合架构效果很好，但在测试时（即实际使用时），我们发现它们效果不佳。这是其中一个原因：它们学会了很好地压缩它们见过的数据。

例如，如果它看到一段长的 Python 源代码，它知道不应该关注可能重复的注释，而应该关注代码；或者当它看到 C 代码时，不应该关注括号，因为它们只是冗余的，而应该关注表达式等等。因此，它实际上学会了压缩信息，但仅限于训练时见过的信息。

现在我们可以谈谈这篇论文。论文声称我们有这些需要某种记忆的模型。在 Transformer 模型中，我们有这个 K 缓存，问题是 K 缓存会增长。K 缓存增长的问题在于它需要大量内存。实际上，大多数模型的限制在于我们无法在当前模型中拥有很大的上下文窗口，因为这些模型的推理成本非常高。因为我们需要保留 K 缓存，K 缓存是每一层都有的，对于较大的模型，它们有很多层，所以你需要为模型的每一层保留所有标记，以预测每个标记。这非常昂贵。解决这个无限增长的记忆问题的方法是使用压缩记忆，但这种压缩记忆仅在训练时效果很好。





so the claim
              
                  20:09
                  is can we have a memory module that is trained at test time and that's why we are talking about learning to memorize at test time that is effective at retrieval because the goal of the memory is to retrieve the information that is Salient that is needed by the module that is effective in retrieving the information that is being fed exactly uh at test time not only the one that it has seen at the training time this is the problem that we are trying to solve with Titans now the way they do it is as follows so they say Okay imagine we have
              
                  20:47
                  a module uh imagine we have a module that we will call M and this module Let's uh think of it as a uh layer in a module so okay let me draw actually I think it's much easier if we can draw it let's add a new paper new page so okay imagine we have a very long sequence we have seen that with the recurrent the job of the recurrent network is compress this very long sequence so that the Transformer can use it let's do with Titans Now how does it differ and then we'll check all the details so we have this input so let's
              
                  21:29
                  go here again so we have this input we transform into embeddings then we I will draw a little differently and then later I will explain why we have some suppose we have a hybrid architecture again of Transformer and recurrent layers but we I will not draw the recurrent layers so this is the first layer of the toic I think okay let's call it L1 so the first layer with attention the second layer with attention uh the third layer with attention and then we have the output which is the logits okay I think now it's more visible right
              
                  22:17
                  okay so imagine we have another module in this architecture that we will call the memory module uh let's call it neural memory because this is how the they call it here so let's call it neural memory and I will draw it as external module neural memory now I want to show you how it would work with the Neal memory and then we check the detail on how it is actually trained so the way we usually train models so imagine okay let's take a step back how would we train this model we would feed it a sequence imagine 1
              
                  23:02
                  million tokens so imagine a very big sequence so let's say 1 million tokens you convert this sequence of tokens 1 million tokens into embeddings you run this embeddings in the neural networks recurring neural network which will compress this 1 million tokens maybe in let's say 1,000 tokens because its goal is to compress stuff right so the sequence that is fed to the attention because the goal the problem of the attention is is that it's quadratic um so having a smaller input results in better computation so we feed this 1,000
              
                  23:40
                  compressed token to the attention and then we force it to predict the next token only leveraging this 1,000 compressed token so we feed 1 million token but we force the attention layer to predict the next token only leveraging much less information so we hope that the recurring L Network is good at choosing the right tokens to keep and not and discarding the one that it doesn't keep uh actually okay it's not really um token pruning mechanism it's a token compression mechanism but okay you can always you can think of it
              
                  24:14
                  as a token pruning like it's it's being fed 1 million tokens and it just keeps the top 1,000 that are the most important for predicting the next token um and this is done at a training time so we feed this 1 million token at a training time we compute the output we know what should be the next token because at training time we know what is the next token we force we computed the loss with respect to what we we think should be the next token and then we back propagate to update the parameters of the model and we keep doing it for
              
                  24:46
                  all the sequences that we have with the Titans it works it would work differently imagine you have 1 million token again and what you do is you do uh two steps the first thing that we do okay we we have this input we convert it into embeddings the first thing we do is in the training Loop so imagine we are training this Titans architecture we first train this Neal module to learn to memorize our 1 million tokens and then we ask it to retrieve the information necessary for predicting the next token and feed it to the
              
                  25:30
                  attention layer so this is a let's call it attention layer so this is an attention layer this is an attention layer and this is an attention layer so look at the difference here before we had an input we predict the output we compute the loss and we back propagate and we update all the parameters of the model here we will do something different we have an input which is 1 million tokens we convert them into magazing blah blah blah we train this model here which is separate and in the paper they refer to it as the inner loop of the training we
              
                  26:09
                  train this neural memory and later we will see how we train it with the sole purpose for this neural memory to learn everything about this data so that it can easily retrieve this data when we will need it so we take this 1 million tokens we conver them into embeddings we train this Neal memory at uh in in in an inner loop then we take this neural memory which has been trained to memorize this data and then we ask it to retrieve whatever information is important from whatever it has seen and use it as input for the attention layers here so that
              
                  26:52
                  the attention layers can leverage this compressed memory to uh produce the output and predict the next token this not only at training but also at test time so when we use the attention U with the hybrid architectures for example attention plus recurr n networks at test time so at inference time what we have is usually a prompt imagine this prompt is huge because you are asking charb for example to analyze the entire GitHub repository of a very big repository uh what will happen is that this 1 million token will be fed to the
              
                  27:29
                  recurrent which is fixed now so we are using the model so we are not changing its parameters anymore the recur Network his job is to compress data so it will compress these tokens into a smaller sequence that we will Fed to the attention layer and it will produce the output logits however maybe the information that we are feeding to this recurrent L networks are kind of out of distribution and the recurrent level network has never seen something like this and it will do probably a very bad job at compressing this data so because it will
              
                  28:04
                  do a very bad job at compressing this data because it doesn't know what to keep and what not to keep the attention layer will not be able to leverage the most important information and then it will not be able to predict the next token uh very well so it will result in a bad output and uh with uh Titans even at test time so even at inference time we are actually training a model and and now I I show you how imagine now we have again a GitHub repository and it's very big and we it results in 1 million tokens that we want
              
                  28:38
                  the language model to analyze we convert it into embeddings then we take this 1 million tokens we train on the Fly this Neal memory whose job will be to just learn as much information as possible about this this 1 million tokens retrieve the most silent information because the memory his job is to uh compress information so now then we after we have train it in this inner loop we retrieve this information we feed it to the attention layers then the attention layers should be able to ret um should be able to uh leverage the
              
                  29:13
                  information uh retrieved by the neural memory so with Titans basically we don't just uh have a uh RNN which is our memory that is trained at training time and then never trained again and every time it sees something that it has never seen it just goes crazy we have a neural memory that is can be trained at inference time on the fly with the sole purpose of compressing stuff and because we are training it at inference time uh we hope that it will perform better even on data it has never seen uh now according to the Benchmark
              
                  29:53
                  they publish in the paper but this actually happens in all papers so you never trust the benchmarks um it looks like it is doing a good job now let's look at the details so I want to remind you the problem we are solving is long context modeling long context modeling has one issue which is with the Transformer it is very expensive to inference for long context uh with rnns we have the problem that we train them on some data but when you use them on something that they have never seen they don't know how to
              
                  30:24
                  compress and how what to what to keep and what to not keep so they go crazy and because they go crazy they they they don't do this job very well the the attention layers cannot leverage this information so they just result in very bad output um with the Neal met memory we want to train on the fly a memory while inferencing the model to just do the job of compressing stuff on whatever data it is fed now we can look at the details okay um okay here they do some preliminary uh how to say um view of what is memory or what is linear
              
                  31:01
                  attention Etc we don't care about that for now they say Okay imagine we have a memory module that only has two operations one is the right operation one is the read operation um we want to write and read at inference time and also at training time to this memory how do we train this memory first of all this memory neural memory is a neural network by itself meaning that you can think of it as an external neural network that is um separated from the rest of the architecture uh that that will use this uh neural
              
                  31:42
                  memory so you you you need to think that you have like a Transformer model that is leveraging this Neal uh memory um now how to train this neural memory at inference time because that's our problem at the training time we know how to do it we just uh put the compute the output back propagate and voila how to do that at inference time it's what they they see here they say Okay imagine we have this memory uh how first of all how we want to update its information they want to update its information uh okay another step back
              
                  32:18
                  what we want this memory to do we want this memory to learn uh to extract information about whatever it they should memorize and for that they use a very particular loss which is the kind of the Reconstruction loss so imagine we have this memory if we ask it uh to memorize the uh okay imagine we have a an input sequence let's call it X this XT XT here we project it with the two linear projections called WK and W which are basically the same equivalent of that the one that we use in the attention mechanism
              
                  32:59
                  we how can this um memory do its job very well only if it learns to recreate the data it has seen uh and this is the uh the the the loss that you see here this is just the L2 uh L2 loss that you can see here uh which basically it learns to memorize the mapping between a projection called key and a projection called V of the same data so it's kinds of learns to recreate the same data um this is the job of the memory so if I put some stuff I should be able to retrieve the same stuff so I should be able to get as much as possible from the
              
                  33:41
                  stuff that I put inside how to train it how to train it is they say okay I have this memory I want to update this memory by using uh kind of a grid in descent so how gradient descent Works uh imagine we have an AAL Network the basic version of gradient descent uh work as follows so we have a a neural network with some parameters let's call them Theta so let's say Theta um the the parameters Theta at the time I so at the step I of the training are updated with the previous uh parameters of the model so at the
              
                  34:27
                  previous time minus a learning rate that we will call gamma multiplied by the gradient of the loss of of the loss with respect to the parameters of the model the gradient tells us how we should change the um the parameters in order to maximize a loss but we move against the direction of this gradient and that's why you see a sign minus so we update the parameters in the direction opposite to the one that would maximize the loss so we update the parameters to reduce the loss and this is what we do here we say we
              
                  35:10
                  want to update our memory in such a way such that we minimize this loss here which is the memorization loss which is the Reconstruction loss that we saw before so a loss that tells if I ask the memory to retrieve some information which is the key projection of the data uh it should recreate this data um and this memory is the in the paper they model it as a linear layer so in the linear layer is just a matrix multiplication with a weight Matrix so this memory module so M here uh it's nothing more than just a weight Matrix
              
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
              
            