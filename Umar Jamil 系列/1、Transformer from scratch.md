
大家好，欢迎来到关于 Transformer 的又一集节目，在本集中，我们将使用 PyTorch 从零开始构建 Transformer，从编写代码开始一步步实现，我们将构建模型，并编写训练代码、推理代码以及用于可视化注意力分数的代码。请继续关注，因为这将是一段较长的视频，但我保证，到视频结束时，您将对 Transformer 模型有深刻的理解，不仅从概念层面，还从实践层面。我们将构建一个翻译模型，这意味着我们的模型将能够实现从一种语言到另一种语言的翻译，我选择了一个名为 Opus Books 的数据集，它包含了从著名书籍中提取的句子，我选择了从英语到意大利语的翻译，因为我是意大利人，所以我能理解并判断翻译的质量是否良好，但我会向你展示如何更改语言设置，以便你可以用你选择的语言测试同一个模型，让我们开始吧。

让我们打开我们选择的 IDE 一一我非常喜欢 Visual Studio Code 一一 然后创建我们的第一个文件，即 Transformer 模型文件，好的，首先让我们看一下 Transformer 模型，这样我们就知道要先构建哪个部分，然后我们将逐一构建每个部分。

我们将首先构建的部分是输入嵌入层，如你所见，输入嵌入层将输入转换为嵌入表示，那么，什么是输入嵌入呢？正如你在我的上一期视频中记得的那样，输入嵌入层将原始
句子转换为 512 维的向量，例如，在这个句子中，"your cat is a lovely cat“，首先，我们将句子转换为输入 ID 列表，即词汇表中每个单词位置对应的数字，然后每个数字对应一个嵌入，这是一个大小为 512 的向量，所以让我们先构建这一层。

首先，我们需要导入 torch，然后创建我们的类，这是构造函数，我们需要告诉它模型的维度，即论文中向量的维度，这称为模型，我们还需要告诉它词汇表的大小，即词汇表中有多少个单词。保存这两个值，现在我们可以创建实际的嵌入层了，好的，实际上 PyTorch 已经提供了一个层，它正好能实现我们想要的功能，也就是说，给定一个数字，它每次都会提供相同的向量，而这正是嵌入层所做的，它只是数字与 512 维向量之间的一个映射，在我们这个例子中，512 就是 d_model，这是通过嵌入层完成的，即 `nn.Embedding(vocab_size, dmodel)`，让我检查一下为什么我的自动补全功能不工作。

好了，现在让我们实现前向传播方法，在嵌入层中，我们只是使用 PyTorch 提供的嵌入层来完成这个映射。所以，返回 `self.embedding`。

现在，实际上论文中有一个小细节，也就是说，让我们实际看一下论文，让我们来看看这里，如果我们查看 “3.4 Embeddings and Softmax” 部分，我们会发现在这个句子中，在嵌入层中，我们将嵌入的权重乘以模型的平方根。那么作者们做了什么呢？他们采用了这个嵌入层提供的嵌入，我提醒你，这只是一个类似字典的层，每次都将数字映射到相同的向量，而这个向量是由模型学习的，所以我们只需将其乘以 `math.sqrt(d_model)`。你还需要导入 `math` 模块，好的，现在输入嵌入层已经准备好了，让我们进入下一个模块。

接下来我们要构建的是位置编码模块，让我们快速看一下位置编码是什么，所以我们之前看到，原始句子通过嵌入层映射成了一组向量，这就是我们的嵌入结果，现在我们想要做的是，向模型传达句子中每个单词的位置信息，这是通过添加另一个与嵌入向量大小相同的向量（即大小为 512）来实现的，这个向量包含由一个稍后我会展示的公式给出的特殊值，这些值告诉模型这个特定的单词在句子中的位置，因此, 我们将创建这些称为位置嵌入的向量，并将它们添加到嵌入向量中，好的，让我们开始吧。

好的，让我们定义位置编码类，并且我们定义构造函数，好的，我们需要传递给构造函数的是 d_model，因为这是位置编码向量的大小，以及序列长度，这是句子的最大长度，因为我们需要为每个位置创建一个向量，并且还需要提供 dropout，dropout 是为了使模型不那么容易过拟合，好的，让我们实际构建一个位置编码，首先，位置编码是一个形状为 (seq_len, d_model) 的矩阵，为什么是序列长度乘以 d_model 呢？因为我们需要的向量大小是 d_model，即 512，但我们需要的数量是序列长度，因为句子的最大长度是序列长度，所以让我们开始吧。

好的，在我们创建矩阵之前一一我们知道如何创建矩阵，让我们看一下用于创建位置编码的公式，所以，让我们看一下用于创建位置编码的公式一一这是我之前视频中的幻灯片一一并看看如何构建这些向量，所以，正如你记得的，我们有一个句子，假设在这个例子中我们有三个单词，我们使用论文中的这两个公式：$$\begin{align*}
PE(pos, 2i) &= \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) \\[1.5ex]
PE(pos, 2i + 1) &= \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{align*}$$
我们创建一个大小为 512 的向量，并为每个可能的位置（最多到序列长度）创建一个向量，在向量的偶数位置应用第一个公式，在奇数位置应用第二个公式，在这种情况下，我们应用第二个公式，我实际上会简化计算，因为我在网上看到它也被简化了，因此，我们将使用对数空间进行稍微修改的计算，这是为了数值稳定性。因此，当你在对数内部应用指数函数，然后再取对数，结果是相同的数字，但数值上更稳定，因此，首先我们创建一个名为 position 的向量，它将表示单词在句子中的位置，这个向量可以从 0 到序列长度减一，所以我们实际上是创建一个形状为 (seq_len, 1) 的张量，这是错误的，OK，现在我们创建公式的分母，这些是我们看到的公式中的两个项，让我们回到幻灯片，所以我们构建的第一个张量叫做 position 一一就是这里的这个 pos，而我们构建的第二个张量是这里的分母，但我们为了数值稳定性，在对数空间中计算它，实际值会有所不同，但结果将是相同的，模型将学习这个位置编码，如果你不完全理解这部分，别担心，它只是非常特殊，可以说，这些函数向模型传递位置信息，如果你看了我之前的视频，你也会明白为什么，现在我们将这个应用于分母，并将分母应用于正弦和余弦，正如你记得的，正弦仅用于偶数位置，余弦仅用于奇数位置，所以我们需要应用两次，让我们这样做：应用每个位置都会有符号，但只有这样，每个单词都会有符号，但只有偶数维度，所以从零开始一直到结束，每次前进两个位置意味着从零开始，然后是二，然后是四，依此类推，位置乘以 div_item，然后我们对余弦做同样的事情，在这种情况下，我们从 1 开始，每次前进 2，这意味着 1,3,5, 等等，然后我们需要为这个张量添加批次维度，以便我们可以将其应用于整个句子，即所有句子批次，因为现在的形状是序列长度，2D 模型，但我们会有一个句子批次，所以我们做的是为这个 pe 添加一个新的维度，这是通过使用 unsqueeze 并在第一个位置完成的，所以它将变成形状为 (1, seq_len, d_model) 的张量，最后我们可以在这个模块的缓冲区中注册这个张量，那么模块的缓冲区是什么？让我们先做注册缓冲区，所以基本上，当你有一个张量，你想把它保存在模块内部，而不是作为一个学习参数，但你希望它在保存模块文件时被保存，你应该将其注册为缓冲区，这样，张量将随模块的状态一起保存在文件中，然后我们进行前向方法，所以，正如你之前记得的，我们需要将这个位置编码添加到句子中的每个单词，让我们来做吧，

所以我们只需做：x 等于 x 加上这个特定句子的位置编码，我们还告诉模型我们不想学习这个位置编码，因为它们是固定的，它们将始终保持不变，它们不是在训练过程中学习的，所以我们只需将其梯度需求属性设置为假，这将使这个特定的张量不被学习，然后我们应用 dropout，就是这样，这就是位置编码，让我们看看下一个模块。

好的，我们首先将构建 Transformer 的编码器部分，即这里的左侧，我们仍然需要构建多头注意力、Add 和归一化以及前馈网络，实际上，还有另一层将这个跳跃连接连接到所有这些子层，所以让我们从最简单的开始，让我们从层归一化开始，也就是这个 Add 和归一化，正如你从我之前的视频中记得的，让我们简要回顾一下层归一化，所以层归一化基本上意味着，如果你有一个包含 n 个项目的批次一一在这种情况下只有三个一一每个项目都会有一些特征，假设这些实际上是句子，每个句子由许多带有数字的单词组成，所以这是我们的三个项目，层归一化意味着我们，对于批次中的每个项目，我们独立于批次中的其他项目计算均值和方差，然后我们使用它们自己的均值和方差计算每个项目的新值，在层归一化中，我们通常还会引入一些参数，称为 $\gamma$ 和 $\beta$，有些人称之为 $\alpha$ 和 $\beta$，有些人称之为 $\alpha$ 和 偏置，好吧，这并不重要，一个是乘法的，所以它乘以每一个 x，另一个是加法的，所以它加到每一个 x 上一一 为什么？因为我们希望模型在需要放大这些值时能够放大它们，所以模型将学习以这种方式将 $\gamma$ 乘以这些值，以放大它想要放大的值，好吧，让我们来构建这个层的代码。

让我们定义层归一化类和构造函数，像往常一样，在这种情况下，我们不需要任何参数，除了我现在要展示的一个，即 $\epsilon$，通常 eps 代表 epsilon，这是一个你需要给模型的小数字，我也会向你展示为什么我们需要这个数字，在这种情况下，我们使用 10 的负 6 次方，让我们保存它，好的，这个 epsilon 是必需的，因为如果我们看幻灯片，我们会在这个公式的分母中看到这个 epsilon，所以 x 帽等于 xj 减去 μ 除以 sigma 平方加 ε 的平方根，为什么我们需要这个 epsilon？想象一下这个分母，如果 sigma 恰好为零或非常接近零，这个 x  新将变得非常大，这是我们不希望的，因为正如我们所知，CPU 或 GPU 只能表示到一定位置和尺度的数字，所以我们不希望非常大的数字或非常小的数字，因此，通常为了数值稳定性，我们也使用这个 epsilon 来避免除以零，让我们继续，所以现在让我们引入我们将用于层归一化的两个参数，一个是称为 alpha 的，它将被乘以，另一个是偏置，它将被加上，通常加法项称为偏置，它总是被加上，而 alpha 是被乘以的那个，在这种情况下，我们将使用 `nn.Parameter` 这使得参数可学习，我们也定义了偏置，这个，我想提醒你，是乘以的，而这个是加上的，让我们定义前向传播，好的，正如你记得的，我们需要计算均值和标准差或方差，我们将计算最后一个维度的标准差，所以在批次之后的一切，我们保持维度，所以这个参数 keep_dim 意味着通常 mean 会取消它所应用的维度，但我们希望保持它，然后我们只需应用我们在幻灯片上看到的公式，所以 alpha 乘以 x 减去其均值，除以标准差加 self.eps，再加上偏置，这就是我们的层归一化，好的，让我们看看接下来要构建的下一层。

我们接下来要构建的下一层是前馈层，你可以在这里看到，前馈层基本上是一个全连接层，模型在编码器和解码器中都使用它，让我们先看看论文，了解一下论文中前馈层的细节，前馈层基本上是两个矩阵，W1 和 W2，它们依次乘以这个 x，中间有 relu 和偏置，我们可以使用 PyTorch 中的线性层来实现这一点，其中我们定义第一个层为带有 W1 和 B1 的矩阵，第二个层为带有 W2 和 B2 的矩阵，中间应用 ReLU，在论文中我们也可以看到这些矩阵的维度，所以第一个是从 d_model 到 d_ff，第二个是从 d_ff 到 d_model，所以 d_ff 是 2048，d_model 是 512，让我们来构建它，

类前馈块，我们也在这个情况下构建构造函数，在构造函数中我们需要定义我们在论文中看到的这两个值，所以 d_model、dff，还有在这种情况下 dropout，我们定义第一个矩阵，即 W1 和 B1 为线性层，并且它从 d_model 到 d_ff，然后我们应用 dropout，实际上，我们定义了 dropout，然后我们定义第二个矩阵 W2 和 B2，所以让我写注释，d_ff 到 d_model，这是 w2 和 b2，为什么我们有 b2？因为实际上，正如你在这里看到的，偏置默认是 true，所以它已经为我们定义了一个偏置矩阵，好的，让我们定义前向方法，在这种情况下，我们要做的是我们有一个输入句子，它是批次的，它是一个维度为批次、序列长度和 d_model 的张量，首先我们将使用线性层将其转换为另一个张量，从批次到序列计划到 d_ff，因为如果我们应用这个线性层，它会将 d_model 转换为d_ff，然后我们应用线性层到，这将把它转换回模型，我们在中间应用 dropout，这就是我们的前馈块，让我们看看下一个块。

我们的下一个块是最重要和最有趣的，它是多头注意力，我们在上一视频中详细地看到了多头注意力是如何工作的，所以我现在将再次打开幻灯片来复习它是如何工作的，然后我们将通过编码来实际操作，如你所记，在编码器中我们有多头注意力，它三次使用编码器的输入，一次称为查询，一次称为键，一次称为值，你也可以认为这是输入的三次复制，或者简单地说，这是同一个输入应用了三次，多头注意力的基本工作原理如下：我们有一个输入序列，它是序列长度乘以 d_model，我们将其转换为三个矩阵：q、k 和 V，它们与输入完全相同，在这种情况下，因为我们讨论的是编码器，我们会发现在解码器中它略有不同一一然后我们将其乘以称为 W_q、w_k 和 w_v 的矩阵，这会产生一个新的维度为序列乘以 d_model 的矩阵，然后我们将这些矩阵分成 H 个矩阵，模矩阵，为什么是 H？因为这是我们想要的多头注意力的头数，我们沿着嵌入维度而不是序列维度分割这些矩阵，这意味着每个头将能够访问整个句子，但每个词的不同嵌入部分，我们使用这个公式对每个模矩阵应用注意力，这将给我们作为结果的较小矩阵，然后我们将它们组合回来，就像论文所说的那样，因此，将头 1 到头 H 连接起来，最后我们乘以 w_o 以得到多头注意力的输出，这同样是一个矩阵，其维度与输入矩阵相同，如你所见，多头注意力的输出在本幻灯片中也是序列乘以模型，实际上，我没有展示批量维度，因为我们正在讨论一个句子，但在我们编写 Transformer 时，我们不仅处理一个句子，而是多个句子，因此，我们需要考虑这里还有一个维度，即批量，好的，让我们来编写这个多头注意力的代码，我会稍微放慢速度，这样我们可以详细地看到每一步是如何完成的，但我真的希望你能再次了解它是如何工作的以及我们为什么要做这些，那么，让我们开始编写代码吧。

此外，在这种情况下，我们定义了构造函数以及我们需要提供给这个多头注意力的内容，作为参数，当然有模型的 d_model，在我们的例子中是 512，头数，我们在论文中称为 h，所以 h 表示我们想要多少个头，然后是 dropout 值，我们保存这些值，如你所见，我们需要将这个嵌入向量分成 H 个头，这意味着这个 d_modeI 应该能被 H 整除，否则我们无法将代表嵌入的相同向量平均分配给每个头的矩阵，所以我们确保 d_model 基本上能被 H 整除，这会进行检查，如果我们再看一遍我的幻灯片，我们可以看到 d_model 除以 h 的值被称为 d_k，正如我们在这里看到的，如果我们将 d_model 除以 h 个头，我们会得到一个新值，称为 d_k，并且为了与论文中的命名保持一致，我们也将其称为 d_k，所以 d_k 是 d_model 除以 h，好的，让我们也定义我们将用于乘以查询、键和值的矩阵以及输出矩阵 Wo，这同样是一个线性的，从 d_model 到 d_model，为什么从 d_model 到 d_model？因为如你所见，我的幻灯片中这是 d_model 乘以 d_model，所以输出将是序列乘以 d_model，所以这是 WQ，这是 wk，这是 WV，最后，我们还有一个输出矩阵，这里称为 Wo，这个 Wo 是 h 乘以 dv，乘以 d_model，所以 h 乘以 dv，dv 实际上等于 dk，因为它是 d_model 除以 h，但为什么这里称为 dv，这里称为 dk？因为这个头实际上是结果，这个头来自这个乘法，最后的乘法是乘以 V，在论文中他们称这个值为 dv，但在实际层面上它等于 dk，所以我们的 Wo 也是一个矩阵，是 d_model 乘以 d_model，因为 h 乘以 dv 等于 d_model，这是 Wo，最后，我们创建了 dropout。

让我们实现前向方法，并看看在编码过程中多头注意力是如何详细工作的，我们定义了查询、键和值，还有一个掩码，那么这个掩码是什么？掩码基本上是如果我们希望某些词不与其他词交互，我们就屏蔽它们，我们在我之前的视频中看到了，但现在让我们回到那些幻灯片，看看掩码在做什么，如你所记，当我们使用这个公式计算注意力时，即 softmax (q 乘以 kt 除以根号 dk) 再乘以 V，我们得到这个头矩阵，但在我们乘以 v 之前，只有这里的 q 乘以 k 的乘法，我们得到这个矩阵，它是每个词与每个其他词的组合，它是一个序列乘以序列的矩阵，如果我们不希望某些词与其他词交互，我们基本上将它们的值，即注意力分数，替换为非常小的值，在我们应用 softmax 之前，当我们应用 softmax 时，这些值将变为零，因为如你所记，softmax 的分子是 e 的 x 次方，所以如果 x 趋向于负无穷，即非常小的数，e 的负无穷次方将变得非常小，即非常接近零，所以我们基本上隐藏了这两个词的注意力，这就是掩码的工作，按照我的幻灯片，我们一个接一个地进行乘法，所以，如我们所记，我们首先计算查询乘以 wq，所以 self.dot wq 乘以查询给我们一个新的矩阵，称为 q'，在我的幻灯片中，我只是在这里称为查询，我们对键和值做同样的事情，让我也写下维度，所以我们从批量序列长度到 d_model，通过这个乘法，我们将得到另一个矩阵，它是批量序列长度和 d_model，你可以在幻灯片中看到，所以当我们做序列乘以 d_model 再乘以 d_model，我们得到一个与初始矩阵相同维度的新矩阵，即序列乘以 d_model，这对它们三个都是一样的，现在我们想做的是，我们想将这个查询、键和值分成更小的矩阵，以便我们可以将每个小矩阵分配给不同的头，让我们来做吧。

我们将使用 pytorch 的 view 方法进行分割，这意味着我们保持批量维度，因为我们不想分割句子，我们想将嵌入分割成各个部分，我们也想保持第二个维度，即序列，因为我们不想分割它，还有第三个维度，即 d_model，我们想将其分割成两个更小的维度，即 H 乘以 DK，所以 self.H, self.DK，如你所记，DK基本上是 d_model 除以 H，所以这个乘以这个，给你 d_model，然后我们转置，为什么要转置？因为我们更喜欢有边缘维度，而不是作为第三个维度，我们希望它成为第二个维度，这样每个头都能看到整个句子，所以我们会看到这个维度，所以序列长度乘以 dk一一让我也在这里写上注释，所以我们从批量序列长度，d_model 到批量序列长度，边缘，衰减，然后通过使用转置，我们将到批量边缘序列长度和衰减，这非常重要，因为我们希望，我们希望每个头都能观察到这些内容。所以序列长度乘以 dk，这意味着每个头我们会看到完整的句子，所以句子中的每个词，但只是嵌入的一小部分，我们对键和值做同样的事情，好的，现在我们有了这些更小的矩阵，让我回到幻灯片，这样我可以向你展示我们在哪里，

所以我们做了这个乘法，我们得到了查询、键和值，我们将其分割成更小的矩阵，现在我们需要使用这里的公式计算注意力，在我们计算注意力之前，让我们创建一个函数来计算扩展，所以如果我们创建一个新函数，也可以在以后使用，所以自注意力，让我们将其定义为静态方法，所以静态方法基本上意味着你可以在没有这个类实例的情况下，调用这个函数，你可以直接说多头注意力块点注意力，而不需要这个类的实例。我们还给了它一个 dropout 层，我们做的是我们得到衰减，什么是衰减？它是查询、键和值的最后一个维度，我们将使用这里的这个函数，让我先调用它，这样你可以理解我们将如何使用它，然后我们再定义它，所以我们希望从这个函数中得到两样东西：输出和我们想要的注意力分数，所以 softmax 注意力分数的输出，我们将这样调用它：所以我们给它查询、键、值、掩码和 dropout 层，现在让我们回到这里，所以我们有了 dk，现在我们做的是首先应用公式的前半部分，即查询乘以键的转置，除以 dk 的平方根，所以这些是我们的注意力分数：查询，矩阵乘法，所以这个 @ 表示矩阵乘法，在 PyTorch 中，我们转置最后两个维度，-2，-1 意味着转置最后两个维度，所以这将成为最后一个维度，是序列乘以序列长度乘以 dk，它将变成 dk 乘以序列长度，然后我们将其除以 √(dk)，我们在应用 softmax 之前，正如我们之前看到的，需要应用掩码，所以我们想要隐藏一些词之间的交互，我们应用掩码，然后应用 softmax，所以 softmax 会处理我们替换的值，我们如何应用掩码？我们只需将所有我们想要掩码的值替换为非常非常小的值，这样 softmax 就会将它们替换为零，所以如果定义了掩码，就应用它，这意味着基本上将所有满足此条件的值替换为这个值，我们将这样定义掩码，使得在这个值、这个表达式为真时，我们希望它被替换为这个值，稍后我们还将看到如何构建掩码，现在，姑且认为这些都是我们不想要的值，在注意力中，所以我们不希望，例如，某个词去关注未来的词，例如当我们构建解码器时，或者我们不希望填充值与其他值交互，因为它们只是为了达到序列长度而填充的词，我们将用负 10 的 9 次方替换它们，这是一个在负数范围内非常大的数，基本上代表负无穷大，然后当我们应用 softmax 时，它将被替换为零，应用到这个维度，好吧，让我写一些注释，所以在这种情况下，我们有个批次乘以边, 所以每个头都会，然后是序列长度和序列长度，如果我们也有 dropout 的话，所以如果 dropout 没有错，我们也应用 dropout，最后，正如我们在原始幻灯片中看到的，我们将 softmax 的输出与 V 矩阵相乘，所以我们返回注意力得分乘以值，以及注意力得分本身，那么我们为什么要返回一个元组呢？因为我们想要这个，当然，我们需要它用于模型，因为我们需要将其传递给下一层，但这将用于可视化，所以自注意力的输出，在这种情况下是多头注意力的输出，实际上会在这里，我们将用它来进行可视化，所以为了可视化，模型对这个特定交互给出的分数是多少？我也要在这里写一些注释，所以这里我们这样做批次，让我们回到这里，所以现在我们有多头注意力，多头注意力的输出，我们最终要做的是，好吧，让我们回到幻灯片。

首先，我们在哪里，我们计算了这里较小的矩阵，所以我们应用了 softmax，k 乘以 kt 除以 dv 的平方根，然后我们也乘以 V 一一我们可以在这里看到——我们这里的小矩阵，头一、头二、头三和头四，现在我们需要将它们组合在一起，连接，就像论文中的公式所说，最后乘以 wo，所以让我们来做吧，我们转置是因为在我们将矩阵转换为序列长度乘以之前，我们将序列长度作为第三维度，我们最初想要将它们组合在一起，因为结果张量，我们希望序列长度在第二位置，所以让我先写下来，我们想要做什么，批次，我们从这一个序列长度开始，首先我们做一个转置，然后我们想要的是这个：所以这个转置带我们到这里，然后我们做一个视图，但我们不能这样做，我们需要使用连续的，这基本上意味着 PyTorch，为了改变张量的形状，需要我们将内存设置为连续的，这样他就可以就地进行操作，减去一，并乘以 self.h 乘以 self.dk，正如你记得的，这是模型，因为我们在这里定义了 dk，模型乘以h，除以 h，最后我们将这个 x 乘以 wo，这是我们的输出矩阵，这将给我们一一我们从批次到，这就是我们的多头注意力块，

我们现在有了所有必要的成分，可以将它们全部组合在一起，我们只是漏掉了一层，让我们先去看看它，我们还需要构建最后一层，就是这里看到的连接，例如，这里我们有一些这层的输出，所以在这里加上一个带有这个连接的归一化，这一部分被发送到这里，然后这个的输出被发送到归一化，然后通过这层组合在一起，所以我们需要创建这个管理跳跃连接的层，所以我们取输入，我们让它跳过一层，我们取前一层的输出，所以在这种情况下，多头注意力，我们把它给这层，但也与这部分结合，所以让我们构建这层，我称之为残差连接，因为它基本上是一个跳跃连接，好的，让我们构建这个残差连接，

像往常一样，我们定义构造函数，在这种情况下，我们只需要一个 dropout，如你所记，脚本连接是在加和归一化与前一层之间，所以我们还需要一个归一化，这是我们之前定义的层归一化，然后我们定义前向方法和子层，这是前一层，我们所做的是我们取 x，并将其与下一层的输出结合，在这种情况下，称为子层，我们应用 dropout，所以这是加和归一化的定义，实际上，有一个细微的差别：我们首先应用归一化，然后应用子层，在论文的情况下，他们首先应用子层，然后是归一化，我看到了很多实现，其中大多数实际上是这样做的，所以我们也会坚持这一点，如你所记，这些块通过这里更大的块组合在一起，我们有 n 个这样的块，所以这个大块我们称之为编码器块，每个编码器块重复 n 次，前一个的输出发送到下一个，最后一个的输出发送到解码器，所以我们需要创建这个块，它将包含一个多头注意力，两个加和归一化，以及一个前馈，所以让我们来做吧。

我们称这个块为编码器块，因为解码器内部有三个块，而编码器只有两个，并且，如我之前所见，我们内部有自注意力块，即多头注意力，我们称之为自注意力，因为在编码器的情况下，它应用于具有三种不同角色的相同输入：查询、键和值的角色，这是我们的前馈，然后我们有一个 dropout，这是一个浮点数，然后我们定义，然后我们定义两个残差连接，我们使用模块列表，这是一种组织模块列表的方式，在这种情况下，我们需要两个，好的，让我们定义前向方法，我定义了源掩码，源掩码是什么？它是我们想要应用于编码器输入的掩码，为什么我们需要为编码器的输入设置掩码？因为我们想要隐藏填充词与其他词的交互，我们不希望填充词与其他词交互，所以我们应用掩码，让我们进行第一个残差连接，让我们回去查看视频，实际上是查看幻灯片，这样我们可以理解我们现在在做什么，所以第一个跳跃连接是这样的：x 从这里到这里，但在添加之前，我们需要先应用多头注意力，所以我们取这个 x，我们把它发送到多头注意力，同时我们也把它发送到这里，然后我们结合这两者，

所以第一个跳跃连接是在 x 之间，然后另一个 x 来自自注意力，所以这是函数，所以我将使用 lambda 定义子层，所以这基本上意味着首先应用自注意力，自注意力，其中我们给出查询、键和值是我们的 x，即我们的输入，这就是为什么它被称为自注意力，因为查询、键和值的角色是 x 本身，即输入本身，所以句子在观察自己，因此，一个句子中的每个词都在与同一句子中的其他词交互，我们将在解码器中看到这一点不同，因为我们在解码器中有交叉注意力，所以来自解码器的键在观察一一抱歉, 来自解码器的查询，在观察来自编码器的键和值，我们给它源掩码，所以这是什么？基本上，我们调用这个函数为多头注意力块的前向函数，所以我们给出查询、键、值和掩码，这将通过使用残差连接与这个结合，然后我们再次进行第二个，第二个是前馈一一我实际上需要在这里使用 lambda，然后我们返回 x，所以这意味着结合前馈和 x 本身，即前一层的输出，也就是这个，然后应用残差连接，这定义了我们的编码器块，

现在我们可以定义编码器对象，因为编码器由许多编码器块组成，根据论文，我们可以有最多 n 个，所以让我们定义编码器：我们将有多少层？我们将有 N 层，所以我们有很多层，它们一个接一个地应用，所以这是一个模块列表，最后我们将应用层归一化，所以我们一层接一层地应用，前一层的输出成为下一层的输入，最后我们应用归一化，这结束了我们对编码器的探索之旅，

让我们简要回顾一下我们所做的事情，我们已经获取了输入，目前还没有将所有块组合在一起，我们刚刚在这里构建了这个名为编码器的大块，其中包含两个较小的块，即跳跃连接，第一个跳跃连接是在多头注意力和发送到这里的这个 x 之间，第二个是在这个前馈和发送到这里的这个 x 之间，我们有 n 个这样的块，一个接一个，最后一个的输出将被发送到解码器，但在应用归一化之前，现在我们会，我们构建了解码器部分，

现在在解码器中，输出嵌入与输入嵌入相同，我的意思是，我们需要定义的类是相同的，所以我们只需要初始化两次，位置编码也是如此，我们可以使用与编码器相同的值，也用于解码器，我们需要定义的是这里的大块，它由掩码多头注意力和加法归一化组成，所以这里有一个跳跃连接, 另一个多头注意力与另一个跳跃连接，以及前馈与这里的跳跃连接，我们定义多头注意力类的方式实际上已经考虑到了掩码，所以我们不需要为解码器重新发明轮子，我们可以定义解码器块，即这里由三个子层组成的大块，然后我们使用这个 n 个解码器块来构建解码器，让我们开始吧，

首先让我们定义解码器块，解码器中，我们有自注意力，即：让我们回到这里，这是自注意力，因为我们有这个输入在多头注意力中被使用了三次，所以这被称为自注意力，因为相同的输入扮演了查询、键和值的角色，这意味着句子中的每个词都与同一句子中的其他词相匹配，但在这里的部分，我们将使用来自解码器的查询来计算注意力，而键和值将来自编码器，所以这不是自注意力，这被称为交叉注意力，因为我们正在将两种不同类型的对象交叉在一起，并以某种方式匹配它们来计算它们之间的关系，好吧，让我们定义，

这是交叉注意力块，基本上是多头注意力，但我们会给它不同的参数，这是我们的前馈层，然后我们有一个 dropout，好的，我们还需要找到残差连接，在这种情况下，我们有三个，太棒了，好的，让我们构建前向方法，它与编码器非常相似，但有一点不同，我会强调需要 x，x 是解码器的输入，但我们还需要编码器的输出，我们需要源掩码，即应用于编码器的掩码，以及目标掩码，即应用于解码器的掩码，为什么它们被称为源掩码和目标掩码？因为在这种特定情况下，我们处理的是翻译任务，所以我们有一个源语言，在这种情况下是英语，我们有一个目标语言，在我们的例子中是意大利语，所以你可以称之为编码器掩码或解码器掩码，但基本上我们有两个掩码：一个是来自编码器的，一个是来自解码器的，所以在我们的例子中我们称之为源掩码，源掩码是来自编码器的，即源语言，而目标掩码是来自解码器的，即目标语言，而且，就像之前一样，我们首先计算自注意力，这是解码器块的第一部分，其中查询、键和值是相同的输入，但带有解码器的掩码, 因为这是解码器的自注意力块，然后我们需要结合，我们需要计算交叉注意力，这是我们的第二个残差连接，我们给它 一一好的，在这种情况下，我们给它来自解码器的查询，所以 x，键和值来自编码器，以及编码器的掩码，最后是前馈块，就像之前一样，就是这样，

我们现在实际上已经有了构建解码器的所有成分，它只是这个块的 n 次重复，一个接一个，就像我们对编码器所做的那样，在这种情况下我们也会提供多层，所以层 一一这只是一个模块列表，我们最后还会有一个归一化，就像之前一样，我们将输入应用于一层，然后使用前一层的输出作为下一层的输入，嗯，每一层都是一个解码器块，所以我们需要给它 x，我们需要给它编码器输出，然后是源掩码和目标掩码，所以每一个都是这样的：我们在这里调用前向方法，所以没有什么不同，最后我们应用归一化，这就是我们的解码器，

还有一个最后的成分，我们需要有一个完整的 Transformer，让我们来看看它，我们需要的最后一个成分是这里的这个线性层，所以，正如你从我的幻灯片中记得的那样，多头注意力的输出是按 D 模块序列化的，所以在这里，如果我们不考虑批次维度，我们期望输出是按 D 模块序列化的，然而，我们希望将这些词映射回词汇表，这就是为什么我们需要这个线性层，它将嵌入转换为词汇表中的位置，我将称这个层为投影层，因为它将嵌入投影到词汇表中，让我们来构建它。

我们需要这个层的 D 模型，所以 D 模型是一个整数，还有词汇表大小，这基本上是一个从 D 模型转换到词汇表大小的线性层，所以投影层是..，让我们定义前向方法，好的，我们想要做什么？让我写一个小注释，我们想要将批次序列长度转换为批次序列长度词汇表大小，在这种情况下，我们还将应用 softmax：实际上我们将应用 log softmax 以获得数值稳定性，就像我之前展示的那样，到最后一维，就这样，这是我们的投影层，现在我们有了 Transformer 所需的所有成分，所以让我们在 Transformer 中定义我们的 Transformer 块，我们有一个编码器，也就是我们的编码器，我们有一个解码器，也就是我们的解码器，我们有一个源嵌入，为什么我们需要源嵌入和目标嵌入？因为我们处理的是多种语言，所以我们有一个源语言的输入嵌入和一个目标语言的输入嵌入，我们还有目标嵌入，然后我们有源位置和目标位置，实际上它们是一样的，然后我们有了投影层，我们刚刚保存了这个，

现在我们定义了三种方法：一种用于编码，一种用于解码，一种用于投影，我们将依次应用它们，为什么？为什么我们不直接构建一个前向方法？
> TIME UNE
7:41 PM
ENG
5/22/2023
> OUT U NE 一种用于投影. 我们将依次应用它们. 为什么? 为什么
> WSL : Ubuntu > TIME LNE apply them in succession why why we don't just build one forward method because as weonda
7:41 PM 口 通 人
5/22/2023
> OUT U NE 因为, 正如我们在推理过程中将看到的, 我们可以重用
WSL : Ubuntu > TIMELINE will see Ln 201, Col 5
Pytho
4. 10. 6(tra
7:41 PM
5/22/2023
> OUTUNE 编码器的输出. 我们不需要每次都计算它, 而且我们也倾向
> WSL: Ubuntu00
> TIMELNE
during inferencing we can reuse the output of the encoder we don't need to calculate 人
5/22/2023
7:41 PM
> OUT UNE 编码器的输出. 我们不需要每次都计算它, 而且我们也倾向
WSL: Ubuntu00
> TIMEUNE
it every time and also we prefer to keep the this output separate also for 3. 10. 6tra 人
5/22/2023
7:41 PM
> OUT U NE 于将这个输出分开保存, 也为了可视化注意力
> WSL : Ubuntu > TIME UNE visualizing the attention Ln201, Col5 S
4. 10. 6 (trans Python ENG
7:41 PM 通
Z
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME UNE
7:41 PM
5/22/2023
> OUT UNE 所以对于编码器, 我们有源, 因为我们有源语言和源掩码.
> WSL: Ubuntu10
> TIMELNE
So for the encoder we have the source of the...
because we have. the. source language 7:41 PM
5/22/2023
> OUT U NE 所以对于编码器, 我们有源, 因为我们有源语言和源掩码.
WSL : Ubuntu > TIME UNE and the source mass.
Ln 201, Col 27 通
5/22/2023
7:41 PM
> OUT U NE
> WSL : Ubuntu > TIME UNE 10
Ln201. Col 31 Spaces:4 UTF-8 LFPython 3. 10. 6(translo
Z
5/22/2023
7:41 PM
> OUT UNE 所以我们首先应用嵌入.
> WSL: Ubuntu
> TIMEUNE
1 A0
Spaces :4 UTF-8 LFPython3. 10. 6(transformer :co
7:41 PM
Z
5/22/2023
> OUT U NE 然后我们应用位置编码
WSL: Ubuntu
> TIMEUNE 通
5/22/2023
7:42 PM
> OUT U NE
WSL : Ubuntu > TIME UNE Ln203, Col11 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
Z
5/22/2023
7:42 PM
204 最后我们应用编码器.
> OUTUNE WSL: Ubuntu0
> TIMELNE
And finally we apply the encoder.
Ln204, Col9 Spaces:4 UTF-8 LFPython 3. 10. 6(transformer :co
5/22/2023
7:42 PM
204
> OUTUNE
WSL: Ubuntu0
> TIME LNE
5/22/2023
7:42 PM
204 return self. encoder(src, src_mask)
205
> OUTUNE 然后我们定义解码方法, 它接收编码器输出(即张量)
WSL: Ubuntu > TIMELINE Then we define the decode method, which takes the encoder output, which is the 5/22/2023
7:42 PM
284 return self. encoder(src, src_mask)
205
> OUTUNE 然后我们定义解码方法, 它接收编码器输出(即张量)
WSL: Ubuntu > TIME UNE
5 A0
tensor, the source mask, which is the tensor, the target, and. the. target mask.(tra 人
/22/2023
7:42 PM
return self. encode r(src, src_mask)
206
def decode(self, encoder _output, src _mask, tgt, tgt_mask)
> OUTUNE
> WSL: Ubuntu > TIMELINE 1 A0
Ln206, Col61
Spaces4 UIF-8 LF Python 3. 10. 6(transfo
5/22/2023
7:42 PM
> OUTUNE 我们做的是目标. 我们首先将目标嵌入应用于目标句子
WSL: Ubuntu10
> TIMELINE and what we do is target we first apply the target embedding. to the target sentence 7:42 PM 人
ENG
5/22/2023
> OUT U NE 我们做的是目标. 我们首先将目标嵌入应用于目标句子
WSL : Ubuntu > TIME UNE then we apply the positional encoding to the target sentence and finally with the.
5/22/2023
7:42 PM
然后对目标句子应用位置编码, 最后, 通过代码.
> OUT U NE
WSL : Ubuntu > TIME UNE
code
Ln 209, Col 9
4. 10. 6(tra
( Python 7:43 PM
ENG
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME UNE Ln 209, Col9 Spaces:4 UTF-8 LF( Python 3. 10. 6(trans
5/22/2023
7:43 PM
> OUTUNE 这基本上是解码器的前向方法, 所以我们有相同的参数顺序.
> WSL: Ubuntu00
> TIMELNE
this is basically the method the forward method of this decoder so we have the same ENG
5/22/2023
7:43 PM
207
208
tgt= self. tgt_pos(tgt)
tgt = self. tgt_embed(tgt)
209
return self. decoder (tgt, encoder _output, src _
> OUT U NE 这基本上是解码器的前向方法, 所以我们有相同的参数顺序.
WSL : Ubuntu > TIME UNE order of parameters yes finally we define the project method in which we just. apply 7:43 PM
5/22/2023
208
tgt= self. tgt_embed(tgt)
tgt= self. tgt_pos(tgt)
209 是的.
> OUTUNE
WSL: Ubuntu > TIME UNE order of parameters yes finally we define the project method. in. which we just. apply 7:43 PM 口
5/22/2023
207
208
tgt= self. tgt_embed(tgt)
tgt= self. tgt_pos(tgt)
209
return self. decoder (tgt, encoder _output, src_m
> OUTUNE 最后, 我们定义了投影方法, 在其中我们只是应用投影
> WSL: Ubuntu00
> TIME UNE order of parameters yes finally we define the project method in which we just. apply /22/2023
7:43 PM
207
208
tgt = self. tgt_embed(tgt)
tgt= self. tgt_pos(tgt)
209
return self. decoder (tgt, encoder _out put, src_m 最后, 我们定义了投影方法, 在其中我们只是应用投影
> OUTUNE
> WSL: Ubuntu
> TIMEUNE
1 A0
the Ln 212, Col9
( Pyth
7:43 PM
5/22/2023
207
208
tgt=self. tgt_pos(tgt)
tgt= self. tgt_embed(tgt)
209
return self. decoder (tgt, encoder _output, src _mask, tgt _mask )
> OUT UNE 最后, 我们定义了投影方法, 在其中我们只是应用投影
WSL: Ubuntu
> TIMEUNE
1 A0
projection so we take from the embedding to the vocabulary size ython
7:43 PM
5/22/2023
207
208
t gt= self. tgt _pos(tgt)
tgt = self. tgt_embed (tgt)
209
210
> OUTUNE
211
212
def
project(sc 从嵌入到词汇表大小
WSL: Ubuntu
> TIMEUNE
1 A0
projection so we take from the embedding to the vocabulary size yh on 3. os tarnsik
7:43 PM
5/22/2023
tgt = self. tgt _embed (tgt)
tgt= self. tgt_pos(tgt)
21
return self. decoder (tgt, encoder _output, src_mask, tgt_me
> OUTUNE
212
def project(self, x):
> WSL: Ubuntu
> TIMELNE
1 A0
Ln212, Col9
Spaces4 UTF-8 LF( Python 3. 10. 6(transfc
7:43 PM
5/22/2023
209
210
return self. decoder (tgt, encoder _output, src _mask, tgt_mask)
211
> OUTUINE 好的, 这也是我们最后需要构建的模块,
WSL: Ubuntu > TIME LNE Okay, this is also the last block we had to build.
4 UTF-8 LFPython3. 10. 6(trans ormer:co 通
5/22/2023
7:43 PM
return self. decoder (tgt, encoder _output, src _mask, tgt _mask )
def project (self, x ):
return self. projection _layer (x)
> OUTUINE
214
WSL: Ubuntu > TIME UNE
Ln214, Col1
Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/22/2023
7:43 PM
209
210
return self. decoder (tgt, encoder _output, src _mask, tgt _mask ) 但我们还没有创建一个方法来将所有这些模块组合在一起.
ef project(self, x)
> OUTUINE
WSL: Ubuntu00
> TIME UNE But we didn't make a method to combine all these blocks together. ytbon
4. 10. 6tra
ENG
7:43 PM
#( Batch, h, Seq_ Len, d_k)-->( Batch, h, Seq_ Len, Seq_ Len)
attention _scores =(query @key. transpose (-2,-1))/math. sqrt(d_k)
if mask is not None :
> OUTUNE 所以我们构建了许多模块
> WSL: Ubuntu
> TIMELNE
92
So we built many blocks.
Ln 214, Col1 Spaces4 UTF-8 LF{ Python 3. 10. 6(transfo
5/22/2023
7:43 PM
def _in it_(self, eps:float=1e*-6)-> None:
super ()._in it_()
self. alpha =nn. Panameter(torch. ones(1)# Multiplied self. epseps
> OUTUNE
self. bias=nn. Parameter (to rch. zeros(1))# Added
WSL: Ubuntu > TIME LNE
48
5/22/2023
7:43 PM
82
def attention (query, key, v @static method value, mask, dropout :nn. Drop but ): 我们需要一个方法, 给定 Transformer 的超参数, 为我们构建一个
83
> OUTUNE
WSL: Ubuntu
> TIMELNE
We need one that, given the hyperparameters of the transformer, builds for us one 人
ENG
5/22/2023
7:43 PM
单一的 Transformer, 初始化所有的编码器、解码器、嵌入等.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
single transformer, initializing all the encoder, decoder, the embedding s, etc. rtan
Z
ENG
5/22/2023
7:44 PM
> OUTUNE 所以让我们构建这个函数.
WSL: Ubuntu00
> TIMELNE
So let'sbuild this function.
Ln 214, Col1 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
7:44 PM
5/22/2023
> OUT UNE 我们称之为 build Transformer.
WSL: Ubuntu00
> TIMELNE
Let'scall it build Transformer.
Ln214, Col1 Spaces4 UTF-8 LF Python 3. 10. 6(trans
7:44 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME UNE Ln214, Col1 Spaces:4 UTF-8 LFPython 3. 10. 6(transfc
5/22/2023
7:44 PM
给定所有超参数, 它将为我们构建 Transformer, 并用一些初始值
> OUTUNE
> WSL: Ubuntu00
> TIMEUNE
that given all the hyperparameters will build the transformer for us and. also octrn 人
ENG
/22/2023
7:44 PM
给定所有超参数, 它将为我们构建 Transformer, 并用一些初始值
> OUT U NE
> WSL : Ubuntu > TIME UNE initialize the parameters with some initial values aces4 UTF-8 LF. Python ENG
5/22/2023
7:44 PM
> OUT UNE 初始化参数. 我们需要定义 Transformer 的哪些部分?
WSL : Ubuntu > TIME UNE initialize the parameters with some initial values 3. 10. 6 (tr
7:44 PM
5/22/2023
> OUT UNE 初始化参数. 我们需要定义 Transformer 的哪些部分?
WSL : Ubuntu > TIME UNE
Pytho
7:44 PM
5/22/2023
> OUT UNE 初始化参数. 我们需要定义 Transformer 白 的哪些部分?
WSL : Ubuntu > TIME UNE What do we need to define a transformer? co 3
UTF-8 LF{ Pytho
7:44 PM
5/22/2023
当然, 在这种情况下, 我们谈论的是翻译.
> OUT UNE
WSL: Ubuntu
> TIMEUNE
1 A0
For sure, in this case, we are talking about translation.
UTF-8 LFPython 3. 10. 6(trans 办 ENG
5/22/2023
7:44 PM
> OUTUNE 好的, 我们正在构建的这个模型, 我们将用于翻译, 但你
WSL: Ubuntu10
> TIMELINE Okay, this model that we are building, we will be using for translation, but you can 人
5/22/2023
7:44 PM
> OUT UNE 可以将其用于任何任务.
> WSL: Ubuntu
> TIMELNE
1 A0
use it for any task.
Spaces 4 UTF-8 LFPython3. 10. 6(trans
Ln 214, Col 23
ENG
7:44 PM 通
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE 1 A0
Ln214. Col23
Spaces4 UTF-8 LF Python 3. 10. 6(trans
ENG
5/22/2023
7:44 PM
> OUT UNE 所以我使用的命名基本上是翻译任务中常用的那些
WSL: Ubuntu
> TIMELNE
1 A0
So the naming I'm using are basically the ones used in the. translation task. 口
5/22/2023
7:44 PM
> OUT U NE
> WSL : Ubuntu > TIME LN E
Ln 214, Col 23
Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
5/22/2023
7:44 PM
> OUT U NE 稍后你可以更改命名, 但结构是相同的.
> WSL : Ubuntu > TIME LNE Later, you can change the naming, but the structure is the same.
yth on 3. o transtd
ENG
7:44 PM
5/22/2023
因此, 你可以将其用于 Transformer 适用的任何其他任务.
> OUT UNE
WSL: Ubuntu
> TIMELNE
1 A0
So you can use it for any other task for which the transformer is applicable. so tan 人
ENG
5/22/2023
7:44 PM
> OUT U NE
WSL : Ubuntu > TIME LNE 1 A0
Ln 214, Col 23
Spaces:4 UTF-8 LF ↓ Python 3. 10. 6(trans
ENG
5/22/2023
7:44 PM
所以我们首先需要的是源语言和目标语言的词汇表大小.
> OUT UNE
WSL: Ubuntu
> TIMELNE
1 A0
So the first thing we need is the vocabulary size of the source and. the target. not an
ENG
5/22/2023
7:44 PM
> OUT U NE
> WSL : Ubuntu > TIME UNE 1 A0
Ln 214. Col23
Spaces4 UTF-8 LF Python 3. 10. 6(trans 办 ENG
5/22/2023
7:44 PM
因为嵌入需要将词汇表中的标记转换为大小为512的向量.
> OUTUNE
WSL: Ubuntu > TIMELINE Because the embedding need to convert from the token of the vocabulary into a vector 人
5/22/2023
7:44 PM
因为嵌入需要将词汇表中的标记转换为大小为512的向量.
> OUTUNE
WSL: Ubuntu > TIMELINE of size 512.
( Pytho
. 10. 6(
7:45 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln214. Col 27 Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:45 PM
> OUT UNE 所以它需要知道词汇表有多大.
> WSL: Ubuntu
> TIMELINE
1 A0
So it needs to know how big is the vocabulary.
Spaces 4 UTF-8 LFPython 3. 10. 6 Ctrans 通
5/22/2023
7:45 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 1 A0
Ln214. Col 27
Spaces:4 UTF-8 LF ( Python 3. 10. 6(transf
7:45 PM
5/22/2023
> OUT UNE 所以它需要知道要创建多少个向量.
> WSL: Ubuntu
> TIMELINE
1 A0
So how many vectors it needs to create. n. col27
7:45 PM 通
Z
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE 1 A0
Ln214. Col 27
Spaces4 UTF-8 LF Python 3. 10. 6(transf
7:45 PM 口
5/22/2023
然后是目标语言.
> OUT UNE
> WSL: Ubuntu
> TIMELNE
1 A0
Then the target.
Ln 214, Col 44
Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:45 PM
> OUT U NE
> WSL : Ubuntu > TIME LN E
1 A0
Ln 214, Col 47
5/22/2023
7:45 PM
Indentation Error is instance 这也是一个整数
[e ] Not Implemented
Error
> OUT UNE
> WSL: Ubuntu
> TIMELNE
1 A0
which is also an integer.
Ln 214, Col63
Z
5/22/2023
7:45 PM
> OUT U NE
> WSL : Ubuntu > TIME LN E
1 A0
Ln 214, Col 65
5/22/2023
7:45 PM
然后我们需要告诉它源序列长度和目标序列长度.
> OUT UNE
> WSL: Ubuntu
> TIMELNE
1 A0
Then we need to tell him what is the source sequence length and the target sequence 7:45 PM
5/22/2023
然后我们需要告诉它源序列长度和目标序列长度.
> OUT U NE
WSL: Ubuntu
> TIMEUNE
length.
n214, Col. 65
Python
7:45 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME UNE 1 A0
Ln 214, Col65
Spaces4 UTF-8 LF Python 3. 10. 6(trans
5/22/2023
7:45 PM
> OUT UINE 这一点非常重要
> WSL: Ubuntu
> TIMELNE
1 A0
This is very important.
Ln 214, Col101 Spaces:4 UTF-8 LFPython 3. 10. 6(transfo 通
ENG
5/22/2023
7:45 PM
> OUT UINE 它们也可以相同.
> WSL: Ubuntu
> TIMELNE
1 A0
They could also be the same.
Ln214, Col 101 Spaces4 UTF-8 LFPython 3. 10. 6(trans fc
ENG
7:45 PM
5/22/2023
> OUT UIN E 在我们的例子中, 它们将是相同的, 但它们也可以不同.
> WSL: Ubuntu
> TIMELNE
1 A0
In our case, it will be the same, but they can also be different. u
pyhon
5/22/2023
7:45 PM
> OUT UNE
> WSL: Ubuntu
> TIMELINE
1 A0
Ln214, Col101
Spaces4 UIF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:45 PM
> OUT UNE 例如, 如果你使用的是处理两种非常不同语言的 Transformer, 比如
> WSL: Ubuntu
> TIMELINE
1 A0
For example, in case you are using the transformer that is dealing with two very ENG
/22/2023
7:45 PM
> OUT UNE 例如, 如果你使用的是处理两种非常不同语言的 Transformer, 比如
> WSL: Ubuntu
> TIMELNE
1 A0
different languages, for example, for translation, in which the. tokens needed for the 人
ENG
/22/2023
7:45 PM
> OUT U NE 翻译, 其中源语言所需的标记数远高于或远低于另一种语言
WSL : Ubuntu > TIME UNE source n214, Col 101
4 UTF-8 LFPython 3. 10. 6 (tra
7:45 PM
5/22/2023
> OUT UNE 翻译, 其中源语言所需的标记数远高于或远低于另一种语言
> TIMEUNE
7:45 PM
5/22/2023
> OUT U NE 那么你就不需要保持相同的长度.
WSL: Ubuntu
> TIMELNE 人
5/22/2023
7:45 PM
那么你就不需要保持相同的长度.
> OUT U NE
> WSL : Ubuntu > TIME LNE the same length.
aces 4 UTF-8 LF( Python3. 10. 6(trans
n 214, Col 101
ENG
7:45 PM
5/22/2023
你可以使用不同的长度.
> OUT U NE
> WSL : Ubuntu > TIME LNE You can use different lengths.
Ln214, Col101
Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:45 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE Ln214, Col101 Spaces:4 UTF-8 LF( Python 3. 10. 6 Ctrans
7:45 PM
5/22/2023
> OUT U NE 下一个超参数是解码器模块.
> WSL : Ubuntu > TIME LNE The next hyper barometer is the demo dule. d1o1
Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:45 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln214, Col102
Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
7:45 PM
5/22/2023
> OUT UNE 我们将其初始化为512, 因为我们希望保持与论文相同的值.
> WSL: Ubuntu
> TIMELINE
1 A0
which we initialize with 512, because we want to keep the same values as the paper.
7:46 PM 口 人
ENG
/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln214. Col121 Spaces:4 UIF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:46 PM
> OUT UNE 然后我们定义超参数n, 即层数, 因此我们将使用的编码器
> WSL : Ubuntu > TIMELINE i Then we define the hyper parameter n, which is the number of. layers, so the number of /22/2023
7:46 PM
> OUT UNE 然后我们定义超参数n, 即层数, 因此我们将使用的编码器
> WSL: Ubuntu
> TIMELINE
2 A0
encoder blocks and the number of decoder blocks that we will be using is, according /22/2023
7:46 PM
> OUT UNE 块和解码器块的数量, 根据论文, 是六个.
> WSL: Ubuntu
> TIMELNE
5 A0
to the paper, is six.
UTF-8 LFPython3. 10. 6(trans
Ln 214, Col 130
ENG
7:46 PM
5/22/2023
> OUT UNE
> WSL: Ubuntu
> TIMELNE
5 A0
Ln214, Col131 Spaces:4 UTF-8 LFPython 3. 10. 6(transfo
ENG
5/22/2023
7:46 PM
然后我们定义超参数h, 即我们想要的头数, 根据论文
> OUTUNE WSL: Ubuntu10
> TIMELNE
Then we define the hyper parameter h, which is the number of heads we want, and 人
/22/2023
7:46 PM
然后我们定义超参数h, 即我们想要的头数, 根据论文
> OUTUNE
> WSL: Ubuntu
> TIMELNE
6 A0
according to the paper, it is eight. n214. co136 sp
( Pyth
7:46 PM
5/22/2023
Keyboard Interrupt Indentation Error is instance ] Not Implemented 它是八个.
T Not Implemented Error > OUT UNE
> WSL: Ubuntu
> TIMELINE
6 A0
according to the paper, it is eight. in214. col 39 Spaces4 ufs u F ( python 3. 106(tasto
P
ENG
5/22/2023
7:46 PM
> OUT UNE dropout 率为 0. 1.
> WSL: Ubuntu
> TIMELINE
1 A0
The dropout is 0. 1.
Ln 214, Col143 Spaces:4 UTF-8 LF Python 3. 10. 6(transfo
P
ENG
7:46 PN
> OUT UNE 最后, 我们有一个前馈层的隐藏层dff, 如我们在论文中看到
WSL: Ubuntu10
> TIMEUNE
and finally we have the hidden layer dff of the feed forward. layer which is 2048as 通
Z
/22/2023
7:46 PM
> OUT UNE 最后, 我们有一个前馈层的隐藏层dff, 如我们在论文中看到
WSL: Ubuntu
> TIMELNE
1 A0
we saw before on the paper and this builds a transformer okay so first we do. iswe 通
/22/2023
7:46 PM
> OUTUNE 的, 它是2048. 这样就构建了一个 Transformer. 首先, 我们创建
WSL: Ubuntu10
> TIMELNE
we sawbefore on the paper and this builds a transformer okay so first we doiswe 人
ENG
/22/2023
7:46 PM
的, 它是 2048. 这样就构建了一个 Transformer. 首先, 我们创建
> OUT UNE
> WSL: Ubuntu
> TIMELINE
1 A0
Create the embedding layers so source embedding acsa ufrsu ython
的, 它是 2048. 这样就构建了一个 Transformer. 首先, 我们创建
> OUT U NE
> WSL: Ubuntu
> TIMELNE
n216, Col 15
> OUT U NE 嵌入层, 即源嵌入和目标嵌入.
> WSL Ubuntu > TIME LNE Then the target embedding.
aces4 UTF-8 LFPython 3. 10. 6(trans 通 办 ENG
5/22/2023
7:47 PM
[o]tgt_vocab_size
> OUTUNE
> WSL: Ubuntu > TIME LNE
Ln 217, Col7 Spaces4 UTF-8 LF( Python 3. 10. 6(transfc 办 ENG
5/22/2023
7:47 PM
219
> OUTUNE 然后我们创建位置编码层.
> WSL: Ubuntu > TIMELINE Then we create the positional encoding layers.
Spaces :4 UTF-8 LFPython 3. 10. 6(transfo
7:47 PM 通
5/22/2023
219
> OUTUNE
> WSL: Ubuntu > TIME UNE
Ln 219, Col7 Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :co
ENG
5/22/2023
7:47 PM
219
220
# Create the positional encoding layers src _pos = Positional Encoding (d _model, srq )
Script Module.
> OUTUNE 我们不需要创建两个位置编码层, 因为实际上它们做的是同样
> TIMELNE
/22/2023
7:47 PM
220
src_pos = Positional Encoding (d _model, src 的工
c_seq_len
> OUTUNE
TFc Oocab_size > WSL : Ubuntu > TIMELINE o job they and we they also don't add any parameter but because. they have the drop out 7:47 PM
219
220
# Create the positional encoding layers Script Module. 它们也不会增加任何参数, 但由于它们有dropout, 而且我也想让
> OUT UNE > TIMELINE 人
7:47 PM
219
220
# Create the positional encoding layers src _pos = Positional Encoding (d _model, src _seq _len )
Script Module.
> OUT UNE 它们也不会增加任何参数, 但由于它们有dropout, 而且我也想让
WSL : Ubuntu > TIMELINE and also Ln 220, Col 54
ENG
/22/2023
7:47 PM
219
220
# Create the positional encoding layers src _pos = Positional Encoding (d _model, src _seq _len )
Script Module. 它们也不会增加任何参数, 但由于它们有dropout, 而且我也想让
> OUTUNE
> WSL: Ubuntu00
> TIMELINE because i want to make it verbal so you can understand each part without making any 人
/22/2023
7:47 PM
219
220
# Create the positional encoding layers src _pos = Positional Encoding (d _model, src _seq _len )
Script Module.
> OUT UNE 它更直观, 这样你可以在不做任何优化的情况下理解每个
> TIMELINE 7:47 PM
219
220
src_pos= Positional Encoding (d _model, src _seq _len )
Script Module.
> OUTUNE 部分, 我认为这实际上是可以的, 因为这是出于教育目的
WSL: Ubuntu00
> TIMELINE optimization i think actually it's fine because this is for educational purpose soi 人
7:47 PM
219
220
Script Module.
> OUTUNE 所以我不想优化代码, 我只想让它尽可能易懂. 所以我做了
WSL: Ubuntu00
> TIMELNE
optimization i think actually it's fine because this is for educational purpose soi 人
/22/2023
7:47 PM
219
220
# Create the positional encoding layers Script Module.
> OUT U NE 我需要的每一个部分, 没有走捷径,
> WSL : Ubuntu > TIME LNE to optimize the code i want to make it as much comprehensible as possible soi do
ENG
7:47 PM
219
220
# Create the positional encoding layers Script Module.
> OUT U NE 我需要的每一个部分, 没有走捷径
> WSL : Ubuntu > TIME LNE every part i need i don't take shortcut S 2o. cos5s
4 UIF-8 LEPython 3. 10. 6tran
7:47 PM
5/22/2023
220
src_pos= Positional Encoding (d _model, src _seq _len, drop ou )
# Create the positional encoding layers [e ]dropout =
[e ]dropout > OUT U NE
> WSL Ubuntu > TIMELINE Ln220, Col61 Space:4 UTF-8 LF( Python 3. 10. 6 (trans
Z 心
ENG
5/22/2023
7:47 PM
22
src_pos = Positional Encoding (d _model, src_seq_len, dropout)
tgt_pos = Positional Encoding (d _model, tgt _seq _len,
dropout )
> OUT U NE
WSL : Ubuntu > TIME UNE Ln 223, Col5 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
22
tgt_pos= Positional Encoding (d _model, tgt_seq_len, dropout )
223
222 我们有 n 个, 所以让我们定义.
> OU TUNE
WSL : Ubuntu > TIME UNE We have nof them, so let's define.
Ln223, Col7 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
22
src_pos = Positional Encoding (d _model, src_seq_len, dropout)
tgt_pos = Positional Encoding (d _model, tgt_seq_l en, dropout)
223
> OUTUNE
WSL: Ubuntu > TIME LNE
Ln223, Col8
Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
220
221
tgt_pos
= Positional Encoding (d _model, tg t_seq_len, 让我们创建一个空数组.
dropout)
> OUTUINE
224
# Create encoder _bl
WSL : Ubuntu > TIME LNE Let's create an empty array.
Ln224, Col 24
Spaces:4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
src _pos = Positional Encoding (d _model, src_seq_len, dropout)
2
tgt_pos = Positional Encoding (d _model, tgt_seq_len, dropout)
224
# Create the encoder blocks encoder _blocks =[]
> OUTUINE
225
WSL: Ubuntu
> TIMELNE
Ln225, Col5
Spaces4 UTF-8 LF { Python 3. 10. 6(transfo
ENG
5/22/2023
7:48 PM
221
tgt_pos= Positional Encoding (d _model, tgt_seq_l en, dropout) 所以我们有n个, 每个编码器块都有自注意力机制.
> OUTUINE
> WSL: Ubuntu > TIME LNE So we have n of them, so each encoder block has self-attention.
Python 7:48 PM
5/22/2023
221
tgt_pos = Positional Encoding (d _model, tgt_seq_len, dropout)
225
224
encoder _blocks =[]
# Create the encoder blocks for _in range ( N ):
> OUT UNE
226
WSL: Ubuntu
> TIMEUNE
1 A0
Ln226, Col9
ENG
5/22/2023
7:48 PM
221
222
tgt_pos =
Positional Encoding (d model, tgt seq len, dropou
[]encoder _blocks 223
224
encoder_blo
# Create th
> OUTUNE
225
for 所以编码器自注意力.
> TIMEUNE
226
encoder l WSL: Ubuntu So encoder self-attention.
Ln 226, Col 16
Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
221
222
tgt_pos = Positional Encoding (d _model, tgt_seq_len, dropout)
224
223
encoder _blocks =[]
# Create the encoder blocks > OUT UNE 225
for_in range( N):
> TIMEUNE
226
encoder self attention ↓
WSL : Ubuntu Ln226, Col32
Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
7:48 PM
221
tgt_pos = Positional Encoding (d _model, tgt_seq_len, dropout)
224
# Create the 这是一个多头注意力块.
> OUTUNE
226
for
WSL: Ubuntu > TIME UNE which is a multi-head attention block. n22. col#
ENG
7:48 PM
P
5/22/2023
221
tgt_pos = Positional Encoding (d _model, tgt _seq _len, dropout )
(d _model :in t, h:int, dropout:float)-> None
224
encoder _blocks =[]
# Create the encoder blocks In itil izes ntma Module stat, shared by both n. Module nd
> OUTUNE
225
for_in range( N):
Script Module.
> TIME UNE
226
encoder _self _attention _block = Multi Head Attention Block (
WSL: Ubuntu
Ln226, Col64
Spaces 4 UTF-8 LFPython3. 10. 6(transfo
P
ENG
5/22/2023
7:48 PM
222
221
tgt_pos = Positional Encoding (d _model, tgt _seq _len, dropout )
blocks oth nn. Module and > OUT UNE 多头注意力需要dmodel 、h 和dropout WSL : Ubuntu > TIME UNE The multi-head attention requires the d model, the etch and the drop out value. ctrar
5/22/2023
7:48 PM
222
221
tgt_pos = Positional Encoding (d _model, tgt _seq _len, dropout )
(d _model :in t, h:int, dropout:float)-> None
223
224
encoder _blocks =[]
# Create the encoder blocks > OUTUNE
225
for_in range( N):
Script Module.
> TIME UNE
226
encoder _self _attention _block = Multi Head Attention Block (d _model, h, dropout )
> WSL: Ubuntu
2 A0
Ln226, Col83
Spaces 4 UTF-8 LFPython3. 10. 6 Ctransformer :conda
Z
5/22/2023
7:48 PM
function False # Cre
enco
ard Bloc
226 然后我们有一个前馈块.*rpt)
> OUTUNE
227
WSL: Ubuntu > TIME LNE
ENG
7:48 PM
5/22/2023
224 # Create the encoder blocks encoder _b
ocks[]
> OUTUNE
226
enco File Not Found Error el, h, dropout)
> TIMELNE
227
feed
> WSL: Ubuntu
0 A1
Ln227, Col13
Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/22/2023
7:48 PM
223
224
# Create the blocks encoder _blocks =[]
> OUT U NE 正如你所见, 我使用的名称相当长, 主要是为了尽可能让
> WSL : Ubuntu > TIME UNE o. As you can see also the name s I'musing are quite long, mostly because l want to make 人
7:49 PM
224
223
# Create the blocks 我使用的名称相当长, 主要是为了尽可能让
e( N):
> OUT U NE 正如你所见,
WSL : Ubuntu > TIME UNE it as comprehensible as possible for everyone.
/22/2023
7:49 PM
224
223
# Create the encoder blocks 225 encoder _blocks =[]
for _in range ( N ):
encoder_sel > OUT UNE
feed_forwa
> TIMEUNE
228
it as comprehensible as possible for everyone.
WSL: Ubuntu
Spaces4 UTF-8 LF( Python 3. 10. 6(transfc
ENG
5/22/2023
7:49 PM
# Create th e encoder
block s
encoder_blocks=[]
for _in range( N):
encoder_self_attention_block = Multi Head Attention Block(d_model, h, dropout)
feed_forward_block = Feed F ordward Block(d_model, d_f, dropout )
> OUT UNE
228
WSL: Ubuntu
> TIMELNE
Ln 228, Col9
ENG
7:49 PM
5/22/2023
223
224
# Create the encc
blocks feed_forward_block : Feed For d
rd Block, dropout :
float )
> None 所以每个编码器块由一个自注意力和一个前馈组成.
> OUT UNE
WSL: Ubuntu
> TIMELNE
1 A0
So each encoder block is made of a self-attention and a feedforward.
3. 10. 6(tra
7:49 PM
5/22/2023
223 # Create the encoder blocks feed_forward_block : Feed For dward Block, dropout :float )
-> None encoder _blocks =[]
for _in range ( N ):
Initializes in teal Module state ha red by both nn. Modle and encoder _self _attention _block = Multi Head Attention Block (d_mo
feed_forward_block = Feed For dward Block (d _model, d_ff, drop o Script Module.
> OUTUNE
228
encoder _block = Encoder Block (encoder _self _attention _block,
WSL : Ubuntu > TIME LN E
Ln 228, Col68
Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
7:49 PM
5/22/2023
223
224
# Create the blocks feed_forward_block : Feed For dward Block, dropout :float)
225
ge( N):
> None 最后, 我们告诉它dropout率是多少
s internal Module state, shared by both nn. Module and > OUT U NE
WSL : Ubuntu > TIME LNE And finally, we tell him how much is the dropout. p
paces4 UTF-8 LF{ Python3. 10. 6(trans l
ENG
7:49 PM 通
5/22/2023
224 # Create th e encoder block s
Decoder Block
oeprecation Narning
encoder _block s =[]
for
_in range( N):
Zero Division Errnr
encoder _self _attention _block = Multi Head Attention Block (d_model, h, dropout)
feed_forw ard_block = Feed Fordward Block(d_model, d_ff, dropout)
unicodeoecode Erron
Pendsngeprecationlarning
> OUTUNE
228
encoder_block= Encoder Block(encoder_self_attention_block, feed_forward_block, dor)
> WSL : Ubuntu > TIME LN E
Ln 228, Col 91
Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
5/22/2023
7:49 PM
For _in range ( N ):
self _attention _block Multi l
(dmodel, h, drc
228 最后, 我们添加这个编码器块d
> OUTUNE
229
opout)
> WSL : Ubuntu > TIME LNE Finally, we add this encoder block.
Ln 229, Col9
4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:49 PM
224
encode(e) Environment Error 2 Unicode Encode Error For en2s Exception odel, h, dropout)
228
fe Potitional co coding en Bose Exception out )
feed_forward_block, dropout )
> OUTUNE
229
end
WSL: Ubuntu > TIMELINE Ln229, Col11
Spaces4 UTF-8 LFPython 3. 10. 6(transfc
ENG
5/22/2023
7:49 PM
feed_forward_block = Feed For dward Block (d _model, d_ff, dropout )
encoder _self _attention _block = Multi Head Atter lock(d_model, h, drop out) 然后我们可以创建解码器块
er Block (enc
elf _attention _block,
feed_for
ward_block, dropout )
> OUTUNE
230
> TIMEUNE
231
And then we can create the decoder blocks. caos > WSL: Ubuntu
ENG
5/22/2023
7:49 PM
encoder _self _attention _block = Multi Head Attention Block (d _model, h, dropout )
feed_forward_block = Feed For dward Block (d _model, d_ff, dropout )
encoder _blocks. append (encoder _block )
encoder _block = Encoder Block (encoder _self _attention _block, feed_forward_block, dropout )
> OUT UNE
230
> TIMEUNE
231
Crea
WSL: Ubuntu Ln 231, Col11
ENG
5/22/2023
7:49 PM
231
230
# Create the decoder blocks 232
> OUTUNE 我们也有用于解码器块的交叉注意力.
WSL: Ubuntu > TIME UNE We also have the cross attention for the decoder block.
UTF-8 LFPython 3. 10. 6(trans 通
ENG
5/22/2023
7:50 PM
# Create the decoder blocks decoder _blocks =[]
234
for_in range( N):
decoder _self _attention _block = Multi Head Attention Block (d _model, h, dropout)
> OUTUNE
235
> WSL: Ubuntu > TIME UNE
Ln 235. Col9
ec4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:50 PM
231
232
# Create the decoder blocks 233 decoder _blocks =[]
ge( N):
234
235 我们也有前馈, 就像编码器一样.
> OUTUNE
236
> WSL: Ubuntu > TIME LNE We also have the feed forward, just like the encoder. s4uifs ython 3. io6ctrans o
ENG
5/22/2023
7:50 PM
231
232
# Cre
sfrozen set function 233
decod
or
False
235
234
d File Exists Error Feed For dward Block odel, h, dropout)
> OUTUNE
236
model, h, dropout )
> WSL : Ubuntu > TIME UNE Ln 236, Col 10
Spaces4 UTF-8 LF(↓ Python 3. 10. 6(transfo 人
5/22/2023
7:50 PM
233
232
for_in range( N):
> OUTUNE 然后我们定义解码器块本身:包括解码器块交叉注意力和最终
> WSL : Ubuntu > TIME LNE then we define the decoder block itself which is decoder block cross attention and. 人
/22/2023
7:50 PM
233
232
decoder _blocks =[]
[]feed_forward_block Feed For dward Block
234
_in range( N):
deco _self _attention _blo File Exists Er 的前馈与dropout 最后将其保存到数组中. 现在我们可以创建
> OUTUNE
WSL: Ubuntu00
> TIMELNE
finally the feedforward and the drop out and finally we save it in its array.
3. 10. 6ta 人
5/22/2023
7:50 PM
234
233
For
in range( N):
239
decoder _self _attention _block 的前馈与dropout, 最后将其保存到数组中现在我们可以创建
> OUT U NE
> WSL : Ubuntu > TIME UNE
7:50 PM
/22/2023
234
233
for
in range ( N):
decode r_self_attention_block= Multi
decoder _cross_attention_bloc mode1h dropout)
dropout)
feed_forwar
decoder_blo 扁码器和解码器了
the
e list
> OUTUNE
238
decoder_blo
cross _attention _block, feed_forward_block, dropout )
WSL : Ubuntu > TIME UNE
aces4 UTF-8 LF ( Python3. 10. 6(trans
Ln 238, Col 44
7:51 PM 人
ENG
5/22/2023
decoder _cross _attention _block = Multi Head Attention Block (d _model, h, dropout )
feed_forward_block = Feed For dward Block (d _model, d_ff, dropout )
decoder _blc decoder _cross _attention _block, feed_forward_block, dropout )
> OUTUNE 编码器和解码器了
> TIMEUNE
240
We now can create the encoder and the decoder. spaes. 4 ufr-su ( yhon 31o tarsk
WSL: Ubuntu 办 ENG
5/22/2023
7:51 PM
decoder _cross _attention _block = Multi Head Attention Block (d _model, h, dropout )
feed_forward_block = Feed For dward Block (d_model, d_ff, dropout)
238
decoder _block = Decoder Block (decoder _self _attention _block, decoder _cross _attention _block, feed_forward_block, dropout )
decoder _blocks. append (decoder _block )
> OUT UNE
239
> TIMEUNE
240
> WSLUbuntu Ln240, Col5 Spaces:4 UTF-8 LF( Python 3. 10. 6 (trans
Z
P 办 ENG
5/22/2023
7:51 PM
238
237
decoder _block = Decoder Block (decoder _self _attention _block, decoder _cross _attention _block, feed_forward_block, dropout )
decoder _blocks. append (decoder _block ) 我们给它所有的块, 包括 N和解码器
> OUTU NE
WSL : Ubuntu > TIME LNE We give him all his blocks, which are N and then also. the decoder. o ython 3. ios tansto ENG
7:51 PM
5/22/2023
d[e)de code r_cross_attention_block
d[e)decoder _self _attention _block, decoder _cross _attention _block, feed_forward_block, dropout )
Decoder > OUTUNE
241
encodsunicode Decode Error > TIME LNE
242
decod
WSL: Ubuntu Ln242, Col10
Spaces:4 UTF-8 LF( Python 3. 10. 6 (trans fomer:cond)
5/22/2023
7:51 PM
239
240
241
Create the encoder and the decoder > OUT U NE 我们创建一个投影层, 将模型转换为词汇大小, 具体是哪个
> WSL : Ubuntu > TIMELINE and we create the projection layer which will convert the model into vocabulary size 7:51 PM
5/22/2023
241
240
Create the der and the decode 42
encoder= Encoder(nn. Module List (encoder _bloc (d 我们创建一个投影层将模型转换为词汇大小, 具体是哪个
> OUTU NE
WSL : Ubuntu > TIME UNE which vocabulary?
Ln 245, Col 49
Pyth
7:51 PM
5/22/2023
# Create the encoder and the decoder encoder = Encoder (nn. Module List (encoder _bloc (d _model :in t, vocab_size:int)-> None
243
decoder = Decoder (nn. Module List (decoder _bloc niti a lies in tral Module state shared by both n Module and 244 # Create the projection layer 词汇
pt Module.
> OUTUNE
245
projection _layer = Projection Layer > WSL : Ubuntu > TIME UNE which vocabulary?
Ln245., Col49 Spaces:4 UTF-8 LF(↓ Python 3. 10. 6 (transfo
5/22/2023
7:51 PM
241
240
Create the der and the decode encoder = Encoder (nn. Module List (encoder _bloc (d_ 当然是目标语言因为我们想从源语言转换到目标语言
> OUTU NE
> WSL : Ubuntu > TIME UNE of course the target because we want to take from the source language ito the target.
5/22/2023
7:51 PM
241
240
Create the enc
der and the decode l encoder = Encoder (nn. Module List (encoder _blocks ))
(nn. Module List (dec coder _blocks (d _model :int, vocab _size :in t)-> Nc
> OUT UNE 所以我们希望将输出投影到目标词汇中. 然后我们构建 Transformer,
> WSL : Ubuntu > TIMELINE language so we want to project our output into the target vocabulary and then we 2 ENG
5/22/2023
7:52 PM
243
242
decoder = Decoder(nn. Module List (decoder _blocks )) 所以我们希望将输出投影到自标词汇中. 然后我们构建 Transformer,
> OUT U NE
> WSL : Ubuntu > TIMELINE build the transformer Ln 247, Col5 Spa
4 UTF-8 LFPython ENG
/22/2023
7:52 PM
24
[e]transformer
245
projection _layer =
# Create the projec
247
2. 46
# Create the trans t 它需要什么呢?
> OUTUNE
248
transformer = t cans WSL : Ubuntu > TIME UNE Ln 248, Col 24
Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:52 PM
244
243
# Create the projection lay projection _layer : Projection Layer )-> None Positional Encoding, tgt _pos : Positional Encoding,
> OUT U NE 它需要一个编码器一 个解码器、源嵌入、目标嵌入
> WSL : Ubuntu > TIME UNE It needs an encoder, a decoder, source embedding, target embedding, then source 7:52 PM
5/22/2023
244
243
Positional Encoding, tgt _pos : Positional Encoding,
projection _layer : Projection Layer )-> None # Create the projection layer Project io
l el, tgt_vocab_size) 然后是源位置编码、, 目标位置编码, 最后是投影层
> OUTUNE
WSL: Ubuntu > TIME LNE positional encoding, target positional encoding, and finally the projection layer.
7:52 PM
5/22/2023
Positional Encoding, tgt _pos : Positional Encoding,
projection _layer : Projection L. ayer)-> None projection _layer = Projection Layer (d _model, tgt _vocab _size )
# Create the projection layer Initializes intal Module state, shared by oth Module and > OUT UNE # Create the transformer Script Module.
> TIME UNE
248
transformer = Transformer (encoder, decoder, src _embed, tgt _embed, src _pos, tgt_pos,
> WSL: Ubuntu
2 A0
Ln248. Col 88 Space :4 UTF-8 LF( Python3. 10. 6 (transformer :conda)
ENG
5/22/2023
7:52 PM
# Create the projection layer 246 projection _layer = Projection Layer (d_model, tgt_vocab_size)
248
# Create the transformer transformer = Transformer 就这样了
ed, src _pos, tgt _pos, projection _layer )
> OUTUNE
249
tgt_e
> WSL: Ubuntu > TIME LNE
And that's it.
Ln 249, Col5 Spaces :4 UTF-8 LFPython3. 10. 6(trans fc
P 办 ENG
5/22/2023
7:52 PM
projection _layer = Projection layer (d_model, tgt_vocab_size)
248
transformer = Transformer (encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection _layer )
# Create the transformer > OUTUNE
249
> TIMELNE
250
> WSL: Ubuntu 办 ENG
5/22/2023
7:52 PM
245
246
projection _layer = Projection Layer (d _model, tgt _vocab _size )
> OUT UNE 现在我们可以使用 Xavier Uniform 初始化参数.
> WSL : Ubuntu > TIME LNE Now we can just initialize the parameters. cos
( Pytho
7:52 PM
5/22/2023
projection _layer = Projection layer (d _model, tgt _vocab _size )
# Create the transformer transformer = Transformer (encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection _layer )
> OUTUNE
250
WSL: Ubuntu00
> TIMELINE Ln250, Col5 Space:4 UTF-8 LF( Python 3. 10. 6 (trans
ENG
5/22/2023
7:52 PM
245
246
projection _layer = Projection layer (d _model, tgt _vocab _size )
> OUTUNE 这是一种初始化参数的方法, 以加快训练速度, 使它们不从
WSL: Ubuntu00
> TIMELNE
This is a way to initialize the parameters to make the training faster so they don't 人
ENG
5/22/2023
7:52 PM
> OUT U NE 这是一种初始化参数的方法, 以加快训练速度, 使它们不从
> WSL : Ubuntu > TIME LNE just start with random values.
Ln250, Col7 S
UTF-8 LFPytho
7:52 PM
/22/2023
随机值开始.
> OUT U NE
> WSL : Ubuntu > TIME LNE just start with random values.
Ln250, Col7 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
7:52 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln250, Col7 Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
7:52 PM
> OUT U NE 有很多算法可以做到这一点
> WSL Ubuntu > TIME LNE And there are many algorithms to do it. 50. col30
Spaces4 UTF-8 LF( Python 3. 10. 6(trans
5/22/2023
7:52 PM
> OUT U NE
> WSL Ubuntu > TIME LNE
Ln 251, Col5 Spaces4 UTF-8 LF (↓ Python 3. 10. 6(trans
ENG
5/22/2023
7:52 PM
> OUT UNE 我看到很多实现使用了 Xavier, 所以我认为这对模型来说是一个很
> WSL Ubuntu > TIME LNE I saw many implementations using Xavier, so I think it's a quite good start. for the ENG
/22/2023
7:53 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 251, Col SSpaces4 UTF-8 LFPython 3. 10. 6(transfo
5/22/2023
7:53 PM
254
255 最后返回我们心爱的 Transformer. 就是这样, 这就是你构建模型的
> OUT U NE
> WSL : Ubuntu > TIME UNE and finally return our beloved transformer and this is it this is how you build. the 人
ENG
/22/2023
7:53 PM
254
255 方法. 现在我们已经构建了模型, 我们将进一步使用它.
> OUTUNE
WSL: Ubuntu > TIME LNE and finally return our beloved transformer and this is it this is. how you build. the 7:53 PM 人
5/22/2023
254
55 方法. 现在我们已经构建了模型, 我们将进一步使用它.
> OUTUNE
WSL: Ubuntu > TIME LNE model and now that we have built the model we will go further to use it so. we will 7:53 PM
/22/2023
nn. in it. xavier _uniform _(p)
254
255
return transform e > OUT U NE 所以我们将创建,
> WSL : Ubuntu > TIMELINE model and now that we have built the model we will go further to use it so. we will ENG
5/22/2023
7:53 PM
nn. in it. xavier _uniform _(p)
255
254
return transformer 所以我们将创建.
> OUT U NE
> WSL : Ubuntu > TIMELINE create the Ln 255, Col23 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:53 PM
254
5
> OUTUNE 我们首先会看一下数据集, 然后我们将构建训练循环,
WSL: Ubuntu > TIMELINE we will first have a look at the data set then we will build the training loop 人
/22/2023
7:53 PM
nn. init. xavier_uniform_(p)
255
return transformer > OUT U NE
WSL : Ubuntu > TIME UNE Ln255, Col23
Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
7:53 PM
5/22/2023
254
55
> OUTUNE 在训练循环之后, 我们还将构建推理部分和可视化注意力的
WSL: Ubuntu00
> TIME UNE after the training loop we will also build the inferencing part and the code for oscr
7:53 PM 人
5/22/2023
253
nn. init. xavier _uniform _(p)
254
55
> OUTUNE 在训练循环之后, 我们还将构建推理部分和可视化注意力的
> WSL: Ubuntu > TIMELINE visualizing the attention so hold on and take some coffee take some tea because it's. 人
/22/2023
7:53 PM
254
55
> OUTUNE 代码. 所以请耐心等待, 喝点咖啡或茶, 因为这会稍微长
> WSL: Ubuntu > TIMELINE visualizing the attention so hold on and take some coffee take some tea because it's
/22/2023
7:53 PM
253
nn. init. xavier _uniform _(p)
254
55
> OUTUNE 代码. 所以请耐心等待, 喝点咖啡或茶, 因为这会稍微长
> WSL: Ubuntu > TIMELINE gonna be a little long but it's gonna be worth it ≤4 UTF-8 LFPytho
3. 10. 6 (t
7:53 PM
5/22/2023
254
255
return transformer 些, 但绝对值得.
> OUT U NE
> WSL : Ubuntu > TIMELINE gonna be a little long but it's gonna be worth it Spaces 4 UTF-8 LF Python3. 10. 6(trans P 办 ENG
5/22/2023
7:53 PM
nn. init. xavier_uniform_(p)
255
return transformer > OUT U NE
> WSL : Ubuntu > TIMELINE Ln255, Col23
Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
7:53 PM
现在我们已经完成了模型的代码, 下一步是构建训练代码.
now that we have built the the code for the model our next step is to build the
但在我们开始之前, 我们先来复查一下代码, 因为可能会有
training code but before we do that we first i let's re check the code because um we
但在我们开始之前, 我们先来复查一下代码, 因为可能会有
may have some
一些拼写错误. 我实际上已经做了这个检查, 代码中有一些
may have some
错误. 我对比了旧代码和新代码.
typos i actually already made this check and there are few mistakes in the code i
#( Batch, Seq _ Len
t(torch. re lu (self. linear _1(x)))
66
return self. linear_2(self. dropout (torch. relu(self. linear_1(x)))
#( Batch, Seq_ Len,
d_model )-->( Batch, Seq_ Len, d_ff)-->( Batch
67
Multi Head Attention Block (nn. Module ): 错误. 我对比了旧代码和新代码
> OUT UNE super ()._in it _()
in it_(self, d _model :int, h:int, dropout :float )-> No ne:
WSL : Ubuntu > TIME LNE compared the old with the new one Ln 56. Col1 Spaces:4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
ENG
8:17 PM
return self. linear _2(self. dropout(torch. relu(self. linear _1(x))))
#( Batch, Seq_ Len,
d_model )-->( Batch, Seq _ Len, d_ff )-->( Batch return self. linear _2(self. dropout(torch. relu(self. linear _1(x))))
#( Batch, Seq_ Len,
d_model )-->( Batch, Seq _ Len, d_ff )-->( Batch class Multi Head Attention Block (n. Module):
68
class Multi Head Attention Block (nn. Module ):
> OUT UNE def _in it_(self, d _model :in t, h:in t, d ropout:float)-> None:
69
70
def
_init_(self, d_model :int, h:int, dropout :float )-> None :
> TIME LNE
super()._init_()
71
super()._init_()
WSL: Ubuntu 人
ENG
8:17 PM
/22/2023
class Layer No malization (nn. Module ):
class Layer Normalization (nn. Module ):
> OUT UNE 这些都是非常小的问题:所以我们在这里写的是
"feedforward "而
WSL: Ubuntu
> TIMELINE
it is very minor problems sowe wrote feed forward instead of feed forward here and 人
8:17 PM
98
98
99
99 不是"feedforward ", 同样的问题也出现在所有提到
J"feedforward "的地方
so thesameproblem isalsopresentinevery
referencetofeed forward and also here 8:17 PM
168
169
170
> OUTUNE
172
173 以及我们在构建解码器块时
> WSL: Ubuntu > TIME UNE so the same problem is also present in every reference to feed forwardand also here omer:om 口 通
ENG
5/22/2023
8:17 PM
168
169
170
> OUTUNE 以及我们在构建解码器块时
WSL: Ubuntu > TIME UNE
174 通
P
ENG
5/22/2023
8:17 PM
171
> OUTUINE
172
173
> WSL: Ubuntu > TIMELINE 1. 74
174
Ln 56. Col1 Spaces :4 UTF-8 LF(↓ Python3. 10. 6(thsfomer:conda)
ENG
5/22/2023
8:17 PM
169
der(nn. Module ):
169
der(nn. Module):
170 在这里构建解码器块时, 我们只写
170
> OUTUNE 另一个问题是"
J"nnmodul
e
> WSL: Ubuntu > TIMELINE and the other problem is that here when we built the decoder block we just wrote ENG
8:17 PM
169
Decoder(nn. Module ):
169
170
170
> WSL: Ubuntu
nn. module instead it should be nn. module list and then the feed forward should. be co
8:17 PM 口
ENG
5/22/2023
246
rojection
b_size
246
# Create the projection layer re r(dm 应该改为"nn module list "然后"feedforward "也应该在这里和构建 Transformer WSL : Ubuntu also fixed here 人
5/22/2023
8:17 PM
feed _forw ard_block= Feed For
odel, d_ff, drc
238
decoder_block = Decoder Block(decoder_self_attention_block, decoder
feed_for
d_block = Feed Fom
ward Block(d_model, d_ff, dropout) 方法中修正. 现在我可以删除旧的,
239
oder_block)
> OUTUINE 我们不再需要它
a Ooder
WSL : Ubuntu > TIMELINE 5/22/2023
8:17 PM
43 class Layer Normalization (nn. Module ):
> OUT UNE 让我检查一下模型它是正确的, 包含"feedforward "
WSL: Ubuntu
> TIMELINE
need it anymore letmecheck the model it's the correct one with feed forward.
8:17 PM
249 # Create the transformer transformer = Transformer (encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection _layer )
for p in transformer. parameters ():
# Initialize the parameters > OUTUNE
254
ifp. dim()>1:
WSL: Ubuntu > TIME LNE
255
nn. init. xavier _uniform _(p)
Z
5/22/2023
8:17 PM
249 # Create the transformer 250 transformer = Transformer (encoder, decoder, src projection _layer > OUT U NE 是的, 好的,:我们的下一步是构建训练代码. 但在构建
WSL : Ubuntu > TIME LN E
yes okay our next step is to build the training code but before we build the training ENG
5/22/2023
8:17 PM
249
250
# Create the transformer 251 transformer = Transformer (encoder, decoder, src _
> OUTLINE 训练代码之前,:我们必须看一下数据. 我们将使用什么样的
> WSL : Ubuntu > TIMELINE code we have to look at the data what kind of data are we going to work with so asi
8:17 PM
5/22/2023
249 # Create the transformer, tgt _embed, src _pos, tgt _pos, projection _layer )
transformer = Transformer (encoder,
# Initialize the para for pin transformer. par 数据?
> OUTUNE
254
ifp. dim()>1:
> WSL: Ubuntu > TIMELINE code we have to look at the data what kind of data are we going to work with so asi 人
ENG
5/22/2023
8:18 PM
249 # Create the transformer 250
51
transformer = Transformer (encoder, decoder, src projection _layer )
> OUT U NE 如我之前所说,:我们处理的是翻译任务, 我选择了这个名为
> WSL : Ubuntu > TIMELINE code we have to look at the data what kind of data are we going to work with so asi
5/22/2023
8:18 PM
250
251
transformer = Transformer (encoder, decoder,
> OUT U NE 如我之前所说,:我们处理的是翻译任务, 我选择了这个名为
> WSL : Ubuntu > TIMELINE said UTF-8 LF
226, Col 24
Pytho
B:18 PM
5/22/2023
{"en°:" The said Eliza, John,
and Georg i
nd with hex opus books 的数据集, 可以在 Hugging Face 上找到我们还将
darlings about her (for the time neither rrelling
a ggruppati in
salot to attorno alla.
3
ard fro
{"en°:" The said Eliza, John, and Georgi darlings about her (fox the time neither sofa by the fireside, and with her aggruppati in salotto attorno all a
uarr elling nor crying ) lool perfectly happy."
" Eliza,
nce ;but that until she heard fron
5
sdlataset
for us
ndai.*}
/22/2023
7 da rlings a bout her (for the time neither quaxrelling nor crying) 1ooked perfectly happy.°, *it*: " Eliza, John e Georgiana erano aggruppat i in sa lotto attozno alla
{ "en *: * The said Eliza, John, and Georgiana were now clustered round their mama in the drawing-room: she lay reclined on a sofa by the fireside, and with her
Bessie, and could di scover by her own observati on, that I was endeavouring in good earnest to acquire a more sociable and childlike disposit ion, a more attractive.
{"en*:" Me, she had dispensed from joining the group; saying, \" She regret ted to be under the necessity of keeping me at a distance; but that until she heard from
{"en*:"\" What does Bessie say I have done?\° I asked.*,*it°:"- Che cosa vi ha detto Bessie di nuovo sul con to mio?- domandai." }
5/22/2023
8:18 P
{"en°:" The said Eliza, John and Georg i
nd with her darlings about her (for the time neither rrelling
a ggruppatiin
salot to attor no all a.
ntil she ce attractive.
heard from And this is the only because of course we
darlings about her (for the time neither quarrelling no I
"en": " The said Eliza,
John and Georgia d the ix in the "it ":
" Eliza, John e Georgia na erano aggzuppati in salot to attor no all a
she lay reclined on a sofa by the fireside, and with her ss it y
y of keeping me at a distance ; but that until she heard from more sociable and childlike disposition, a more attractive we w Torch because of course we
ENG
5/22/2023
{"en*: " The said Eliza,
and with hex 我们将使用这个数据集, 并且我们还将使用 Hugging Fakes 的
darlings about her (for the tim
aggruppati in salotto attorno alla 因此,
all so
*en": * The said Eliza, John neither " Eliza and with her 因此, 我们将使用这个数据集, 并且我们还将使用 Hugging Fakes 的
{"en°:" The said Eliza, John darlings about her (for the time neither and Georgi she lay reclined on a sofa by the fireside, and with her quarrelling " Eliza,
of keeping me at a distance ; but that until she heard from ore sociable and childlike disposition, a more attractive _
into vocabulary.
5/22/2023
7 da rlings a bout her (for the time neither quazrelling nor crying) looked perfectly happy.°, *it*: " Eliza, John e Georgiana ezano aggxuppat i in sa lotto attorno alla.
↑ "en*: * The said Eliza, John, and Georgiana were now clustered xound their mama in the drawing-room: she lay reclined on a sofa by the fireside, and with her
8
Bessie, and could di scover by her own observati on, that I was endeavouring in good earnest to acquire a more sociable and childlike disposit ion, a more attractive.
{"en*:" Me, she had dispensed from joining the group; saying, \" She regret ted to be under the necessity of keeping me at a distance; but that until she heard from
{"en2:"\" What does Bessie say I have done?\ I asked.*,*it°:"- Che cosa vi ha detto Bessie di nuovo sul con to mio?- domandai.°}
5/22/2023
8:18 P
{"en°:" The said Eliza, John and Georg and with her 因为我们的目标是构建 Transformer 所以不是要重新发明所有的轮子
darlings about her (for the time neither arr elling he wheel
{"en°:" The said Eliza, John. and with her neither rr elling 因此,
about everything so we wil he transformer
{"en°:" The said Eliza, John,
and Georgi and with her 我们将只专注于构建和训练 Transformer neither aggruppatiin
salotto attorno all a. 因此,
particular case. i
{"en°:* The said Eliza, John,
and Georgi
d with hex 我将使用英语到意大利语的子集,
darlings about her (for the time neither 但我们将以这样的方式
until she attractive.
heard from code in such a way that you can choose accord n gly
7 da rlings a bout her (for the time neither quarrelling nor crying) looked perfectly happy.°, *it°: " Eliza, John e Georgiana ezano aggxuppat i in sa lotto attorno alla.
↑ "en *: * The said Eliza, John, and Georgiana were now clustered round their mama in the drawing-room: she lay xeclined on a sofa by the fireside, and with her
8
Bessie, and could di scover by her own observati on, that I was endeavouring in good earnest to acquire a more sociable and childlike disposit ion, a more attractive.
{"en°:" Me, she had dispensed from joining the group; saying, \" She regret ted to be under the necessity of keeping me at a distance; but that until she heard from
{"en*:"\" What does Bessie say I have done?\* I asked.*,*it*:"- Che cosa vi ha detto Bessie di nuovo su l conto mio?- domandai."}
5/22/2023
如果我们看一下数据,
218 If we look at. the. data, we. can. see. that. each data. item. is. a. pair. of. sentences.
ag1
ndai.*}
10
superiori."
‘11
{"en*:" Be seated some whe 息
If we look at. the. data, we. can. see that. each data. item. is. a. pair. of sentences. i
parlare ragion
mente."
8:19
mio?
domanda i."}
‘10
I don't like
isuperiori.}
‘11
["en ":" Be seated some parlare ragione vol mente.*
English and. in. Italian.
/22/202
8:19
{"en *: *\" What does Bessie say I have done?\* I asked ‘10
{"en:"\ Jane, I don't 1ike cavillers or superior i..
hild taking up her elders in that manner.","it":
Che
‘11
parlare ragionevolmente.*
5/22/2023
8:19 P
那天不可能散
in. quel
‘11
For example, there. was. no. possibility of taking a-walk that. day,. which in. Italian.
8:19
ai.* }
10 如 品
in. quel
‘11
parlare ragione vol men are
sand ai."}
giorno era.
Impossi
‘11
e
means. in. gue l
ce ragionevolme
/22/2023
8:19
{"en *:*\" What does Bessie say I have done?\* I askeo
sul conto mio?- domand ai."}
10
{"en:"\ Jane, I don't 1ike cavillers or bimba trat ti cosi i
superior i.
hild taking up her elders in that manner.","it":
‘11
parlare ragionevolmente.*}
5/22/2023
8:19 P
训练我们的 Transformer 译到
train our transformer. to. translate from the source l
re ragione vol mente guage. which is. english into. the /22/202
9
dai." }
10
‘11
target language. which. is. italian so let's. do. it we. will. do. it. step. by step. so. first parlare ragi
/22/202
8:19
首先, 我们将编写代码来下载
tokenizer we. w
sul conto mio?-domandai.}
‘11
parlare ragione vol men te.
["en":" Be seated somewhere make the code. to. download. this. data set and to create. the. tokenizer. so. what. is. the.
ENG
/22/2023
8:19
d1
sul conto mi o?-domandai.}
10
mi piace di essere I don't like forbidding in superior i.}
child taking up her elders in that manner.,"it":
‘11
["en":" Be seated somewhere tokenizer.
/22/2023
8:19
{"en °: *\" What does Bessie say I have done?\ I as ke
it"
sul conto mio?-doma ndai."}
‘10
Jane, non mi piace di essere interrogata. Sta male, del resto,
{"en:"\ Jane, I don't 1ike cavillers or superior i.*}
‘11
["en*:" Be seated somewhere ;and until you can speak pleas parlare ragionevolmente.*
5/22/2023
让我们回到幻灯片,
let'sgo back to the. slides to. just have. a. brief. overview of what we. are. going. to do
让我们回到幻灯片, 简要概述一下我们将如何处理这些数据.
with this data the tokenizer is what comes before the input embeddings so we have an
tokenizer 是在输入嵌入之前使用的.
with this data the tokenizer is what comes before the input embeddings so we have an
所以我们有一个英语句子, 例如,"你的猫是一只可爱的
with this data the tokenizer is what comes before the input embeddings so we have an
所以我们有一个英语句子, 例如,"你的猫是一只可爱的
english
所以我们有一个英语句子, 例如,"你的猫是一只可爱的
sentence so for example your cat is a lovely cat but this sentence will come from our
tokenizer 的目标是创建这些标记, 民 即将句子分割成单个单词,
data set the goal of the tokenizer is to create this token so split this sentence
tokenizer 的目标是创建这些标记, 即将句子分割成单个单词,
into single
tokenizer 的目标是创建这些标记, 民 即将句子分割成单个单词,
words which has many strategies as you can see here we have a sentence which is your
这有很多策略, 如你所见.
words which has many strategies as you can see here we have a sentence which is your
这里我们有一个句子:"你的猫是一只可爱的猫", tokenizer words which has many strategies as you can see here we have a sentence which is your
这里我们有一个句子:"你的猫是一只可爱的猫", tokenizer cat is a lovely cat
这里我们有一个句子:"你的猫是一只可爱的猫", tokenizer and the goal of the tokenizer is to split this sentence into single words which can
的目标是将这个句子分割成单个单词, 这可以通过多种方式
and the goal of the tokenizer is to split this sentence into single words which can
完成. 有 BPE tokenizer, 有词级tokenizer, 有子词级词部分组织器. 有很
be done in many ways there is the bpe tokenizer there is the word level tokenizer
多种 tokenizer.
there is the subword level word part organizer there are many tokenizer s
我们将使用最简单的一种, 称为词级tokenizer. 所以词级tokenizer the one we will be using is the simplest one called the word level tokenizer so the
所以每个空格定义了一个单词的边界, 从而分割成单个单词,
word level tokenizer basically will split this sentence let'ssay by space so each
所以每个空格定义了一个单词的边界, 从而分割成单个单词,
space
所以每个空格定义了一个单词的边界, 从而分割成单个单词,
defines the boundary of a word and so into the single words and each word wil be
每个单词将被映射到一个数字.
mapped to one number so this is the job of the tokenizer to build the vocabulary and
所以这是tokenizer 的工作:构建词汇表和这些数字, 并将每个
mapped to one number so this is the job of the tokenizer to build the vocabulary and
单词映射到一个数字.
of these numbers and to map each word into a number
单词映射到一个数字.
单词映射到一个数字.
the when we build the tokenizer we can also create special tokens which we will use
当我们构建tokenizer 时, 我们还可以创建一些特殊标记, 这些
the when we build the tokenizer we can also create special tokens which we will use
标记将用于 Transformer,- 例如称为填充的标记、句子开始的标记、
for the transformer for example the tokens called padding they call the token called
标记将用于 Transformer, 1 例如称为填充的标记、 句子开始的标记、
the start
句子结束的标记一一这些对于训练 Transformer 是必要的. 但
of sentence end of sentence which are necessary for training the transformer but we
我们将一步一步来做.
will do it step by step so let's build first the code for the um building the
所以让我们先编写构建tokenizer 和下载数据集的代码. 好的,
will do it step by step so let's build first the code for the um building the
所以让我们先编写构建tokenizer 和下载数据集的代码. 好的
tokenizer and to
让我们创建一个新文件, 我们称之为train. py download the data set okay let's create a new file let's call it train do t pi
249 # Create the transformer 250
251
transformer = Transformer (encoder, decoder, src tgt _pos, projection _layer )
> OUTUINE 让我们创建一个新文件, 我们称之为 train. py.
WSL: Ubuntu > TIME LNE download the data set okay let's create a new file let'scall it train dotpi
5/22/2023
8:21 PM
> OUT UI NE
> WSL : Ubuntu > TIME LINI
226o245
8:21 PM
5/22/2023
好的, 让我们导入我们常用的库, 即torch. 我们还将导入
> OUT UNE
WSL: Ubuntuβ00
> TIMELINI
okay let'simport our usual library so torch we will also import torch. nn and we also
torch nn, 并且因为我们使用的是来自 Hugging Face 的库, 我们还需要
> OUT UN TIMELINE > WSL : Ubuntu okay let'simport our usual library so torch we will also import torch. nn and we also
8:21 PM
导入.
> OUT UNE
> WSL: Ubuntu
TIMELNI 人
8:22 PM
> OUTUNE 这两个库. 我们将使用. 我们将使用datasets库, 你可以
xwsubnu oobecause we we are using a library from hugging phase we also need to import the these TIME LINI
B:22 PM
> OUT UNE 这两个库. 我们将使用. 我们将使用datasets 库, 你可以
TIME UN
two
8:22 PM
> OUT UNE 这两个库. 我们将使用. 我们将使用datasets 库, 你可以
WSLUbuntu01
> TIMEUNI
libraries we will uh using the we will be using the data sets library which you. can 口
2 ENG
8:22 PM
通过 pip 安装, 即 data sets.
> OUT U NE
> WSL Ubuntu > TIME UIN I libraries we will uh using the we will be using the data sets library which you can. 人 办 ENG咖
8:22 PM
通过 pip 安装, 即 datasets.
> OUTUNE
> WSLUbuntu
> TIMEUINI
1 A0
install using pip so data sets actually Ln4. Col8 Spaces 4 UTF-8 LFPythor
P
B:22 PM
实际上我们将使用 load _dataset, 我们还将使用来自 Hugging Face 的 tokenizer s
> WSL : Ubuntu > TIME LIN install using pip so data sets actually Ln4. Col 17 Spaces4 UTF-8 LFPython ENG
5/22/2023
B:22 PM
> OUT UNE 库, 也可以通过 pip 安装. 我们还需要确定我们需要哪种tokenizer.
> TIME LNE we will be using load dataset and we will also be using the tokenizer s library also ENG
B:22 PM
> OUT UNE 所以我们使用词级 tokenizer.
> WSL: Ubuntu00
> TIMELINI
needs o
P
5/22/2023
8:22 PM
所以我们使用词级 tokenizer.
> OUT UNE
WSL: Ubuntu00
> TIMELINI
We will use the word-level tokenizer. Ln6. col4o spxe4 urs u pyhon
3. 10. 6 (t
8:22 PM 通
ENG
5/22/2023
> OUT UI NE
WSL : Ubuntu 00
> TIMELINE
n7, Col1
UIF-8 LFPyth o
4. 10. 6 (tra
8:22 PM
ENG
5/22/2023
> OUT UNE 还有 trainers, 即用于训练tokenizer 的类, 它将根据句子列表创建
> WSL: Ubuntu11
> TIMELNE
And there is also the trainers, so the tokenizer, the class that will train. the a. o tas 人 办 ENG
5/22/2023
8:22 PM
> OUT UNE 还有trainers, 即用于训练tokenizer 的类, 它将根据句子列表创建
WSL: Ubuntu10
> TIMEUNE
tokenizer, so that will create the vocabulary given the list of sentences.
4. 10. 6 Ct
ENG
5/22/2023
8:22 PM
词汇表.
> OUT UNE
> WSL: Ubuntu
> TIMELNI
1 A0
tokenizer, so that will create the vocabulary given the list of sentences.
ENG
8:23 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
1 A0
In 7. Col33
UTF-8 LFPytho
4. 10. 6 (tra
ENG
8:23 PM
5/22/2023
我们将根据空格分割单词. 我将一步一步地构建方法.
> OUT U NE
WSL : Ubuntu > TIME UNE o A and we will split the word according to the whitespace I will build one method by one
8:23 PM
首先, 我将构建创建tokenizer 的方法, 并描述每个参数.
> OUTUNE
> WSL: Ubuntu00
> TIMELINE so l will build first the methods to create the tokenizer and l will describe each ENG
5/22/2023
B:23 PM
首先, 我将构建创建 tokenizer 的方法, 并描述每个参数.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
parameter Ln10. Col1 S
/22/2023
8:23 PM
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
4. 10. 6 (tra
8:23 PM
ENG
5/22/2023
> OUTUNE 现在你可能还没有整体的认识, 但稍后当我们把这些方法结合
WSL: Ubuntu00
> TIMELNE
for now you will not have the bigger picture but later when we. combine all these Z
2 ENG
/22/2023
B:23 PM
> OUT U NE 起来时, 你就会有更全面的理解,
WSL : Ubuntu > TIME L NE
o methods together you will have the bigger picture so let's first make the method that 口 人
5/22/2023
B:24 PM
所以让我们首先创建构建tokenizer 的方法, 我们称之为get Or Build Tokenizer,
8:24 PM 口
> OUT U NE 这个方法接受配置, 即我们模型的配置.
> WSL Ubuntu > TIME UNE
10
Ln 10, Col 25
UTF-8 LF{ Pytho
4. 10. 6 (tr
B:24 PM
5/22/2023
> OUTUNE 这个方法接受配置, 即我们模型的配置.
WSL: Ubuntu10
> TIMELNE
And this method takes the configuration, which is the configuration of our. model. 人
8:24 PM
> OUT UNE 我们稍后会定义它.
> WSL: Ubuntu
> TIMELINE
1 A0
We will define it later.
Ln 10. Col36
20
4. 10. 6 (tr
B:24 PM
P
ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE 10
Ln 10. Col 36
Spaces4 UTF-8 LFPytho r
4. 10. 6 (tr
ENG
8:24 PM
5/22/2023
> OUT UNE 数据集和我们将为其构建tokenizer 的语言.
> WSL : Ubuntu > TIMELINE The dataset and the language for which we are going to build the. tokenizer. ocm
B:24 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 1 A0
Ln. 10. Col44
UIF8 LFPyon
4. 10. 6(ta
ENG
8:24 PM
5/22/2023
我们定义 tokenizer 路径.
> OUT U NE
> WSL : Ubuntu > TIMELINE We define the tokenizer path.
4. 10. 6 (t
8:24 PM 通
ENG
5/22/2023
即我们将保存此 tokenizer 的文件.
> OUT U NE
WSL : Ubuntu > TIMELINE 3. 10. 6(tra
ENG
B:24 PM 通
5/22/2023
> OUT U NE 我们通过配置路径来实现.
WSL : Ubuntu > TIME LNE And we do it path of configuration.
Ln11. Col26 Spaces4 UTF-8 LFPython 3. 10. 6(tra
ENG
B:24 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 11. Col33
4. 10. 6 (tra
8:24 PM
ENG
5/22/2023
> OUT UNE 好的, 让我定义一些东西. 首先, 这个路径来自pathlib
WSL: Ubuntu
> TIMELNE
01
okay let me define some things first of all this path is coming from the path lib so 通
ENG
8:24 PM
> OUT U NE 这是一个允许你根据相对路径创建绝对路径的库, 我们假设有
WSL : Ubuntu > TIME LNE from path lib this is a library that allows you to create absolute paths given ao6 tran
ENG
/22/2023
B:24 PM
> OUT U NE 这是一个允许你根据相对路径创建绝对路径的库, 我们假设有
> WSL : Ubuntu > TIME LNE relative path s
Pytho
8:24 PM 通
P
ENG
5/22/2023
一个名为 tokenizer _file 的配置, 它是tokenizer 文件的路径, 并且
> OUT UNE > TIME LNE and we pretend that we have a configuration called the tokenizer file which is the 人
ENG
825 PM
> OUT UNE 一个名为tokenizer file 的配置, 它是tokenizer 文件的路径, 并且
WSL: Ubuntu00
> TIMEUNE
path to the tokenizer file and this path is format able using the language so forscm
8:25 PM
> OUTUNE 这个路径可以使用语言进行格式化. 例如, 我们可以这样写.
WSL: Ubuntu00
> TIMEUNE
path to the tokenizer file and this path is format able using the language so fors. cm
/22/2023
8:25 PM
这个路径可以使用语言进行格式化. 例如, 我们可以这样写.
> OUTUNE
WSL: Ubuntu00
> TIMELINE example we can have something like this u cals Space UTF-8 LFPython
8:25 PM
Z
5/22/2023
> OUTUNE
WSL: Ubuntu00
> TIMELINE Ln 13, Col14 Spaces4 UTF-8 LF( Python ENG
5/22/2023
8:25 PM
例如, 像这样:
> OUT U NE
WSL Ubuntu > TIME LNE For example, something like this.
Ln 13, Col37 Spaces4 UTF-8 LF( Python 5/22/2023
8:25 PM
> OUTUNE
WSL: Ubuntu00
> TIMELNE
Ln 13, Col61 Spaces4 UTF-8 LF{ Python ENG
5/22/2023
8:25 PM
并且这将根据语言创建, 例如, 英语 tokenizer 或意大利语 tokenizer.
> OUT UNE > TIME UNE And this will be, given the language, it will create a tokenizer English or tokenizer.
> WSLUbuntu 人
ENG
5/22/2023
:25 PM
并且这将根据语言创建, 例如, 英语 tokenizer 或意大利语tokenizer.
> TIMELINE > OUT U NE
> WSL : Ubuntu Italian, for example.
Ln 13, Col 59 (10 selected )
Pytho
ENG
/22/2023
8:25 PM
> OUTUNE
WSL: Ubuntu00
> TIMELNE
4. 10. 6 (tra
ENG
B:25 PM
5/22/2023
因此, 如果 tokenizer 不存在, 我们就创建它,
> OUT UNE
> WSL: Ubuntu
> TIMELNE
2 A0
So if the tokenizer doesn't exist, we create it.
4. 10. 6(trar
UTF-8 LFPython ENG
8:25 PM
5/22/2023
> OUTUNE
WSL: Ubuntu20
> TIMEUNE
Ln14. Col17 Spaces4 UIF-8 LF{ Python 3. 10. 6(trans
ENG
5/22/2023
8:25 PM
我实际上是从 Hugging Face 那里获取了所有这些代码, 这并不
> OUT UNE
WSL: Ubuntu10
> TIMELNE
I took all this code actually from Hugging Face, it's nothing complicated, ljustscm
ENG
5/22/2023
8:25 PM
> OUT UNE 复杂, 我只是快速浏览了他们的tokenizer 库, 使用起来
> WSL: Ubuntu
> TIMEUNE
1 A0
I took all this code actually from Hugging Face, it's nothing complicated, ljusto
ENG
5/22/2023
8:25 PM
> OUT UNE 复杂, 我只是快速浏览了他们的tokenizer 库, 使用起来
> WSL: Ubuntu
> TIMEUNE
1 A0
took a quick tour of their tokenizer library and it's really easy to use it and save s 口
ENG
5/22/2023
8:25 PM
> OUT UNE 非常简单, 并且能节省大量时间, 因为构建tokenizer 真的是在
> WSL : Ubuntu > TIME LNE
10
youalot
Ln15, Col9 Spaces UTF-8 LFPytho
ENG
8:26 PM
5/22/2023
> OUT UNE 非常简单, 并且能节省大量时间, 因为构建tokenizer 真的是在
> WSL: Ubuntu
> TIMELINE
1 A0
of time because to build a tokenizer is really reinventing the wheel. yhon
4. 10. 6 人
2 ENG
8:26 PM
重新发明轮子.
> OUTUNE
> WSL: Ubuntu
> TIMELNE
10
of time because to build a tokenizer is really reinventing the wheel. wo. uo. tam
ENG
5/22/2023
8:26 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 10
Ln 15. Col9 Spaces4 UTF-8 LFPython 办 ENG
5/22/2023
8:26 PM
> OUT UN E 我们还将引l入未知词"unknown > WSL : Ubuntu > TIME UNE And we will also introduce the unknown word unknown. ur:s w
yion 3. 06ctacome r. com
ENG
8:26 PM 口 通
Z
5/22/2023
> OUT UNE 那么这是什么意思呢?
WSL: Ubuntu00
> TIMEUNE
So what does it mean?
Ln 15. Col52 Spaces4 UTF-8 LF { Python 3. 10. 6(trans fo
ENG
8:26 PM
5/22/2023
> OUT UNE 如果我们的tokenizer 在其词汇表中遇到不认识的词, 它将用这个
> WSL: Ubuntu00
> TIMEUNE
If our tokenizer sees a word that it doesn't recognize in its vocabulary, it will octar
ENG
5/22/2023
8:26 PM
> OUT UNE 如果我们的tokenizer 在其词汇表中遇到不认识的词, 它将用这个
WSL Ubuntu > TIME UNE replace it with this word unknown. as6(tscketed)
LF Python 3. 10. 6(ta
ENG
8:26 PM 通
5/22/2023
"unknown "词替换它.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
replace it with this Word unknown. a34 seketed Spaces4u F8 ( Python 310. 6(transto
ENG
5/22/2023
8:26 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 15. Col57 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
8:26 PM
5/22/2023
它将映射到与这个"unknown "词对应的数字.
> OUT U NE
> WSL : Ubuntu > TIME LNE It will map it to the number corresponding to this word unknown. py ion 3. o. ctam
ENG
8:26 PM 人
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln15. Col 52(5 selected ) Spaces 4 UTF-8 UF( Python 3. 10. 6(tran
ENG
8:26 PM
5/22/2023
> OUT UNE 预tokenizer 基本上意味着我们按空白字符分割, 然后我们构建
> TIME LNE The pre-tokenizer means basically that we split by white spaceand then we build. the ENG
/22/2023
训练器来训练我们的 tokenizer.
> OUT U NE
> WSL : Ubuntu > TIME LNE The pre-tokenizer means basically that we split by white spaceand then we build. the 5/22/2023
8:26 PM
训练器来训练我们的 tokenizer.
> OUT U NE
WSL : Ubuntu > TIME LNE trainer to train our tokenizer.
Ln 17. Col17 Spaces4 UTF-8 LF { Python 3. 10. 6(trans 通
ENG
5/22/2023
8:26 PM
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 17. Col17 Spaces4 UTF-8 LFPython ENG
5/22/2023
8:26 PM
好的, 这是训练器.
> OUT U NE
> WSL : Ubuntu > TIME UNE Okay, this is the trainer.
Ln18. Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
8:27 PM 通
5/22/2023
> OUT U NE 这意味着它将是一个词级别的训练器, 因此它将使用空白字符
> WSL : Ubuntu > TIME UNE It means it will be a word level trainer, so it will split words using the white o tn
8:27 PM 口
ENG
5/22/2023
和单个词来分割词, 并且还将有四个特殊标记.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
space and using the single words and it will also have four special tokens.
4. 10. 6(tran
8:27 PM
ENG
5/22/2023
> OUT UNE 一个是"unknown ", 这意味着如果在词汇表中找不到那个特定的词,
> WSL: Ubuntu00
> TIMELNE
One is unknown, which means that if you can not find that particular word in. the ENG
5/22/2023
8:27 PM
> OUT UNE 一个是"unknown ", 这意味着如果在词汇表中找不到那个特定的词,
> WSL: Ubuntu00
> TIMELINE vocabulary, just replace it with unknown. col s spaa
Pytho
:27 PM
ENG
5/22/2023
就将其替换为"unknown'
> OUT U NE
> WSL Ubuntu > TIMELINE 5/22/2023
8:27 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 17. Col55(3 selected ) Spaces 4 UTF-8 LF{ Python 3. 10. 6(trans
5/22/2023
8:27 PM
> OUT UNE 它还将有填充标记, 我们将用它来训练 Transformer, 以及句子的开始
> WSL : Ubuntu > TIMELINE It will also have the padding, which we will use to train the transformer, the start 人
ENG
5/22/2023
B:28 PM
> OUT UNE 它还将有填充标记, 我们将用它来训练 Transformer, 以及句子的开始
> WSL : Ubuntu > TIMELINE of sentence and the end of sentence special tokens. es urs u ython
ENG
5/22/2023
8:28 PM
和结束特殊标记.
> OUTUNE
WSL: Ubuntu00
> TIMELINE of sentence and the end of sentence special tokens. esa urs u
ython 310o6tansto ENG
8:28 PM
5/22/2023
> OUT U NE
WSL : Ubuntu 00
> TIMELINE
Ln 17. Col84(3 selected ) Spaces 4 UTF-8 LF( Python 3. 10. 6(trans
ENG
5/22/2023
8:28 PM
平均频率意味着一个词要想出现在我们的词汇表中, 它的频率
> OUTUNE
WSL: Ubuntu00
> TIMELINE Mean frequency means that a word for a word to appear in our vocabulary, it has. to 通
ENG
5/22/2023
8:28 PM
> OUT U NE 平均频率意味着一个词要想出现在我们的词汇表中, 它的频率
> WSL Ubuntu > TIME LNE have a frequency of at least two. col102(13 ekts spa
Python B:28 PM
5/22/2023
必须至少为两次.
> OUT U NE
WSL : Ubuntu 00
> TIMELNE
ENG
5/22/2023
8:28 PM
> OUT U NE
> WSL Ubuntu > TIME LNE
Ln:17. Col 102(13 selected ) Spaces 4 UTF-8 LF( Python 3. 10. 6(tran
5/22/2023
8:28 PM
> OUT UI NE 现在我们可以训练tokenizer 了.
> WSL : Ubuntu > TIME LNE Now we can train the tokenizer.
4. 10. 6(trans
ENG
828 PM 通
5/22/2023
> OUT UI NE
> WSL Ubuntu > TIME LNE
Ln18. Col9 Spaces4 UTF-8 LFPython ENG
8:28 PM
5/22/2023
方法从我们的数据集中获取所有句子, 我们稍后会构建它.
> OUTUNE
WSL: Ubuntu01
> TIMELNE
We use this method, which means we build first a method that gives all the sentences 5/22/2023
8:28 PM
方法从我们的数据集中获取所有句子, 我们稍后会构建它.
> OUTUNE
> WSL: Ubuntu01
> TIMELNE
from our data set and we will build it later. col42 spaces 4ur8u
Pytho
8:28 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 18. Col55 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
5/22/2023
8:28 PM
好的, 所以让我们也构建一个名为 get Al I Sentence 的方法, 这样我们
> OUT U NE
> WSL : Ubuntu > TIMELINE Okay, so let's build also this method called get All sentence so that we can iterate 2 ENG
5/22/2023
8:29 PM
> OUT UNE 就可以遍历数据集, 获取与我们要为其创建tokenizer 白 的特定语言
WSL: Ubuntu01
> TIMEUNE
through the data set to get all the sentences corresponding to the particular language merco
5/22/2023
8:29 PM
> OUT UNE 就可以遍历数据集, 获取与我们要为其创建tokenizer 的特定语言
> WSL : Ubuntu > TIME UNE for which we are creating the tokenizer. 2. col spe
UTF-8 LFPytho
4. 10. 6(ta
ENG
8:29 PM 口 通
5/22/2023
> OUT U NE 对应的所有句子.
> WSL: Ubuntu
> TIMEUNE
5/22/2023
8:29 PM
> OUTUNE
WSL: Ubuntu01
> TIMELINE Ln12, Col1 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
8:29 PM
5/22/2023
> OUTUNE 如你所记, 数据集中的每个项目都是一对句子, 一个是英语
WSL: Ubuntu10
> TIMEUNE
As you remember, each item in the dataset, it's a pair of sentences, one in English,
5/22/2023
B:29 PM
> OUT U NE 的, 一个是意大利语的.
> WSL : Ubuntu > TIMELINE one in l talia n.
Ln14 Col9 Spaces4 UTF-8 LF↓ Python 3. 10. 6(trans 通
5/22/2023
8:29 PM
> OUT UNE 我们只想提取一种特定的语言.
> WSL: Ubuntu
> TIMELINE
1 A0
We just want to extract one particular language. spacs4 ursu (ython 3. 106trans to ENG
5/22/2023
8:29 PM
> OUT UNE
> WSL: Ubuntu
> TIMEUNE
Ln 14. Col1 Spaces4 UTF-8 LFPython3. 10. 6(transfc
ENG
8:29 PM
5/22/2023
> OUT U NE 这是代表这对句子的项目, 从这对句子中我们只提取我们想要
> WSL : Ubuntu > TIMELINE This is the item representing the pair and from this pair we extract only. the one ENG
5/22/2023
B:29 PM
的那种语言.
> OUT U NE
> WSL : Ubuntu > TIMELINE language that we want.
Ln 14. Col40 Spaces4 UTF-8 LF{ Python 3. 10. 6(transfc 通
ENG
5/22/2023
8:29 PM
这是构建 tokenizer 的代码.
> OUT U NE
> WSL : Ubuntu > TIME UNE And this is the code to build the tokenizer col spaces4uf-su F QPython 3. 1o6transto 通
Z
5/22/2023
8:30 PM
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln27, Col1 Spaces 4 UTF-8 LF { Python3. 10. 6(trans fo
Z
ENG
5/22/2023
8:30 PM
> OUT UNE 现在让我们编写代码来加载数据集, 然后构建tokenizer.
> WSL : Ubuntu > TIME UNE Now let'swrite the code to load the dataset and then to build the tokenizer. osca 人
5/22/2023
B:30 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 28, Col1 Spaces4 UTF-8 LFPython 3. 10. 6(transfc
5/22/2023
8:30 PM
我们将调用这个方法get Data set, 它还接受我们稍后将定义的模型
> OUT UNE
WSL: Ubuntu00
> TIMELINE We will call this method get Dataset and which also takes the configuration of the 2 ENG
5/22/2023
8:30 PM
我们将调用这个方法get Data set, 它还接受我们稍后将定义的模型
> OUT UNE
WSL: Ubuntu10
> TIMELINE model which we will define later.
Ln29, Col5 Spa
( Pytl
8:30 PM 通
Z
5/22/2023
配置.
> OUT U NE
> WSL : Ubuntu > TIME UNE model which we will define later.
Ln29, Col5 Spaces4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
8:30 PM
> OUT UN E 所以让我们加载数据集, 我们称之为 dsrow.
> WSL : Ubuntu > TIME UNE So let's load the dataset, we will call it dsrow. s spacsa ufs F ython
4. 10. 6(trans
ENG
8:30 PM 口
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 29. Col13 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
8:30 PM
5/22/2023
好的, Hugging Face 让我们可以非常容易地下载它的数据集
> OUTUNE
WSL: Ubuntu00
> TIMELNE
Okay, Hugging Face allows us to download its data sets very easily, we just need. to ENG
5/22/2023
B:30 PM
> OUTUNE 我们只需要告诉它数据集的名称.
WSL: Ubuntu00
> TIMELINE ENG
8:30 PM 口
5/22/2023
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 29. Col31 Spaces4 UTF-8 LF Python 3. 10. 6(transfc
5/22/2023
8:30 PM
> OUT U NE 然后告诉它我们想要的子集是什么.
> WSL : Ubuntu > TIMELINE and then tell him what is the subset we want.
Spaces 4 UTF-8 LFPython 3. 10. 6(trans 通
ENG
5/22/2023
8:30 PM
我们想要的是英意翻译的子集, 但我们还想让它对你们来说是
> OUT U NE
WSL : Ubuntu > TIMELINE We want the subset that is English tolt alian, but we want to also make it ENG
8:30 PM 口
5/22/2023
可配置的, 以便你们能快速更改语言.
> OUT U NE
> WSL : Ubuntu > TIMELINE configurable for you guy s to change the language very fast.&u
yhon x iosrtrani
ENG
8:30 PM
5/22/2023
所以让我们动态地构建这个子集.
> OUT U NE
> WSL : Ubuntu > TIME LNE So let's build this subset dynamically. n 29. col41 Spaces4usu Fpthon 3. 106 Ctrans
ENG
8:30 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 29. Col41 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
8:30 PM
5/22/2023
> OUT U NE 我们将在配置中有两个参数.
> WSL : Ubuntu > TIME UNE
ENG
8:31 PM
5/22/2023
> OUT UNE 一个是名为language source, 另一个是名为language target.
> WSL : Ubuntu > TIME UNE One is called language source and one is called language. target. 4 yhon 3. o6 tr 口 人
ENG
5/22/2023
8:31 PM
> OUT U NE
> WSLUbuntu
> TIMEUNE
Ln 29. Col64 Spaces 4 UTF-8 LF { Python3. 10. 6(trans fo
ENG
8:31 PM
Z
5/22/2023
稍后, 我们还可以定义我们想要的数据集的分片.
> OUT U NE
> WSL : Ubuntu > TIMELINE Later, we can also define what split we want of this. dataset.
4. 10. 6 (tra
LFPython 8:31 PM
ENG
5/22/2023
在我们的例子中, Hugging Face 的原始数据集中只有训练分片, 但
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
In our case, there is only the training split in the original dataset from uor
4. 10. 6(tra
ENG
5/22/2023
8:31 PM
> OUTUNE 我们将自己将其分为验证数据和训练数据.
WSL: Ubuntu00
> TIMELINE Hugging Face, but we will split by ourselves into the validation and the training o
ENG
5/22/2023
8:31 PM
> OUT U NE 我们将自己将其分为验证数据和训练数据.
> WSL : Ubuntu > TIME UNE data.
UTF-8 LFPython3. 10. 6trans
Ln 29, Col. 101
ENG
8:31 PM
P
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln29, Col101 Spaces 4 UTF-8 LFPython3. 10. 6(trans fo
ENG
8:31 PM
5/22/2023
> OUT UNE 所以让我们构建一个 tokenizer.
WSL : Ubuntu > TIME UNE So let'sbuild a tokenizer.
Ln31, Col6 Spaces4 UTF-8 LFPython 3. 10. 6trans
ENG
8:31 PM
P
5/22/2023
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln31. Col13 Spaces 4 UTF-8 LF (↓ Python3. 10. 6(trans fc
ENG
8:31 PM
5/22/2023
这是原始数据集.
> OUT U NE
WSL : Ubuntu > TIMELINE This is the raw data set.
Ln 32. Col58 Spaces:4 UTF-8 LFPython 3. 10. 6 Ctrans fo
ENG
8:31 PM
5/22/2023
> OUT U NE 我们也有目标数据.
> WSL : Ubuntu > TIMELINE And we also have the target.
Ln. 33, Col5 Spaces4 UTF-8 LFPython 3. 10. 6(transfo
ENG
5/22/2023
8:31 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 33, Col78 Spaces:4 UTF-8 LF{ Python 3. 10. 6(trans
ENG
8:31 PM
5/22/2023
好的, 现在因为我们只有从 Hugging Face 获得的训练分片
> OUT UNE
WSL: Ubuntu0
> TIMEUNE
okay now because we only have the training split from hugging face we can split it by
8:32 PM
5/22/2023
> OUT UNE 我们可以自己将其分为训练和验证. 我们将90
> WSL: Ubuntu
> TIMELNE
by ourselfinto a training and the validation we keep 9o of the data for training and.
5/22/2023
8:32 PM
> OUT UNE 我们可以自己将其分为训练和验证. 我们将90
> WSL: Ubuntu
> TIMELNE
10for validation Ln35, Col5 Spa
{ Pytho
8:32 PM 通
2 ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln35, Col5 Spaces:4 UTF-8 LF { Python 3. 10. 6(trans
ENG
8:32 PM
5/22/2023
> OUT UNE random _split 方法允许. 这是 Py To rch的一个方法, 允许我们使用
> WSL: Ubuntu
> TIMELINE
01
The method random split allows, it's a method from Py Torch that allows to split a 口 人 办 ENG
5/22/2023
8:33 PM
> OUT U NE 作为输入给出的尺寸来分割数据集
> WSL : Ubuntu > TIME LNE The method random split allows, it's a method from Py Torch that allows to split a
8:33 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME UNE
Ln38. Col67 (13 selected ) Spaces :4 UTF-8 LF (↓ Python3. 10. 6(transformer :conda)
ENG
8:33 PM
5/22/2023
> OUT U NE 所以在这种情况下, 这意味着将这个数据集分割成两个较小的
WSL : Ubuntu > TIME UNE So in this case, it means split this data set into two smaller data set, one of this 8:33 PM
5/22/2023
> OUT U NE 数据集, 一个具有这个尺寸, 另一个具有这个尺寸.
> WSL : Ubuntu > TIMELINE So in this case, it means split this data set into two smaller data set, one of this ENG
5/22/2023
8:33 PM
> OUT U NE 数据集, 一个具有这个尺寸, 另一个具有这个尺寸.
> WSL : Ubuntu > TIMELINE size and one of this size.
Ln38. Col61 Spa
Pytho
4. 10. 6 (tra
8:33 PM 口 通
ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 38, Col 80(11selected ) Spaces :4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
ENG
8:33 PM
5/22/2023
> OUT UNE 但让我们从 Torch 导入这个方法.
> WSL : Ubuntu > TIMELINE 通
ENG
5/22/2023
8:33 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln8, Col 49 Spaces:4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
ENG
5/22/2023
8:33 PM
> OUT UNE 我们也导入稍后会需要的那一个
> WSL: Ubuntu
> TIMELINE
1 A1
Let's also import the one that we will need later. spes. 4 urs u ( ython 3. o6 tansor
5/22/2023
8:33 PM
总加载器.
> OUT U NE
WSL : Ubuntu > TIME LNE
1
Total loader.
Ln3. Col46 Spaces :4 UTF-8 LF ( Python3. 10. 6(trans fo
ENG
8:33 PM
P
5/22/2023
> OUT U NE 和随机分割.
> WSL : Ubuntu > TIMELINE 1 A1
And random split.
Ln3, Col55 Spaces:4 UTF-8 LF Python3. 10. 6(transformer :cond
ENG
5/22/2023
8:33 PM
> OUTUNE 现在我们需要创建数据集
WSL: Ubuntu00
> TIMELINE Now we need to create the dataset.
n40, Col 5 Spaces4 UTF-8 LF( Python 3. 10. 6(trans fo
ENG
8:34 PM
5/22/2023
> OUTUNE 我们的模型将使用的数据集, 以便直接访问张量, 因为现在
WSL: Ubuntu00
> TIMELINE The dataset that our model will use to access the tensor directly because now we just B:34 PM
5/22/2023
> OUT UNE 我们刚刚创建了 tokenizer.
> WSL : Ubuntu > TIME LNE created the tokenizer.
Ln40. Col SSpaces4 UTF-8 LF Python 3. 10. 6(transformer :co 通
ENG
5/22/2023
8:34 PM
> OUT UNE
> WSL: Ubuntu
> TIMEUNE
Ln33, Col6 Spaces:4 UTF-8 LFPython3. 10. 6(transfo
ENG
8:34 PM
5/22/2023
而我们刚刚加载了数据.
> OUT U NE
> WSL : Ubuntu > TIME UNE
ENG
8:34 PM
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 30, Col53 Spaces :4 UTF-8 LFPython3. 10. 6(trans fo
ENG
8:34 PM
5/22/2023
I 但我们还需要创建模型将使用的张量.
> OUT U NE
> WSL : Ubuntu > TIME UNE But we need to create the tensors that our model will use. rs u
yhon 3. io. tansto
8:34 PM
5/22/2023
I > OUT U NE 那么, 让我们创建数据集
> WSL : Ubuntu > TIME UNE So let's create the dataset.
Ln40. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(transfo
ENG
5/22/2023
8:34 PM
I > OUT U NE 我们称之为双语数据集
> WSL : Ubuntu > TIME LNE Let's call it bilingual dataset.
Ln40. Col5 Spaces4 UTF-8 UF Python 3. 10. 6(trans fo
ENG
8:34 PM
5/22/2023
I > OUT U NE
> WSL : Ubuntu > TIME LNE
Ln40. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(transfo
8:34 PM
5/22/2023
> OUT U NE 为此, 我们创建了一个新文件
> WSL : Ubuntu > TIME LNE And for that we create a new file.
Ln40. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(trans fo
ENG
8:34 PM
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln40, Col5 Spaces :4 UTF-8 LFPython3. 10. 6(transformer :co
Z
ENG
5/22/2023
8:34 PM
> OUT UI NE 这里我们也导入了 torch.
> WSL : Ubuntu > TIME LINI
Also here we import to rch.
Ln1. Col1 Spaces4 UTF-8 LF{ Python 3. 10. 6 (t
B:34 PM
2 ENG
5/22/2023
就这样.
> OUT U NE
> WSL : Ubuntu > TIME LINI And that'sit.
Ln1. Col1 Spaces4 UTF-8 LFPython 3. 10. 6 (t 办 ENG
8:34 PM
5/22/2023
我们将调用数据集, 我们称之为双语数据集.
> OUTUNE
WSL: Ubuntu20
> TIMELNI
We will call the dataset, we will call it bilingual dataset. urs u
pyton 通
2 ENG
5/22/2023
8:35 PM
> OUT U NE
> WSL : Ubuntu 20
> TIMELINE
n5. Col7
es4 UIF-8 LFPython 3. 10. 6 (tra
8:35 PM
2 ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE 1 A0
okay
5/22/2023
8:35 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 1 A0
n7. Col5
UIF-8 LF{ Pytho
8:35 PM
ENG
5/22/2023
> OUT U NE 像往常一样, 我们定义了构造函数.
> WSL : Ubuntu > TIME LINI
1 A0
4. 10. 6(
8:35 PM 通
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE n7, Col. 12
8:35 PM
ENG
5/22/2023
在这个构造函数中, 我们需要给它从 Hugging Face 下载的数据集.
> OUT U NE
> WSL : Ubuntu > TIME LINI And in this constructor, we need to give him the dataset downloaded from Hugging Face.
8:35 PM
> OUT U NE
> WSL : Ubuntu 00
> TIMELNE
n 7. Col26
UIF-8 LFPython 3. 10. 6 (t
ENG
8:35 PM
5/22/2023
源语言的tokenizer, 目标语言的tokenizer, 源语言的名称, 目标语言的
> OUT UN > TIME UNE source language, the name of the source language, the name of. the. target language,
WSL: Ubuntu
ENG
5/22/2023
8:35 PM
> OUTUNE 名称, 以及我们将使用的序列长度.
> WSL: Ubuntu00
> TIMELNE
source language, the name of the source language, the name of the target language,
B:35 PM
名称, 以及我们将使用的序列长度.
> OUTUNE
> WSL: Ubuntu00
> TIMELINI
4. 10. 6(tra
ENG
8:35 PM 口 通
P
5/22/2023
> OUT UI NE
> WSL: Ubuntu00
> TIMELINE
n7, Col85
4. 10. 6 (tra
8:35 PM
ENG
5/22/2023
好的, 我们保存所有这些值.
> OUT U NE
> WSL : Ubuntu > TIME LNE Okay, we save all these values.
Ln11. Col9 Spaces4 UTF-8 LFPython 3. 10. 6(tra
8:35 PM
2 ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LN E
4. 10. 6(tra
8:35 PM
2 ENG
5/22/2023
我们还可以保存用于为模型创建张量的特定标记.
> OUT U NE
> WSL : Ubuntu > TIME UNE We can also save the tokens, the particular tokens that we will use to create the 通 人
5/22/2023
8:36 PM
我们还可以保存用于为模型创建张量的特定标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE tensors for the model.
Ln 16, Col9 Space Pytho
4. 10. 6(
8:36 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln16, Col9 Spaces4 UTF-8 LFPython 3. 10. 6 (tra
ENG
8:36 PM
5/22/2023
所以我们需要句子开始、句子结束和填充标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE So we need the start of sentence, end of sentence, and the padding to ken.
8:36 PM
ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln16. Col9 Spaces4 UTF-8 LFPython 3. 10. 6(tr
ENG
8:36 PM
5/22/2023
> OUT UNE 那么, 我们如何将句子开始的标记转换为数字, 即输入 ID > WSL : Ubuntu > TIMELINE So how do we convert the token start of sentence into a number, into the input l D? 人
5/22/2023
8:36 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
n 16. Col9 Spa
UTF-8 LF{ Python 3. 10. 6(tra 办 ENG
8:36 PM
tokenizer 有一个专门的方法来完成这个任务, 所以让我们来做吧.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
There is a special method of the tokenizer to do that, so let's do it. phon
ENG
/22/2023
8:36 PM
> OUTUNE
WSL: Ubuntu00
> TIMELNE
4. 10. 6 (tra
ENG
8:36 PM
5/22/2023
这是句子开始的标记.
> OUT UNE
WSL: Ubuntu00
> TIMEUNE
So this is the start of sentence to ken. n16col1 Spaces 4 urs yhon
5/22/2023
8:36 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 16, Col24
Spaces:4 UTF-8 LFPython 3. 10. 6(tra
ENG
8:36 PM
5/22/2023
> OUTUNE 我们希望将其构建为张量.
WSL: Ubuntu10
> TIMELNE
We want to build it into a tensor.
Ln16. Col26 Spaces4 UTF-8 LF( Python 3. 10. 6(ta
ENG
8:36 PM 通
5/22/2023
> OUTUNE
WSL: Ubuntu10
> TIMELINE Ln16. Col29 Spaces4 UTF-8 LF{ Python ENG
8:36 PM 口
5/22/2023
> OUTUNE 这个张量将只包含一个数字, 由.. 给出. 我们可以使用
> WSL: Ubuntu00
> TIMELNE
This tensor will contain only one number, which is given by... We can use this ENG
5/22/2023
8:36 PM
> OUT U NE 这个张量将只包含一个数字, 由.. 给出. 我们可以使用
> WSL : Ubuntu > TIME LNE tokenizer from the source or the target. 16. co46
Z
5/22/2023
8:36 PM
源语言或目标语言的 tokenizer.
> OUT U NE
> WSL : Ubuntu > TIME LNE tokenizer from the source or the target. 16. col46 spaces4urs u Python 3. 10. 6(trar
Z
ENG
5/22/2023
B:36 PM
这无关紧要, 因为它们都包含这些特定标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE It doesn't matter because they both contain these particular tokens. yuh on 3. 10. 6(trar
Z
ENG
5/22/2023
B:36 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 3. 10. 6 (tra
8:36 PM
ENG
5/22/2023
> OUT U NE 这是将标记转换为数字的方法.
> WSL : Ubuntu > TIME LNE This is the method to convert the token into a number.
UTF-8 LFPython3. 10. 6(trans
ENG
8:36 PM 通
Z
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LN E
4. 10. 6(trar
ENG
8:37 PM
5/22/2023
> OUT U NE 所以句子开始, 这个标记的类型, 这个张量的类型是:我们
> WSL Ubuntu > TIME LNE so start of sentence and the type of this token of this tensor is we want it longc
P
/22/2023
:37 PM
> OUT UNE 希望它是长整型, 因为词汇表可能超过32位长, 即
> WSL: Ubuntu
> TIMELNE
so start of sentence and the type of this token of this tensor is we want it longm 通 人
ENG
5/22/2023
8:37 PM
> OUTUNE 希望它是长整型, 因为词汇表可能超过32位长, 即
> WSL: Ubuntu > TIME LNE 通 人
5/22/2023
8:37 PM
> OUT UNE 词汇表的大小.
> TIMELNE
8:37 PM
5/22/2023
> OUTUNE 所以我们通常使用64位长整型, 对句子结束和填充标记也
xwum o because the vocabulary can be more than 32 bit long the vocabulary size so we usually.
> TIME UNE
5/22/2023
8:37 PM
> OUTUNE 所以我们通常使用64位长整型, 对句子结束和填充标记也
> WSL: Ubuntu
> TIMEUNE
use the n. 16, Col97
:37 PM
5/22/2023
> OUT UNE 所以我们通常使用64位长整型, 对句子结束和填充标记也
> WSL: Ubuntu
> TIMEUNE
long64bit and we do the same for the end of sentence and the padding to kentao 口 人
ENG
5/22/2023
8:37 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln17, Col9 Spaces4 UIf-8 IFPython 3. 10. 6trar
ENG
8:37 PM
5/22/2023
> OUTUNE 我们还需要定义这个数据集的长度方法, 它告诉数据集本身的
> WSL: Ubuntu00
> TIMELINE We also need to define the length method of this data set, which tells the length of ENG
5/22/2023
8:38 PM
> OUTUNE 我们还需要定义这个数据集的长度方法, 它告诉数据集本身的
WSL: Ubuntu20
> TIMELINE the data set itself, so basically just the length of the dataset from Hugging Face, m
ENG
5/22/2023
8:38 PM
长度, 基本上就是 Hugging Face 数据集的长度, 然后我们需要
> OUTUNE
WSL: Ubuntu10
> TIMELINE the data set itself, so basically just the length of the dataset from Hugging Face, m 口 人
ENG
5/22/2023
8:38 PM
定义 get lt em 方法.
> OUT U NE
> WSL : Ubuntu > TIME UNE and then we need to define the get l tem method. s
Spaces 4 UTF-8 LFPython3. 10. 6(trans fo
ENG
8:38 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME UNE
Ln23. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
8:38 PM
好的, 首先我们将从 Hugging Face 数据集中提取原始对.
> OUTUNE
WSL: Ubuntu10
> TIMEUNE
Okay, first of all we will extract the original pair from the Hugging Face dataset.
8:38 PM 口
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMELNE
1 A0
Ln 25, Col9 Spaces4 UTF-8 LF ( Python3. 10. 6(trans fo
ENG
8:38 PM
5/22/2023
> OUT U NE 然后我们提取源文本和目标文本.
WSL : Ubuntu > TIME LNE Then we extract the source text and the target text. sa urs u 4 ython 3ios tans
ENG
5/22/2023
8:38 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 26. Col9 Spaces4 UTF-8 LF Python 3. 10. 6(trans
5/22/2023
8:38 PM
> OUTUNE 最后, 我们将每个文本转换为标记, 然后转换为输入 ID.
WSL: Ubuntu
> TIMELNE
And finally we convert each text into tokens and then into input IDs. pthon 人
5/22/2023
8:39 PM
这是什么意思?
> OUT U NE
> WSL : Ubuntu > TIMELINE What does it mean?
Ln 29. Col9 Spaces 4 UTF-8 LF Python 3. 10. 6(transformer :cond
ENG
8:39 PM 通
5/22/2023
> OUT UNE
tokenizer首先将句子拆分为单个单词, 然后将其映射到词汇表中
WSL: Ubuntu00
> TIMELINE The tokenizer will first split the sentence into single words and then will map each.
8:39 PM 人
ENG
5/22/2023
> OUT UNE tokenizer 首先将句子拆分为单个单词, 然后将其映射到词汇表中
WSL : Ubuntu > TIME UNE word into its corresponding number in the vocabulary and it. will do it in one pass.
ENG
5/22/2023
8:39 PM
> OUT U NE 对应的数字, 并且它只进行一次遍历.
WSL : Ubuntu 00
> TIMELNE
only.
4 UTF-8 LFPython 3. 10. 6(trans
ENG
8:39 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln29, Col9 Spaces4 UTF-8 LFPython 3. 10. 6(transfc
8:39 PM
5/22/2023
这是通过 encode 方法的 ids 完成的.
> OUT U NE
WSL : Ubuntu > TIMELINE 通
Z
ENG
5/22/2023
8:39 PM
> OUT U NE
> WSL: Ubuntu00
> TIMELNE
Ln30, Col9 Spaces 4 UTF-8 LF ↓ Python3. 10. 6(trans fon
ENG
8:39 PM 口
5/22/2023
> OUT UNE 这为我们提供了输入 ID, 即原始句子中每个单词对应的数字
WSL: Ubuntu00
> TIMELNE
This gives us the input ids, so the numbers corresponding to each word in the octa
B:39 PM 口
5/22/2023
> OUT UNE 这为我们提供了输入 ID, 即原始句子中每个单词对应的数字
> WSL : Ubuntu > TIME LNE
5/22/2023
B:39 PM
并以数组形式给出.
> OUT U NE
> WSL : Ubuntu > TIME LNE original sentence and it will be given as an array. spaes urs u
yhon aios ctansto 8:39 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 29. Col28(39 selected ): Spaces 4 UTF-8 LF( Python 3. 10. 6 (transfc
8:40 PM
5/22/2023
我们对解码器也做了同样的事情.
> OUT U NE
> WSL : Ubuntu > TIME LNE We did the same for the decoder.
Ln32, Col9 Spaces4 UTF-8 LF Python 3. 10. 6(trans fo 通
Z
ENG
5/22/2023
8:40 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln32, Col9 Spaces4 UTF-8 LF ( Python 3. 10. 6(transfo
8:40 PM
5/22/2023
现在, 如你所记, 我们还需要将句子填充到序列长度.
> OUT UNE
> WSL: Ubuntu
> TIMELNE
B:40 PM
> OUT U NE
WSL: Ubuntu
> TIMELNE
Ln 32, Col9 Spaces 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
ENG
5/22/2023
8:40 PM
> OUT U NE 这非常重要, 因为我们希望模型始终能够工作. 我的意思
WSL : Ubuntu > TIME LNE This is really important because Pytho
3:40 PM 通
5/22/2023
> OUT U NE 这非常重要, 因为我们希望模型始终能够工作. 我的意思
> WSL : Ubuntu > TIME UNE
8:40 PM
5/22/2023
> OUT UNE 这非常重要, 因为我们希望模型始终能够工作. 我的意思
xwu m owe want our model to always work I mean the model always works with a fixed length.
> TIME UNE
8:40 PM
5/22/2023
> OUTUNE 但我们并不是每个句子都有足够的单词, 所以我们使用填充
WSL: Ubuntu00
> TIMELINE sequence length but we don't have enough words in every sentence so we use the 8:40 PM 口
5/22/2023
> OUT U NE 但我们并不是每个句子都有足够的单词, 所以我们使用填充
WSL : Ubuntu > TIMELINE padding token so
Ln32, Col9
UIF-8 LFPytho
8:40 PM
P
5/22/2023
> OUTUNE 但我们并不是每个句子都有足够的单词, 所以我们使用填充
> WSL: Ubuntu00
> TIMELNE
this pad here as the padding token to fill the sentence until it reaches the sequence B:40 PM
ENG
5/22/2023
长度.
> OUT U NE
> WSL : Ubuntu > TIME LNE
8:40 PM
5/22/2023
> OUT UNE 所以我们计算需要为编码器侧和解码器侧添加多少个填充标记
> TIMELNE
5/22/2023
8:40 PM
> OUT U NE 所以我们计算需要为编码器侧和解码器侧添加多少个填充标记
> WSL : Ubuntu > TIME UNE and for the decoders ide
Ln32, Col9 Spac
UIF-8 LFPython :40 PM 通
Z
P
ENG
5/22/2023
> OUT U NE 所以我们计算需要为编码器侧和解码器侧添加多少个填充标记
> WSL : Ubuntu > TIME UNE
Python
:40 PM
ENG
5/22/2023
> OUT U NE 这基本上是我们需要达到序列长度的数量,
> WSL : Ubuntu > TIME UNE
UTF-8 LFPython3. 10. 6 Ctrans
ENG
B:40 PM
5/22/2023
> OUT UI NE 这基本上是我们需要达到序列长度的数量.
> WSL : Ubuntu > TIME UNE which is basically how many we need to reach the sequence length. wyhon 3. o ctas
ENG
5/22/2023
8:40 PM
> OUT UI NE
> WSL : Ubuntu > TIME UNE Ln32, Col44 Spaces4 UTF-8 LF( Python 3. 10. 6(transfc
ENG
5/22/2023
B:40 PM
> OUT U NE 减去二, 为什么这里要减去二?
> WSL : Ubuntu > TIME LNE Minus two, why minus two here?
Ln32. Col74 Spaces 4 UTF-8 LF { Python 3. 10. 6(trans fo
ENG
8:40 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln32. Col74 Spaces4 UTF-8 LF Python 3. 10. 6(transformer :co
ENG
5/22/2023
8:41 PM
> OUT U NE 所以我们已经有这么多标记, 我们需要达到这个数量, 但
> WSL : Ubuntu > TIMELINE So we already have this amount of tokens, we need to reach this one, but we will add.
ENG
8:41 PM
> OUTUNE 我们还会在编码器侧添加句子的起始标记和结束标记, 所以
WSL: Ubuntu00
> TIMELINE also the start of sentence token and the end of sentence token to the encoder side,
5/22/2023
8:41 PM
> OUT U NE 我们还会在编码器侧添加句子的起始标记和结束标记, 所以
> WSL : Ubuntu > TIME LNE So we also have minus two here. col32(23elketed) Spx
4 UTF-8 LFPython 8:41 PM
5/22/2023
> OUT U NE 这里也要减去二
> WSL : Ubuntu > TIME LNE
ENG
5/22/2023
8:41 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 32. Col74 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
5/22/2023
8:41 PM
这里只减去一.
> OUT U NE
> WSL : Ubuntu > TIME LNE here only minus one if you remember my previous video when we do. the training we add
8:41 PM
5/22/2023
> OUT U NE 如果你还记得我之前的视频, 我们在训练时, 只在解码器侧
ws uu n o. here only minus one if you remember my previous video when we do. the training we add
> TIMELNE
5/22/2023
8:41 PM
> OUT UNE 如果你还记得我之前的视频, 我们在训练时, 只在解码器侧
> TIME LNE only the start of sentence token to the decoder side and then in. the label we only 8:41 PM
5/22/2023
添加句子的起始标记, 然后在标签中只添加句子的结束标记.
> OUT UNE > TIME LNE only the start of sentence token to the decoder side and then in. the label we only B:41 PM
ENG
5/22/2023
所以在这种情况下, 我们只需要为句子添加一个特殊标记.
> OUT U NE
WSL : Ubuntu > TIME L NE
add theend
B:41 PM
Z
5/22/2023
所以在这种情况下, 我们只需要为句子添加一个特殊标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE of sentence token so in this case we only need to add one token special token to the 5/22/2023
8:41 PM
所以在这种情况下, 我们只需要为句子添加一个特殊标记.
> OUT U NE
WSL : Ubuntu > TIME UNE sentence Ln33, Col74 S
Pytho
3:41 PM
5/22/2023
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln 33, Col74 Spaces 4 UTF-8 LF Python3. 10. 6(transformer :con
ENG
8:41 PM
5/22/2023
> OUT U NE 我们还确保我们选择的序列长度足以表示数据集中的所有句子
WSL : Ubuntu > TIME UNE we also make sure that this sequence length that we have chosen is enough to ctm 口
3:41 PM
> OUT U NE 我们还确保我们选择的序列长度足以表示数据集中的所有句子
WSL : Ubuntu > TIME LNE represent all the uh sentences in our data set and if we chose a too small one we
8:41 PM 口
5/22/2023
> OUT U NE 如果我们选择得太小, 我们希望抛出一个异常. 所以
> WSL : Ubuntu > TIME UNE want to throw a
Ln35, Col9 S 通
5/22/2023
8:41 PM
基本上, 这个填充标记的数量永远不应该变为负数.
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln35, Col 9
( Pytho
4. 10. 6 (tra
8:42 PM
ENG
5/22/2023
> OUT U NE 基本上, 这个填充标记的数量永远不应该变为负数.
WSL : Ubuntu > TIME LNE raise an exception so if so basically this number of padding toxins should. never 办 ENG
5/22/2023
8:42 PM
基本上, 这个填充标记的数量永远不应该变为负数.
> OUT U NE
WSL : Ubuntu > TIMELINE become negative Pytho 3. 10. 6 (tra
B:42 PM
ENG
5/22/2023
> OUT U NE
WSL : Ubuntu > TIMELINE Ln35, Col9 Spaces4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
5/22/2023
8:42 PM
> OUT U NE 好的, 现在让我们为编码器输入、解码器输入以及标签构建
WSL : Ubuntu > TIME LNE Okay, now let's build the two tensors for the encoder input and for the decoder 5/22/2023
8:42 PM
> OUT U NE 好的, 现在让我们为编码器输入、解码器输入以及标签构建
WSL : Ubuntu > TIME LNE input, but also for the label.
Ln38. Col9 S
Pytho
B:42 PM
P
5/22/2023
两个张量.
> OUT U NE
WSL : Ubuntu > TIME LNE input, but also for the label.
Ln38, Col9 Spaces4 UTF-8 LF ( Python 3. 10. 6(transfo
8:42 PM 口
5/22/2023
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln38. Col9 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :cond
8:42 PM
5/22/2023
> OUT U NE 所以一个句子将被发送到编码器的输入, 一个句子将被发送到
WSL : Ubuntu > TIME UNE So one sentence will be sent to the input of the encoder, one sentence will be sent 口
ENG
3:42 PM
> OUTUNE 所以一个句子将被发送到编码器的输入, 一个句子将被发送到
WSL: Ubuntu00
> TIMELINE to the input of the decoder, and one sentence is the one that we expect rasi the output 8:42 PM
ENG
> OUT UNE 解码器的输入, 还有一个句子是我们期望的解码器输出.
> TIMELINE to the input of the decoder, and one sentence is the one that we expect asi the output 8:42 PM
5/22/2023
解码器的输入, 还有一个句子是我们期望的解码器输出.
> OUT U NE
> WSL : Ubuntu > TIME UNE of the decoder.
L n38, Col 9
Pytl
8:42 PM
P
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln38, Col9 Spaces 4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
8:42 PM
5/22/2023
> OUT U NE 而这个输出我们称之为标签.
> WSL : Ubuntu > TIME UNE And that output we will call label.
Ln38. Col9 Spaces4 UTF-8 LF { Python 3. 10. 6(trans fo
ENG
8:42 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln38, Col9 Spaces4 UTF-8 LF Python 3. 10. 6(transfo
8:42 PM
5/22/2023
> OUT U NE 通常它被称为目标或标签.
WSL : Ubuntu > TIME LNE Usually it's called target or label.
Ln38. Col9 Spaces4 UTF-8 LF ↓ Python 3. 10. 6(trans fc
ENG
8:42 PM 人
5/22/2023
我称之为标签.
> OUT U NE
WSL : Ubuntu > TIME LNE
Icall itlabel.
Ln38, Col9 Spaces 4 UTF-8 LF Python3. 10. 6(trans fo
ENG
8:42 PM
5/22/2023
> OUT UNE 我们可以截取起始张量.
WSL: Ubuntu
> TIMELNE
1 A0
We can cut the tensor of the start.
Ln 38, Col34 Spaces 4 UTF-8 LF( Python 3. 10. 6(transformer :conda) 通
5/22/2023
8:43 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 40. Col17 Spaces4 UTF-8 LF ( Python 3. 10. 6(transfo
5/22/2023
8:43 PM
首先是这个句子的起始标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE First is this start of sentence token.
Ln40. Col 17 Spaces4 UTF-8 LF( Python 3. 10. 6(transfo
ENG
5/22/2023
8:43 PM
> OUT U NE
WSL: Ubuntu
> TIMELNE
Ln 40. Col 18 Spaces :4 UTF-8 LF ( Python3. 10. 6(transformer :conda)
8:43 PM 口
Z
5/22/2023
然后是源文本的标记.
> OUT U NE
> WSL : Ubuntu > TIME LNE Then the tokens of the source text.
Ln41. Col 17 Spaces4 UTF-8 LFPython 3. 10. 6(trans fo
ENG
8:43 PM
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln41. Col17 Spaces:4 UTF-8 LF ( Python 3. 10. 6 (transfo
8:43 PM
5/22/2023
然后是句子的结束标记.
> OUT U NE
> WSL : Ubuntu > TIME UNE Then the end of sentence to ken. 办 ENG
8:43 PM
5/22/2023
> OUTUNE
46
WSL: Ubuntu > TIME LNE
Ln 42. Col 19 Spaces 4 UTF-8 LF { Python3. 10. 6(transformer :conda)
5/22/2023
8:43 PM
42
1/2
_int =θ,*, out: Tensor | None = None )-> Tensor 43 然后添加足够的填充标记以达到序列长度
> OUTUNE
WSL: Ubuntu
> TIMELNE
And then enough padding tokens to reach the sequence length.
Python 3. 10. 6 (transfo
ENG
5/22/2023
8:43 PM
self. eos _token,
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln43, Col17 Spaces4 UTF-8 LF { Python 3. 10. 6(transfc
ENG
5/22/2023
8:43 PM
42
sel f. eos_token,
43 我们已经计算出需要为这个句子添加多少个填充标记.
> OUTUNE
WSL: Ubuntu > TIME LNE We already calculated how many padding tokens we need to. add to. this sentence.
5/22/2023
8:43 PM
self. eos _token, 所以让我们开始吧.
> OUT U NE
> WSL : Ubuntu > TIMELINE So let'sjust do it.
Ln43. Col17 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
8:43 PM
self. eos _token,
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln43, Col17 Spaces4 UTF-8 LF { Python 3. 10. 6(transfo
8:43 PM
5/22/2023
self. eos_token,
43
torch. tensor ([self. pad _token 】*
padding _tokens, dtyp
rch. int64
> OUTUNE 这是编码器输入, 所以让我在这里写一些注释, 这是在源
> WSLUbuntu > TIMELINE and this is the encoder input so let me write some comment here this is a dd so S and.
5/22/2023
B:44 PM
torch. tensor (enc _input _tokens, d type =torch. in t64),
coro
([self. pad_token
dding_tokens, dtype
orch. int64
> OUTUNE 这是编码器输入, 所以让我在这里写一些注释, 这是在源
WSL: Ubuntu > TIME UNE Sos to the source text then we build the decoder s
s4 UTF-8 LFPython 3. 10. 6 (tra
B:44 PM
5/22/2023
43
self. eos_token
[self. p
torch. in t64
> OUTUNE 文本中添加起始标记, 然后我们构建解码器, 这也是标记的
WSL: Ubuntu
> TIMELNE
Sosto the source text then we build the decoder s
4 UTF-8 LFPython
5/22/2023
B:44 PM
44
43
torch. tensor([self. pad_eoken]*er
self. eos_token,
ch. int64
> OUTUNE 文本中添加起始标记, 然后我们构建解码器, 这也是标记的
WSL: Ubuntu > TIME LN E
UTF-8 LF{ Pytho
8:44 PM
5/22/2023
46 文本中添加起始标记, 然后我们构建解码器, 这也是标记的
> OUTUNE
WSL: Ubuntu
> TIMELNE
which is also concatenation of tokens. nso. co7 spa
4. 10. 6 C
UTF-8 LFPython B:44 PM 口 人
5/22/2023
I
1/2
(tensors: Tuple[ T
_int=θ,*,
1 List [ Tensor ], dim:
None)-> Tensor
> OUTUNE
50
> WSL: Ubuntu > TIME LNE
ENG
8:44 PM 口
5/22/2023
I _int =θ,*, out : Tensor | None = None )-> Tensor (tensors : Tuple [ Tensor,
...]| List [ Tensor ], dim :
> OUT U NE 国
WSL: Ubuntu
> TIMELNE
Ln 50. Col17 Spaces :4 UTF-8 LF Python3. 10. 6(transformer :con
ENG
8:44 PM 口
5/22/2023
> OUT U NE 在这种情况下, 我们没有句子的起始标记, 我们只有句子的
WSL : Ubuntu > TIME LNE In this case, we don't have the start of sentence, we just have. the, we don't have B:44 PM
ENG
5/22/2023
> OUT U NE 在这种情况下, 我们没有句子的起始标记, 我们只有句子的
> WSL : Ubuntu > TIME LNE the end of sentence, we just have the start of sentence.
UTF8 LF Python 3. 10. 6t
8:44 PM 口
5/22/2023
decoder _input = torch. cat (
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln50. Col17 Spaces:4 UTF-8 LF { Python 3. 10. 6(trans 办 ENG
5/22/2023
8:44 PM
48 最后, 我们添加了足够的填充标记以达到序列长度, 我们
> OUTUNE
> TIME LNE
8:45 PM
5/22/2023
52
to rchetensor([self. pad_token]*dec_n
rch. int64)
> OUTUNE 已经计算出需要多少个, 现在使用这个值, 然后我们构建
> WSL: Ubuntu > TIME UNE calculated how many we need just use this value now and then we build the label 办 ENG
5/22/2023
8:45 PM
torch. tensor ([self. pad _token ]*dec _num _padding _to k
ch. int64)
> OUTUNE 已经计算出需要多少个, 现在使用这个值, 然后我们构建
> WSL: Ubuntu > TIME UNE
Pytho
4. 10. 6(t
8:45 PM 人
5/22/2023
{)_tensor_str
nsor_docs
54 标签. 在标签中, 我们只添加句子的结束标记
> OUTUNE
WSL: Ubuntu > TIME LNE
Ln58, Col29 Spa
4. 10. 6(tran
Python 8:45 PM
ENG
5/22/2023
label =
(tensors : Tuple [ Tensor,
..]| List [ Tensor ], dim :
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 59. Col 17
Spaces :4 UTF-8 LF { Python3. 10. 6(trans fo
ENG
8:45 PM
5/22/2023
label = torch. cat (
= None, device :
> OUT U NE 让我复制一下, 这样更快
WSL : Ubuntu > TIME UNE Let me copy, it's faster.
Ln 60, Col30
4 UTF-8 LFPython 3. 10. 6(trans 人
ENG
5/22/2023
8:45 PM
label = torch. cat (
torch. tensor (dec _input _tokens, d type =torch. int64),
self. eos_token,
> OUTUNE
torch. tensor ()
WSL : Ubuntu > TIME UNE
Ln 52, Col91 (74 selected ) Spaces 4 UTF-8 LF { Python3. 10. 6(transformer :conda) 办 ENG
5/22/2023
8:45 PM
55
label= torch. cat(
> OUTUNE 是的, 因为我们需要的填充标记数量与解码器输入相同.
WSL: Ubuntu > TIME UNE Yeah, because we need the same number of padding tokens as for the decoder input.
5/22/2023
8:45 PM
label = torch. cat (
torch. tensor (dec _input _tokens, d type =torch. int64),
self. eos_token,
> OUTUNE
torch. tensor ([self. pad _token ]*dec _num _padding _tokens, dtype=torch. int64)
WSL: Ubuntu > TIME LNE
Ln 48, Col 22(13 selected )
Spaces 4 UTF-8 LFPython 3. 10. 6(transfo
8:45 PM
5/22/2023
torch. tensor ([self. pad _token ]*dec _
padding _to k
ch. int64 让我们再检查一遍, 只是为了调试, 确保我们确实达到了
> OUTUNE
> WSL: Ubuntu > TIME LNE And let's double, just for debugging, let's double check that we actually reached the.
ENG
5/22/2023
8:46 PM
序列长度.
> OUT UNE
WSL: Ubuntu
> TIMELNE
1 A0
And let's double, just for debugging, let's double check that we actually reached the.
ENG
5/22/2023
8:46 PM
> OUT U NE
WSL : Ubuntu > TIME LN E
1 A0
Ln64. Col16 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
5/22/2023
8:46 PM
> OUT UNE 好的, 既然我们已经进行了这个检查, 让我在这里也写一些
> WSL: Ubuntu
> TIMELNE
8:46 PM
注释. 这里我们只添加 eos, 不, 这里添加 sos 到解码器
> OUT U NE
WSL : Ubuntu > TIMELINE okay now that we have made this check let me also write some comments here here we.
8:46 PM
5/22/2023
注释. 这里我们只添加 eos, 不, 这里添加 sos 到解码器
> OUTUNE
WSL: Ubuntu0 O
> TIMELINE are only adding eos no here sos to the decoder input and here is add eos to the label 5/22/2023
8:46 PM
72 输入, 这里是:在标签中添加 eos, 这是我们期望从解码器
> OUTUNE
WSL: Ubuntu00
> TIME UNE what we expect as output from the decoder a Spac
Pytho 口
P
74 得到的输出.
> OUTUNE
> WSL: Ubuntu > TIMELINE what we expect as output from the decoder os spaces4 ufr8 u o ython 3. 1o6 trans to ENG
5/22/2023
8:46 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln57. Col74 Spaces4 UTF-8 LFPython 3. 10. 6(transfc
Z
5/22/2023
8:47 PM
72
> OUTUNE 现在我们可以返回所有这些张量, 以便我们的训练可以使用
WSL: Ubuntu00
> TIMELNE
Now we can return all these tensors so that our training can. use the m. thon
4. 10. 6
5/22/2023
8:47 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln70. Col9 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :cond
ENG
5/22/2023
8:47 PM
72
> OUTUNE 我们返回一个由编码器输入组成的字典.
> WSL: Ubuntu
> TIMELNE
We return a dictionary comprised of encoder input. sa ur8 ur
ython 3. no trs 人 办 ENG
5/22/2023
8:47 PM
> OUTUNE
76
> WSL: Ubuntu > TIMELINE Ln 71. Col19 Spaces4 UTF-8 LF ( Python 3. 10. 6 (trans
5/22/2023
8:47 PM
encoder _input ":encoder _input > OUT U NE 编码器输入是什么?
> WSL : Ubuntu > TIME UNE What is the encoder input?
Ln 71. Col43 Spaces4 UTF-8 LF ( Python 3. 10. 6(transfo 办 ENG
5/22/2023
8:47 PM
> OUT U NE 它基本上是偏移的序列长度
> WSL : Ubuntu > TIME UNE It's basically off-site Sequence length. n71. col4s Spaces4uf-s u F python 3. 10. 6(transto
ENG
5/22/2023
8:47 PM
n coder _input ":encoder _input #( Seq _ Len ),
> OUT U NE 然后我们有解码器输入.
> WSL : Ubuntu > TIME LNE Then we have the decoder input.
Ln72. Col 13 Spaces4 UTF-8 LF Python 3. 10. 6(trans
ENG
5/22/2023
8:47 PM
72
> OUTUNE 这也是一个由序列长度数量的标记组成的
> WSL: Ubuntu > TIMELINE 4 A0 which is also just a sequence length number of. tokens.
UTF-8 LFPython
4. 10. 6(trans fc
ENG
8:47 PM 人
5/22/2023
encoder _input ":encoder _input #( Sed Len ),
"decoder _input ":decoder _input #( Seq _ Len )
> OUT U NE 我这里漏了一个逗号.
WSL : Ubuntu > TIMELINE I forgot a comma here.
Ln 71. Col50 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
8:47 PM
"encoder _input ":encoder _input,=( Seq _ Len )
'decoder _input ":decoder _input #( Seq _ Len )
> OUT U NE 然后我们有编码器掩码.
WSL : Ubuntu > TIMELINE ENG
5/22/2023
8:47 PM
encoder _input ":encoder _input, #( Seq _ Len )
> OUT U NE 那么编码器掩码是什么?
> WSL : Ubuntu > TIMELINE So what is the encoder mask?
Ln 73, Col13 Spaces 4 UTF-8 LF { Python3. 10. 6(trans fc
ENG
8:47 PM 口
5/22/2023
72
73
'decoder _input ":decoder _input,#( Seq _ Len )
> OUT U NE 如你所记, 我们通过添加填充标记来增加编码器输入句子的
> WSL Ubuntu > TIMELINE As you remember, we are increasing the size of the encoder input sentence by adding 人
5/22/2023
8:47 PM
72
"decoder _input ":d
decoder_input,#( Seq _ Len )
> OUT U NE 如你所记, 我们通过添加填充标记来增加编码器输入句子的
WSL : Ubuntu > TIME LNE padding tokens.
Ln 39, Col 22 (13. selected ) 通
5/22/2023
8:47 PM
'decoder _input ":
decoder _input,#( Seq_ Len) 长度.
> OUTUNE
76
WSL: Ubuntu > TIME LNE padding tokens.
Ln39, Col 22(13. selected ) Spaces :4 UTF-8 F( Python 3. 10. 6 (trans
8:47 PM
5/22/2023
72 "decoder _input ":
decoder _input,#( Seq _ Len )
> OUT U NE 但我们不希望这些填充标记参与自注意力机制.
WSL : Ubuntu > TIME UNE But we don't want these padding tokens to participate in the self-attention. otan
5/22/2023
8:47 PM
"decoder _input ":decoder _input,#( Seq _ Len )
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln45, Col14 Spaces:4 UTF-8 LF ( Python 3. 10. 6(transformer :conda)
5/22/2023
8:48 PM
72 "decoder _input ": decoder _input,#( Seq _ Len )
> OUT U NE 所以我们需要构建一个掩码, 表示我们不希望这些标记被自
> WSL : Ubuntu > TIMELINE So what we need is to build a mask that says that we don't want these i tokens. to be
ENG
5/22/2023
8:48 PM
73 decoder _inp
ut":decoder _input,#( Seq _ Len )
> OUT U NE 所以我们需要构建一个掩码, 表示我们不希望这些标记被自
WSL : Ubuntu > TIME LNE seen by the self-attention mechanism. n. col3 spm
4. 106 C
Python 8:48 PM 通
P
5/22/2023
t ":decoder _input,#( Seq _ Len )
> OUT U NE 注意力机制看到.
WSL : Ubuntu > TIME LNE 办 ENG
8:48 PM
5/22/2023
"decoder _input ":decoder _input,#( Seq _ Len )
> OUT UNE 75
WSL: Ubuntu
> TIMELNE
Ln 73, Col13 Spaces4 UTF-8 LF ( Python3. 10. 6(transfc
ENG
8:48 PM 口
5/22/2023
"decoder _in pu ut ":decoder _input,#( Seq_ Len)
> OUTUNE
75 因此我们为编码器构建掩码.
WSL: Ubuntu > TIME LNE And so we build the mask for the encoder ocol13 Spaces4 urf-8u ( Python 3. 06tansto 办 ENG
8:48 PM
5/22/2023
"encoder I'decoder n put ":decoder _input,#( Seq _ Len )
> OUT U NE
> WSL : Ubuntu > TIME LN E
1 A0
Ln 73. Col21 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
5/22/2023
8:48 PM
encoder _input ":encoder _input,#( Seq _ Len )
decoder _input,#( Seq _ Len)
> OUTUNE 我们如何构建这个掩码?
WSL: Ubuntu
> TIMELNE
1 A0
How do we build this mask?
Ln 73. Col27 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
8:48 PM
5/22/2023
72
decoder_input,#( Seq_ Len)
> OUTUNE 我们只需说明所有非填充标记都是正常的
WSL: Ubuntu
> TIMELNE
2 A0
We just say that all the tokens that are not padding are okay.
LF { Python 3. 10. 6(trans
ENG
5/22/2023
8:48 PM
"decoder _in "encoder _mal k ":【encoder _input ]
ut":
#( Seq_ Len)
> OUTUNE
> WSL: Ubuntu > TIME UNE Ln73. Col43 Spaces:4 UTF-8 LF { Python 3. 10. 6(transfc 办 ENG
5/22/2023
8:48 PM
le r_input,
#( Seq_ Len)
> OUTUNE
75 所有填充标记都是不正常的.
> WSL: Ubuntu > TIME UNE All the tokens that are padding are not okay. 3 spaces4 urs u
ython 3. 10. 6(trans
ENG
5/22/2023
8:48 PM
"encoder _ma E k ":【encoder _input ]
"decoder _in ut ":
decoder _input, #( Seq _ Len )
> OUT U NE
> WSL : Ubuntu > TIME UNE Ln 73, Col43 Spaces:4 UTF-8 LF { Python 3. 10. 6(transfc
ENG
5/22/2023
8:48 PM
72
oder_input,#( Seq_ L en)
> OUTUNE 我们也解冻以添加这个序列维度, 稍后还会添加批次维度.
> WSL: Ubuntu > TIME UNE We also unscr eeze to add this sequence dimension and also to add the batch dimension 人
8:48 PM
coder _input,#( Seq _ Len )
> OUT U NE 我们也解冻以添加这个序列维度, 稍后还会添加批次维度.
WSL : Ubuntu > TIME UNE later.
n73. Col74
:48 PM
P
5/22/2023
"encoder _mal k ":(encoder _input 1= self. pad_token). unsqueeze(e)
"decoder _inp
ut":
decoder_input,#( Seq _ Len )
add > OUT U NE
WSL: Ubuntu00
> TIMEUNE
Ln 73, Col 76
Spaces4 UTF-8 LFPython3. 10. 6(transformer :co 心
5/22/2023
8:48 PM
encoder _input":
( Seq_ Len
add
acosh 并将其转换为整数.
> OUTUNE
> WSLUbuntu > TIMELINE And we convert into integers.
Ln 73, Col 89
Spaces4 UTF-8 LF{ Python 3. 10. 6(trans
ENG
5/22/2023
8:48 PM
"decoder _inp
ut":
decoder _input,#( Seq _ Len )
> OUT U NE
> WSL Ubuntu > TIMELINE Ln 73, Col 94
Spaces4 UTF-8 LFPython 3. 10. 6(trans
Z
5/22/2023
8:48 PM
e(0). int()
> OUTUNE 所以这是一个序列长度为 1.
> WSL: Ubuntu > TIMELINE So this is one, one sequence length.
Ln 73, Col95 Spa
4 UTF-8 LF{ Python 3. 10. 6(trans
2 ENG
5/22/2023
8:48 PM
"decoder _in decoder _input,#( Seq_ Len)
> OUTUNE
76
WSL: Ubuntu > TIME LN E
Ln73, Col108
Spaces4 UTF-8 LF Python 3. 10. 6(trans 办 ENG
5/22/2023
8:48 PM
n coder _input ":
encoder _input,#( Seq_ L en)
er_input,
#( Seq_ Len
74 因为这将在自注意力机制中使用,
> OUTUNE
WSL: Ubuntu > TIME LNE because this will be used in the self-attention mechanism. rs u ptho
5/22/2023
8:48 PM
"encoder _input ":encoder _input, #( Seq _ Len )
"encoder _mask ":(encoder _input 1= self. pad_token). unsqueeze (e). unsqueeze(e). int() #(1, 1, Seq_ Len)
"decoder _input ":
decoder _input,#( Seq _ Len )
> OUT U NE
WSL : Ubuntu > TIME UNE
76
Ln74, Col13 Space:4 UTF-8 IF ( Python 3. 10. 6 (transo
8:48 PM
Z
5/22/2023
72
> OUTUNE 然而, 对于解码器, 我们需要一个特殊的掩码, 即因果
> WSL: Ubuntu
> TIMEUNE
however for the decoder we need a special mask that is a causal mask which means. that 5/22/2023
8:48 PM
71
coder_input,#( Seq_ L en)
72
> OUTUNE 掩码, 这意味着每个词只能看到前面的词, 并且每个词只能
> WSL: Ubuntu > TIME LNE however for the decoder we need a special mask that is a causal mask which means. that 5/22/2023
8:48 PM
72
encoder_input,#( Seq_ Len)
> OUTUNE 掩码, 这意味着每个词只能看到前面的词, 并且每个词只能
WSL: Ubuntu30
> TIMELNE
each word can only look at the previous words and each word can only look at 2 ENG
5/22/2023
8:48 PM
`decoder _inp
74 看到非填充词
> OUTUNE
75
WSL: Ubuntu > TIMELINE 2 A0 each word can only look at the previous words and each word can only look at c
5/22/2023
8:49 PM
encoder _input ":encoder _input,#( Seq _ Len )
"decoder _inp "encoder _mask "
ze(θ). unsqueeze(θ). int(),#(1, 1, Seq_ Len)
'decoder _mas 看到非填充词
> OUT UNE
WSL: Ubuntu
> TIMELINE
2 A0
non-padding Ln74. Col29 Spaces4 UF-8 LF(1 Python 3. 10. 6(trans
ENG
5/22/2023
8:49 PM
`decoder _inp encoder _mask"
eze(0). int(),#(1, 1, Seq_ Len)
74 看到非填充词
> OUTUNE
75
> WSL : Ubuntu > TIMELINE words so we don't want again we don't want the padding token si to participate. in. the 5/22/2023
8:49 PM
72
> OUTUNE 所以我们再次不希望填充标记参与自注意力机制, 我们只希望
> WSLUbuntu > TIMELINE △ words so we don't'want again we don't want the padding token si to participate. in. the.
ENG
5/22/2023
8:49 PM
72
encoder_input,#( Seq_ L en)
> OUTUNE 所以我们再次不希望填充标记参与自注意力机制, 我们只希望
> WSL: Ubuntu > TIMELINE 2 A0 self-attention we only want real words to participate in this and we also don't. want ENG
5/22/2023
B:49 PM
72
> OUTUNE 真正的词参与其中, 并且我们也不希望每个词看到它后面的
> WSL: Ubuntu > TIMELINE 2 A0 self-attention we only want real words to participate in this and we also don't. want 5/22/2023
B:49 PM
72 encoder _input ":encoder _input,#( Seq _ Len )
> OUT U NE 真正的词参与其中, 并且我们也不希望每个词看到它后面的
> WSL : Ubuntu > TIMELINE each word to watch at
Ln 74. Col29
5/22/2023
8:49 PM
71
ncoder_input ":
encoder _input,#( Seq_ Len)
72
> OUTUNE 词, 而只希望看到它前面的词. 因此, 我将在这里使用
> WSL: Ubuntu > TIMELINE 5/22/2023
8:49 PM
72
der_input,
> OUTUNE 词, 而只希望看到它前面的词. 因此, 我将在这里使用
WSL: Ubuntu > TIME UNE
2 A0
words that come after it but only that words come come before it sol willusear
5/22/2023
B:49 PM
72 一种称为因果掩码的方法, 稍后将构建它. 我们也将构建它.
> OUTUNE
WSL: Ubuntu
> TIMEUNE
words that come after it but only that words come come before it sol will use a
ENG
5/22/2023
8:49 PM
72 一种称为因果掩码的方法, 稍后将构建它. 我们也将构建它.
> OUTUNE
WSL: Ubuntu
> TIMEUNE
method he recalled causal mask that will build it later we will build it also sonow.
5/22/2023
B:49 PM
72
> OUTUNE 所以现在我只是调用它来展示它的使用方式, 然后我们也将
WSL: Ubuntu > TIMELINE 2 A0 method he recalled causal mask that will build it later we will build it also sonow
B:49 PM
5/22/2023
72
encoder_input,# ( Seq_ L en)
> OUTUNE 所以现在我只是调用它来展示它的使用方式, 然后我们也将
WSL: Ubuntu > TIMELINE 2 A0
iustcall Pytho
8:49 PM
5/22/2023
72
> OUTUNE 所以现在我只是调用它来展示它的使用方式, 然后我们也将
WSL: Ubuntu > TIMELINE 2 A0
it to show you how it's used and then we will proceed to build. it also in this case 8:49 PM
5/22/2023
"decoder _in p
'encoder_mask":
t(),#(1, 1, Seq_ Len)
74
decoder _mask 继续构建它
> OUTUNE
75
WSL: Ubuntu > TIMELINE it to show you how it's used and then we will proceed to build. it also in this case 8:49 PM
5/22/2023
72
> OUTUNE 在这种情况下, 我们不希望填充标记, 我们添加必要的
WSL: Ubuntu > TIMELINE it to show you how it's used and then we will proceed to build. it also in this case 人
5/22/2023
B:49 PM
72
delattr
> OUTUNE 在这种情况下, 我们不希望填充标记, 我们添加必要的
WSL: Ubuntu > TIMELINE 2 A0 we don't want the padding tokens Ln 74. Col52 Spa
Pyt
B:49 PM
5/22/2023
72
coder_input,#( Seq_ L en)
> OUTUNE 在这种情况下, 我们不希望填充标记, 我们添加必要的
WSL: Ubuntu > TIMELINE Pytl
B:49 PM 人
5/22/2023
72 encoder _input ":encoder _input,#( Seq _ Len)
#( Seq_ Len 维度, 并且我们还进行布尔与因果掩码的操作, 这是一个
> OUT UNE
> TIMEUNE
5/22/2023
B:49 PM
72
> OUTUNE 我们将立即构建的方法, 这个因果掩码需要构建一个大小为
WSL: Ubuntu > TIME UNE
5/22/2023
8:49 PM
er _input,
que eze (θ). int(),#(1, 1, Seq_ Len)
74
> OUTUNE
75 工
> WSL: Ubuntu > TIME UNE mask which is a method that we will build right now and this. causal mask need to 5/22/2023
8:50 PM
der _input,
queeze(0). int(),=(1, 1, Seq_ Len)
> OUTUNE
75 序列长度乘以序列长度的矩阵
eeze(e). int()&causal_mask( 工
WSL: Ubuntu > TIME UNE build a matrix ln. 74. Gol109
4. 10. 6 Erans 人
P
ENG
5/22/2023
B:50 PM
er _input,
#( Seq_ Len
ueeze(θ). int(),#(1, 1, Seq_ Len)
74
> OUT UNE
75 工
> WSL: Ubuntu
> TIMELINE
A1
of size sequence length to sequence length what is sequence length is basically the 人
5/22/2023
8:50 PM
72
oder_input,#( Seq_ Len
Zero Division Erron code Error 序列长度基本上是我们解码器输入的大小, 这个一让我
# Bilingual Data set > OUT U NE
WSL : Ubuntu > TIME LNE size of our decoder input 5/22/2023
8:50 PM
72 encoder _input ":encoder _input,#( Seq _ Len)
der_input,#( Seq _ Len 为你写一个注释. 所以这是一个, 两个序列长度组合在
input. size (e )
> OUT U NE
WSL : Ubuntu > TIME UNE 0 A1
( Pytho
3. 10. 6
B:50 PM 人
5/22/2023
72
der_input,#( Seq_ Len 为你写一个注释. 所以这是一个, 两个序列长度组合在
input. size (e )
> OUT UNE
> WSL: Ubuntu
> TIMEUNE
0 A1
and this let me write a comment for you so this is one two sequence length combined.
5/22/2023
8:50 PM
72
`decoder_input"
#( Seq_ Len
> OUTUNE 一起. 所以这个布尔与一个序列长度, 序列和, 这个可以
size (θ))=(1, Seq_ Len)
> WSL: Ubuntu
> TIMEUNE
and this let me write a comment for you so this is one two sequence length combined.
5/22/2023
B:50 PM
72 decoder _input,#( Seq_ Len
> OUTUNE 广播. 让我们去定义这个方法:因果掩码. 那么什么是因果
(1, Seq_ Len, Seq_ Ler)
> TIMELINE 人
5/22/2023
8:50 PM
72
decoder_inpu
> OUTUNE 广播. 让我们去定义这个方法:因果掩码. 那么什么是因果
(1, Seq_ Len)&(1,
WSL: Ubuntu
> TIMEUNE
0 A1
let's go define this method causal mask so what is causal. maskr
4. 10. 6(tran
8:50 PM 人
ENG
"decoder _input ":
(encoder _input 1= self. pad_token). unsqueeze (0). unsqueeze(0). int(),#(1, 1, Seq_ Len)
decoder _input,#( Seq _ Len )
"encoder _mask ":
"decoder _mask ":
(decoder _input 1= self. pad_token). unsqueeze (e). unsqueeze(0). int()&causal_mask(decoder _input. size (e)) #(1, Seq_ Len
> OUTUNE
76
I
> TIMELINE def d
WSL: Ubuntu
1 A1
Ln77, Col b Spaces : 4 UTF-8 LF( Python3. 10. 6 (transformer :conda)
ENG
5/22/2023
8:50 PM
73
[= self. pad_token). u
er_input. size (e))#(1, Seq_ Len
> OUTUNE 因果掩码基本上意味着我们想要 让我们回到幻灯片
> WSL: Ubuntu > TIMELINE 1 A1
causal maskbasically means that we want let'sgo back to the slides actually as you.
5/22/2023
B:50 PM
实际上, 如你在幻灯片中记得的, 我们希望解码器中的每个
4. 229
causal mask basically means that we want let's go back to the slides actually as you.
词只看到它前面的词. 所以我们想要的是让这个矩阵对角线
come before
以上的所有值. 这个矩阵表示自注意力机制中查询与键的
it so what we want is to make all these values above this diagonal this matrix
乘积.
5. 195
6. 114
7. 203
8. 103
9. 157
10. 229
represents the multiplication of the queries by the keys in the self-attention.
我们想要隐藏所有这些值, 这样你就不能看到"猫是一只
11. 15
12. 229
What we want is to hide all these values so your can not watch the word cat is a
可爱的猫". 它只能看到它自己.
13. 114
14. 203
15. 103
16. 157
17. 229
What we want is to hide all these values so your can not watch the word cat is a
但这里的这个词, 例如"可爱的", 可以看它前面的
18. 157
19. 229
lovely cat It can only watch itself But this word here for example this word lovely
但这里的这个词, 例如"可爱的", 可以看它前面的
20. 157
21. 229
can watch
但这里的这个词, 例如"可爱的", 可以看它前面的
22. 157
23. 229
everything that comes before it so from your up to lovely itself But not the word cat
"猫"这个词.
AT 0. 195
24. 114
25. 203
26. 103
27. 157
28. 229
everything that comes before it so from your up to lovely itself But not the word cat
"猫"这个词.
AT 0. 195
29. 114
30. 203
31. 103
32. 157
33. 229
that comes after it so what we do is we want all these values here to be musket out
所以我们所做的是我们希望这里的所有这些值都被屏蔽掉, 这
that comes after it so what we do is we want all these values here to be musket out
所以我们所做的是我们希望这里的所有这些值都被屏蔽掉, 这
and
也意味着我们希望这个对角线以上的所有值都被屏蔽掉, 在
也意味着我们希望这个对角线以上的所有值都被屏蔽掉, 在
which also means that we want all the values above this diagonal to be masked out and
Py Torch中有一个非常实用的方法来做到这一点.
34. 103
35. 157
which also means that we want all the values above this diagonal to be masked out and
所以让我们来做, 让我们去构建这个方法. 所以掩码基本上
there is a very practical method in Py Torch to do it so let's do it let's go build
"encoder _mask ":
"decoder _mask ":(decoder _input 1= self. pad_token). un squeeze (0). unsque eze(e). int ()& causal _mask (decoder _inpu
74
Seq_ Ler
> OUTUINE 所以让我们来做让我们去构建这个方法. 所以掩码基本上
WSL: Ubuntu > TIMELINE 10 there is a very practical method in Py Torch to do it so let's do it let's go build 5/22/2023
B:52 PM
> OUT UINE 所以让我们来做, 让我们去构建这个方法. 所以掩码基本上
WSL: Ubuntu
> TIMELINE
1 A0
this 3. 10. 6(ta
B:52 PM
5/22/2023
> OUTUINE 所以让我们来做, 让我们去构建这个方法. 所以掩码基本上
WSL: Ubuntu10
> TIMELINE method so the mask is basically torch. tri u which means give me every value that. is.
5/22/2023
B:52 PM
true _divide f T racing State > OUT UINE 是torch. triu, 这意味着给我所有我告诉你的对角线以上的值.
WSL: Ubuntu10
> TIMELINE method so the mask is basically torch. tri u which means give me every value that. is.
5/22/2023
B:52 PM
> OUT UNE 是torch. tri u, 这意味着给我所有我告诉你的对角线以上的值.
WSL : Ubuntu > TIMELINE above the diagonal that lam telling you so we want a matrix which matrix? ao6ta 人
ENG
5/22/2023
B:52 PM
Type Error Unicode Translate Error > OUT U NE
WSL Ubuntu > TIME UNE Ln78. Col24
1. 10. 6(trans
4 UTF-8 LF{ Python
ENG
8:52 PM
5/22/2023
> OUTUNE 由所有1组成的矩阵, 这个方法将返回对角线以上的所有
WSL: Ubuntu > TIMELINE matrix made of all ones Ln 78, Col 28
B:52 PM
P
5/22/2023
> OUTUNE 值, 其他所有值将变为零. 所以我们想要对角线1
WSL: Ubuntu
> TIMEUNE
and this method will return every value above the diagonal and. everything else will 5/22/2023
8:52 PM
> OUT U NE 值, 其他所有值将变为零. 所以我们想要对角线
> WSL : Ubuntu > TIME UNE become zero so we want diagonal one type we want it to be integer and what we do is 人
5/22/2023
8:52 PM
> OUT U NE 类型, 我们希望它是整数, 我们所做的是返回掩码等于零.
> WSL : Ubuntu > TIME LNE become zero so we want diagonal one type we want it to be integer and what we do is
8:52 PM 人
> OUT U NE 类型, 我们希望它是整数, 我们所做的是返回掩码等于零.
> WSL : Ubuntu > TIMELINE return mask is equal to zero Ln 79, Col 5
8:52 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 79. Col21 Spaces4 UTF-8 LF { Python 3. 10. 6(transfo
Z
5/22/2023
8:52 PM
> OUT U NE 所以这将返回对角线以上的所有值, 而对角线以下的所有值将
> WSL : Ubuntu > TIMELINE so this will return all the values above the diagonal and everything below the ostuar 口
ENG
5/22/2023
8:52 PM
> OUT U NE 变为零. 但实际上我们想要相反的效果.
> WSL: Ubuntu
> TIMELNE
5/22/2023
8:52 PM
> OUT UNE 所以我们说:好吧, 所有为零的值将通过这个表达式变为真
> TIMELNE
8:53 PM
5/22/2023
> OUT U NE 所以我们说:好吧, 所有为零的值将通过这个表达式变为真
WSL : Ubuntu > TIME LNE that is zero In 79, Col 19 (9 sel
8:53 PM
5/22/2023
> OUT U NE 所以我们说:好吧, 所有为零的值将通过这个表达式变为真
WSL : Ubuntu > TIME LNE should will be come true with this expression and everything that is not zero will 8:53 PM 人
ENG
5/22/2023
> OUT U NE 这个掩码. 所以这个掩码将是一个序列长度乘以序列长度的
WSL : Ubuntu > TIMELINE become false so we apply it here to build this mask so this mask. will be one bysc 人
5/22/2023
8:53 PM
> OUT U NE 这个掩码. 所以这个掩码将是一个序列长度乘以序列长度的
WSL : Ubuntu > TIMELINE sequence length by sequence length which is exactly what we want pthon
8:53 PM 口
ENG
5/22/2023
> OUT U NE 矩阵, 这正是我们想要的
> WSL : Ubuntu > TIME LNE sequence length by sequence length which is exactly what. we want yth on 3. ios trast
ENG
5/22/2023
8:53 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 4 UIF-8 LF{ Python 3. 10. 6(tran
ENG
8:53 PM
Z
5/22/2023
> OUT U NE 好的, 让我们也添加标签. 标签也在上面. 我忘了逗号:
WSL : Ubuntu > TIME UNE okay let's add also the label the label is also up l forgot the comma sequence length 5/22/2023
8:53 PM
> OUT U NE 只是为了可视化, 我们可以发送源文本, 然后是自标文本
> WSL : Ubuntu > TIME UNE and then we have the source text just for visualization we can send it source. text 5/22/2023
8:53 PM
> OUT U NE 只是为了可视化, 我们可以发送源文本, 然后是自标文本
> WSL : Ubuntu > TIMELINE and then the target text P
5/22/2023
8:53 PM
86
> OUTUNE 这是我们的数据集,
> WSL: Ubuntu > TIME LNE
Ln 77. Col 27
4 UTF-8 LFPython 3. 10. 6(trar
ENG
8:53 PM 人
5/22/2023
86
> OUTUNE
WSL: Ubuntu > TIMELINE Ln78. Col9 Spaces:4 UTF-8 LF( Python 5/22/2023
8:53 PM
86
> OUTUNE 现在让我们回到我们的训练方法, 继续编写训练循环
WSL: Ubuntu > TIMELINE Now let'sgo back to our training method to continue writing the. training loop. cm
5/22/2023
8:53 PM
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln40. Col5 Spaces 4 UTF-8 LF{ Python3. 10. 6(transformer :conda)
Z
ENG
5/22/2023
8:53 PM
所以现在我们有了数据集, 我们可以创建它.
> OUT U NE
WSL : Ubuntu > TIME UNE So now that we have the dataset, we can create it.
ENG
5/22/2023
8:53 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln40. Col5 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :con
ENG
8:54 PM
5/22/2023
> OUT U NE 我们可以创建两个数据集, 一个用于训练, 一个用于验证
> WSL : Ubuntu > TIME LNE We can create two datasets, one for training, one for validation, and. then we send it 5/22/2023
B:54 PM
然后将它们发送到数据加载器, 最后到我们的训练循环.
> OUT U NE
> WSL : Ubuntu > TIMELINE We can create two datasets, one for training, one for validation, and. then we send it B:54 PM 人
5/22/2023
然后将它们发送到数据加载器, 最后到我们的训练循环.
> OUT U NE
> WSL : Ubuntu > TIMELINE to a data loader and finally to our training loop. spacs urs u
ption
4. 10. 6 (tra
B:54 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln41, Col5 Spaces:4 UTF-8 LF Python 3. 10. 6(transfo
ENG
5/22/2023
8:54 PM
哦, 我们忘记导入数据集了,
> OUT U NE
> WSL : Ubuntu > TIME LNE Oh, we forgot to import the dataset.
Ln41, Col 16 Spaces4 UTF-8 LFPython 3. 10. 6(transfo 通
ENG
5/22/2023
8:54 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln41, Col5 Spaces : 4 UTF-8 LF ( Python3. 10. 6(transformer :conda)
8:54 PM
5/22/2023
> OUT U NE 所以让我们在这里导入它.
> WSL : Ubuntu > TIME LNE So let'simport it here.
Ln41, Col5 Spaces4 UTF-8 LF Python 3. 10. 6(transformer :cond 通
2
5/22/2023
8:54 PM
> OUT U NE
> WSLUbuntu
> TIMELNE
Ln3. Col63 Spaces 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
8:54 PM
5/22/2023
42
43 我们还导入了因果掩码, 我们稍后会用到它.
> OUTUNE
> WSL: Ubuntu101fle to analyze > TIME LNE We also import the causal mask, which we will need later. urs w 4 ython 3. ios transtor
ENG
8:54 PM
5/22/2023
42
43
> OUTUNE
> WSL: Ubuntu > TIME UNE
Ln43, Col5 Spaces 4 UTF-8 LF ( Python 3. 10. 6(transformer :conda)
ENG
8:54 PM
5/22/2023
src _lang : Any, tgt _lang : Any, seq _len: Any) -> None
43
train_ds= Bilingu Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config [lang _src'], config [lang _tgt']
> OUT U NE 我们的序列长度是多少?
> WSL Ubuntu > TIME UNE And what is our sequence length?
Ln43, Col115 Spaces4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
ENG
8:55 PM
5/22/2023
src _lang : Any, tgt _lang : Any, seq _len: Any)-> None
43
train_ds = Bilingu LDataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config ['lang _src'],
config ['l ang_tgt'], config]
[0]config
> OUTUNE 它也在配置中.
> WSL: Ubuntu > TIME UNE It's also in the configuration.
Ln43, Col 123 Spaces 4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
ENG
5/22/2023
8:55 PM
44
train_ds = Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config ['lang _src'], config ['lang _tgt'], config['seq_len'])
> OUTUNE
WSL: Ubuntu > TIME LNE
Ln44, Col5 Spaces:4 UTF-8 LF{ Python 310. 6(transfo
ENG
5/22/2023
8:55 PM
42
43
> OUTUNE 但唯一的区别是我们现在使用这个, 其余的都一样,
> WSL: Ubuntu > TIME UNE but the only difference is that we use this one now, and. the rest is same.
4. 10. 6(trans
ENG
5/22/2023
8:55 PM
44
train_ds= Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config ['lang _src'], config ['lang_tgt'], config['seq_len])
val_ds = Bilingual Dataset (val_ds_aw, tokenizer _src, tokenizer _tgt, config ['lang _src'], config [‘lang _tgt'], config ['seq _len']
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln44, Col 131 Space :4 UTF-8 LF( Python 3. 10. 6 (trans fom
ENG
5/22/2023
8:55 PM
42
43
44
train_ds = Bilingual Dataset (train ds _raw tokenizer _src config [‘lang _tgt'], config ['seq _len'])
config [‘lang _tgt'], config['seq_len'])
> OUTUNE 我们也只是为了选择最大序列长度.
> WSL: Ubuntu > TIME UNE We also, just for choosing the max sequence length, we also want. to watch what is. the.
5/22/2023
8:55 PM
42
43
train_ds = Bilingual Dataset (train _ds _raw config ['seq _len'])
> OUT U NE 我们还希望观察我们在这里创建的两个分割中, 源语言和
> WSLUbuntu
> TIMEUNE
5/22/2023
8:55 PM
43
train_ds = Bilingual Dataset (train _ds _raw config ['seq _len'])
> OUT U NE 我们还希望观察我们在这里创建的两个分割中, 源语言和
WSL : Ubuntu > TIME UNE maximum length of each sentence in the source and the target for each of the two 5/22/2023
8:55 PM
train _ds = Bilingual Dataset (train _ds _raw tokenizer _src,
tokenizer _tgt,
len'])
> OUT UI NE 目标语言中每句话的最大长度, 这样如果我们选择一个非常小
WSL : Ubuntu > TIME LNE maximum length of each sentence in the source and the target for each of the two 人
5/22/2023
8:55 PM
43
train_ds = Bilingual Dataset (train _ds _raw tokenizer _src,
> OUT UI NE 目标语言中每句话的最大长度, 这样如果我们选择一个非常小
WSL : Ubuntu > TIME LNE splits that Ln46. Col5
Z
P
ENG
5/22/2023
8:55 PM
train _ds = Bilingual Data tokenizer _src,
len'])
> OUT U NE 目标语言中每句话的最大长度, 这样如果我们选择一个非常小
> WSL : Ubuntu > TIMELINE we created here, so that if we choose a very small sequence length, we will know.
ENG
/22/2023
8:55 PM
42
44
43
train_ds = Bilingual Dataset (train _ds_raw
val
src'], config ['lang_tgt'], config['seq _len'])
], config [‘lang _tgt ’], config ['seq_len'])
> OUTUNE
46 的序列长度, 我们就会知道
> WSL: Ubuntu > TIMELINE we created here, so that if we choose a very small sequence length, we will know. 人
2 ENG
5/22/2023
8:55 PM
train _ds = Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config [‘lang _src'], config ['lang_tgt'], config['seq_len'])
val_ds = Bilingual Dataset (val _ds _raw, tokenizer _src, tokenizer _tgt, config [‘lang _src'], config [‘lang _tgt'], config['seq_len’])
> OUTUNE
46
WSL: Ubuntu00
> TIME LNE
Ln46, Col5 Spaces:4. UTF-8 LF( Python 3. 10. 6 (transh
P
ENG
5/22/2023
8:55 PM
45
46
nax_len_src0 基本上我们做的是, 我从源语言和目标语言加载每句话
> OUTUNE
WSL: Ubuntu01
> TIMELNE
Basically we do, l load each sentence from each language, from. the source and the 5/22/2023
8:55 PM
45
46
max_len_src=0
ax_len_tgt=θ 使用tokenizer 将其转换为 ID, 并检查长度
> OUT U NE
WSL : Ubuntu > TIME UNE target language, I convert into IDs using the tokenizer and I check. the length. os tans
8:56 PM
ENG
5/22/2023
max_len_src =θ
max_len_tgt =θ
> OUT UNE
for ite in ds_raw:
> TIMEUNE
50
src_ids|
> WSL: Ubuntu
Ln 50, Col17 Spaces: 4 UTF-8 LF( Python3. 10. 6 (trans for er:oonda)
5/22/2023
8:56 PM
45
46
max_l en_src=0 如果长度是, 比如说, 180, 我们可以选择200作为序列
> OUTUNE
WSL: Ubuntu > TIME UNE If the length is, let's say, 180, we can choose 200 assequence. length because it B:56 PM 中
5/22/2023
45
46
nax_l en_src=0
len_tgt =
> OUTUNE 长度, 因为它将覆盖我们在这个数据集中所有可能的句子.
> WSL: Ubuntu > TIME UNE If the length is, let's say, 180, we can choose 200 assequence. length because it 5/22/2023
8:56 PM
nax_l en_src=0
len_tgt =
> OUTUNE 长度, 因为它将覆盖我们在这个数据集中所有可能的句子,
> WSL: Ubuntu > TIME UNE will cover all the possible sentences that we have in. this data set. Mion
4. 10. 6 Ct
8:56 PM 人
max_len_src =θ
max_len_tgt =θ
for it en in ds_raw:
> OUTUNE
50
src_ids
> WSL: Ubuntu > TIME LNE
0 A1
n 50, Col17 Spaces :4 UTF-8 LF(↓ Python3. 10. 6(transformer :conda)
8:56 PM
5/22/2023
45
46
nax_l en_src=0 如果是, 比如说,~500, 我们可以使用510或类似的值
> OUTUNE
WSL: Ubuntu > TIME LNE
0 A1
If it's, let's say, 5o0, we can use 510 or something like this because we also need 8:56 PM
5/22/2023
45
46
nax_len_src=0 因为我们还需要在这些句子中添加句子的开始和结束标记.
> OUTUNE
WSL: Ubuntu > TIME UNE to add the start of sentence and the end of sentence tokens to these i sentences.
8:56 PM
5/22/2023
max_len_src =0
max_len_tgt=θ
> OUTUNE
for ite in ds_raw:
> TIMELNE
50
srcids|
WSL: Ubuntu
Ln 50. Col 17 Spaces : 4 UTF-8 LF( Python3. 10. 6 (transformer :conda)
5/22/2023
8:56 PM
max_len_src =θ
max_len_tgt =0
> OUTUNE 这是源 ID, 然后让我们也创建目标 ID, 这是目标语言
WSLUbuntu > TIME UNE This is the source IDs, then let's create also the target IDs and this is the 8:56 PM
5/22/2023
max_len_src =0
nax_len_tgt =0
> OUTUNE 这是源 ID, 然后让我们也创建目标 ID, 这是目标语言
> WSL: Ubuntu > TIME UNE language of target and then we just say the source maximum length is the maximum of.
5/22/2023
B:56 PM
47
max_len_tgt=0
48
for item 然后我们只是说源的最大长度是当前句子的最大长度.
> OUT U NE
> WSL Ubuntu > TIMELINE 5/22/2023
8:57 PM
47
max_len_tgt=0
With a single iterable argument ret un its biggest item. he 然后我们只是说源的最大长度是当前句子的最大长度.
or item > OUT U NE
> WSL : Ubuntu > TIMELINE the Ln 52, Col 27
( Pytho
4. 10. 6(tra
8:57 PM
5/22/2023
for item in ds _raw :
src _ids = tokenizer _src. encode (item ['translation'][config [lang _src']). ids tgt _ids = tokenizer _src. encode (item ['translation'][config [*lang _tgt']]). ids
> OUTUNE
max_len_src =max(max_len_src, len(src_ids ))
> TIME UNE 53
max_len _src=max(max_len_src, 1en(src_ids)
WSLUbuntu
Ln53, Col 53 Spaces 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/22/2023
8:57 PM
for item in ds_raw: 目标是目标, 目标1 D是然后我们打印这两个值.
> OUTUNE
WSL: Ubuntu > TIME UNE The target is the target and the target ID is...
Then we print these two values. o transk
8:57 PM 口
ENG
5/22/2023
tgt _
fl leaf le like object (stream defaults to the current sys. stdout
g_tgt']]). ids
max.
> OUTUNE
55
print (
> WSL: Ubuntu
> TIMELNE
Ln 55, Col13 Spaces:4 UTF-8 LF( Python3. 10. 6 (transformer :oonda)
8:57 PM
5/22/2023
tgt _ids = tokenizer _src. encode (item ['translation'][config [‘lang _tgt']). ids max_len_src =max(max_len_src, len(src_ids)
max_len_tgt=max(max_len_tgt, len(tgt_ids))
> OUT UNE print(f'Max length of source sentence :{max_len_src )')
> TIME LNE
56
> WSL: Ubuntu
Ln 56, Col 29 Spaces : 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/22/2023
8:57 PM
max _l en_src=
nax_len_tgt
len(src_ids))
> OUTUNE 就是这样. 现在我们可以继续创建数据加载器了. 我们根据
> WSL: Ubuntu > TIME LNE and that's it now we can proceed to create the data loaders we define the batch size ENG
5/22/2023
8:57 PM
Any | None = None,*, prefetch factor :in t =2,
> OUTUNE 就是这样. 现在我们可以继续创建数据加载器了. 我们根据
WSL: Ubuntu
> TIMEUNE
according to our configuration which we still didn't define but you can already guess 5/22/2023
8:57 PM
max_len_tgt =max(max_len_tgt, len(t multip Any = None, generator : 配置定义批量大小, 我们还没有定义, 但你可以猜到它的
or int(f'Max > OUT U NE
WSL : Ubuntu > TIME UNE according to our configuration which we still didn't define but you can already guess 5/22/2023
8:57 PM
max_len_tgt =max(max_len_tgt, len(tgt_ids)
multiprocessing _context : Any | Nc None, generator : 配置定义批量大小, 我们还没有定义,
rint(f'Max
> OUTUNE 但你可以猜到它的
WSL: Ubuntu > TIME UNE what are its values we want it to be shuffle din6
P
ENG
5/22/2023
B:58 PM
print(f'Max length of source sentence :{max_len_src )')
print(f'Max length of target sentence :{max_len_tgt}')
> OUTUNE
58
train _data loader = Data Loader (train _ds, batch _size =config [‘batch _size ′], shuffle = True )
> TIME UNE
59
WSL: Ubuntu
Ln59. Col5 Spaces 4 UTF-8 LFPython3. 10. 6(trans fo
ENG
5/22/2023
8:58 PM
55
print(f'Max length of source sentence :{max_len_src}')
orint(f'Ma> 好的, 对于验证, 我将使用批量大小为1, 因为我希望
> OUTUNE
> WSL: Ubuntu > TIME LNE Okay, for the validation l will use a batch size of 1 because l want to process each 8:58 PM
5/22/2023
print(f'Max length of source sentence :{
max _len_sr
print(f'Max length o persistent _workers :bool = False )-> None > OUT UNE 58
tra in_dataloader 逐句处理每句话
ombines a dataset and a sampler, and provides an > TIME UNE
59
val_dataloader
> WSLUbuntu Okay, for the validation l will use a batch size of 1 because l want to process each 5/22/2023
8:58 PM
multiprocessing _context : Any | None = None, generator :
print(f'Max length o persistent _workers : bool = False )-> None > OUT UNE train _data loader 逐句处理每句话
omb ines a dataset and a sampler and provides an > TIME UNE
59
val_dataloader
> WSLUbuntu sentence one by one.
Ln59, Col55 Space:4 UTF8 LF( Python 3. 10. 6 (trans
ENG
5/22/2023
8:58 PM
print(f Max length of source sentence :{max_len_src Any | None = None,°, prefetch _factor :in t =2,
print(f Max length of target sentence :{max_len_tgt persistent _workers : bool= False)-> None
> OUTUNE
59
val_data loader = Data Loader (val_ds, batch_size=1,) 工
> WSLUbuntu > TIME UNE Ln 59, Col55 Spaces:4 UTF·8 LF{ Python 3. 10. 6 (trans
ENG
8:58 PM
5/22/2023
pr raise
srange shuffle = Tr 这个方法返回训练的数据加载器、验证的数据加载器,
> OU TUNE
WSL : Ubuntu > TIMELINE And this method returns the data loader of the training, the data loader of the 5/22/2023
8:58 PM
r (train _ds, batch izer _src 源语言的tokenizer 和目标语言的tokenizer.
> OUT U NE
WSL : Ubuntu > TIME UNE validation, the tokenizer of the source language and the tokenizer of the target 8:58 PM 办 ENG
5/22/2023
57
r(train_ds, batch_s
ze'], shuffle = True )
> OUT UNE 源语言的tokenizer 和目标语言的tokenizer.
WSL : Ubuntu > TIME LNE language.
Ln 61, Col74 Spac
4. 10. 6 (trans
UTF-8 LFPython
8:58 PM
2 ENG
5/22/2023
train _data loader = Data Loader (train _ds, batch _size =config ['batch _size'], shuffle = True )
val _data loader = Data Loader (val _ds, batch _size =1, shuffle= True)
> OUTUNE
61
return train _data loader, val _data loader, tokenizer _src, tokenizer _tg
> WSL: Ubuntu
> TIMEUNE
Ln 61, Col 74 Spaces : 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
ENG
8:58 PM
5/22/2023
train _data load e
er(train _ds, batch _siz batch _size'], shuffle = True )
> OUT U NE
> WSL : Ubuntu > TIME LNE
ENG
5/22/2023
8:58 PM
train _data loader = Data Loader (train _ds, batch _size =config ['batch _size'], shuffle = True )
val _data loader = Data Loader (val _ds, batch _size =1, shuffle= True)
> OUTUNE
61
return train _data loader, val _data loader, tokenizer _src, tokenizer _tgt 工
WSL : Ubuntu > TIME LNE
Ln61, Col74 Spaces:4 UTF-8 LF {↓ Python 3. 10. 6 (transfo
ENG
5/22/2023
8:58 PM
58 train _data loader = Data loader (train _ds, batch _size =config [‘batch _size'], shuffle = True)
59
val_data loader = Data Loader (val_ds, batch_size=1, shuffle = True ) 所以让我们定义一个新方法, 叫做get Model, 它将根据我们的
> OUT U NE
WSL : Ubuntu > TIME UNE So let's define a new method called get Model, which will, according to. our a. o mtam
B:58 PM
ENG
5/22/2023
58 train _data loader = Data loader (train _ds, batch _size =config ['batch _size'], shuffle = True)
59
al_data loader = Data Loader (val_ds, batch_size=1, shuffle = True ) 配置、词汇表大小构建模型, 即 Transformer 模型
> OUT UNE
WSL: Ubuntu
> TIMELNE
1 A0
configuration, our vocabulary size, build the model, the transformer model. octrar
5/22/2023
8:58 PM
59
True)
60
kenizer_tgt
> OUTUNE 所以模型是, 我们还没有导入模型, 所以让我们导入它.
> WSL: Ubuntu
> TIMELNE
10
So the model is, we didn't import the model, so let's import u it. u
pyhon
4. 10. 6 (tra
8:59 PM
5/22/2023
42
43
val_ds = Bilingual Dataset (val _ds _raw, tokenizer _src, tokenizer _tgt, config [‘lang _src'], config [‘lang _tgt'], config ['seq _len'])
train _ds = Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config ['lang _src'], config ['lang _tgt'], config ['seq_len'])
> OUTUNE
max_1en_src=θ
> TIME LNE
48
max_len_tgt=θ
> WSL: Ubuntu 1 A0
Ln 64, Col5 Spaces:4 UTF-8 LFPython 3. 10. 6 (transfo
ENG
5/22/2023
8:59 PM
49
ax_len_src
pax_len_cgt=0
> OUTUNE 构建
> TIME UNE gt ids =tokenize scc_ids=
> WSL: Ubuntu
52
Build transformer.
Ln6, Col36 Spaces4 UTF-8 LFPython 3. 10. 6 (transfo 人
P 办 ENG
5/22/2023
8:59 PM
> OUT U NE
> WSL : Ubuntu > TIME UNE
Ln65, Col5 Spaces4 UTF-8 LF{ Python 2 ENG
5/22/2023
8:59 PM
源词汇表大小.
> OUT U NE
WSL : Ubuntu > TIMELINE The source vocabulary size.
Ln65, Col31 Spaces4 UTF-8 LF (↓ Python 3. 10. 6(trans
ENG
5/22/2023
8:59 PM
> OUT U NE 和目标词汇表大小.
WSL : Ubuntu > TIME LNE And the target vocabulary size.
Ln65, Col46 Spaces:4 UTF-8 LF (↓ Python 3. 10. 6(trans
ENG
5/22/2023
8:59 PM
Value Error []src _vocab _size =
[)tgt _vocab _size =
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln65, Col 48
Spaces4 UTF-8 LFPython 3. 10. 6(transfo
ENG
5/22/2023
8:59 PM
> OUT U NE 然后我们有序列长度.
WSL : Ubuntu > TIME UNE
ENG
5/22/2023
8:59 PM
> OUT U NE
WSL : Ubuntu > TIME UNE Ln65, Col65 Spaces4 UTF-8 LF {↓ Python 3. 10. 6(trans 办 ENG
8:59 PM
5/22/2023
> OUT U NE 我们有源语言的序列长度和目标语言的序列长度
> WSL : Ubuntu > TIME UNE And we have the sequence length of the source language and the sequence length of the 8:59 PM
5/22/2023
我们有源语言的序列长度和目标语言的序列长度.
> OUT U NE
WSL : Ubuntu > TIME UNE target language.
Ln 65, Col. 80
( Pytho
4. 10. 6 (t
B:59 PM
5/22/2023
> OUT U NE 我们将使用相同的.
WSL : Ubuntu > TIME UNE We will use the same.
Ln 65, Col91 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
8:59 PM
> OUT UNE 对于两者, 然后我们有dmodule, 目 即嵌入的大小, 我们可以保持
> WSL : Ubuntu > TIME UNE for both and then we have the dmodule which is the size of the embedding we can keep..
5/22/2023
8:59 PM
> OUT UNE 其余的默认值, 就像在论文中一样. 如果模型对于你的 GPU
> WSL: Ubuntu
> TIMELNE
for both and then we have the dmodule which is the size of the. embedding we can keep.
5/22/2023
8:59 PM
> OUT UNE 其余的默认值, 就像在论文中一样. 如果模型对于你的 GPU
> WSL: Ubuntu
> TIMELNE
all the rest the default as in the paper ns. colms
5/22/2023
8:59 PM
> OUT UNE 其余的默认值, 就像在论文中一 如果模型对于你的 GPU WSL : Ubuntu > TIMELINE ( Pytho
B:59 PM
5/22/2023
> OUT U NE 来说太大, 无法训练, 你可以尝试减少头数或层数
> WSL : Ubuntu > TIME UNE If the model is too big for your GPu to be trained on, you can. try to reduce the 5/22/2023
B:59 PM
> OUT U NE 来说太大, 无法训练, 你可以尝试减少头数或层数.
> WSL : Ubuntu > TIMELINE number of heads or the number of layers.. cal spoc
Pyth
9:00 PM 口
5/22/2023
> OUT U NE
> WSL Ubuntu > TIME UNE
Ln67, Col1 Spaces4 UTF-8 LF { Python 3. 10. 6(trans 办 ENG
5/22/2023
9:00 PM
当然, 这会影响模型的性能
> OUTUNE
> WSL: Ubuntu00
> TIMELINE Of course, it will impact the performance of the model.
UTF-8 LF Python3. 10. 6(trans fo
ENG
9:00 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln67, Col1 Spaces4 UTF-8 LFPython 9:00 PM
2 ENG
5/22/2023
> OUTUNE 但我认为, 考虑到数据集并不大且不复杂, 这应该不是个大
WSL: Ubuntu00
> TIMELINE But l think given the data set, which is not so big and not so complicated, it should ENG
/22/2023
9:00 PM
> OUT U NE 问题, 因为我们无论如何都不会构建一个庞大的数据集.
WSL: Ubuntu00
> TIMELNE
Butl think given the data set, which is not so big and not so complicated, it should 9:00 PM
问题, 因为我们无论如何都不会构建一个庞大的数据集.
> OUTUNE
WSL: Ubuntu00
> TIMELNE
not be a big problem because we are not building a huge data set anyway.
4. 10. 6
5/22/2023
9:00 PM
> OUT UNE
> WSL: Ubuntu
> TIMELNE
Ln67, Col1 Spaces:4 UTF-8 LF Python3. 10. 6(trans
9:00 PM
2 ENG
5/22/2023
> OUT U NE 但在构建训练循环之前, 让我先定义这个配置, 因为它
> WSL : Ubuntu > TIME L NE
oknow that we have the model we can start building the training loop but before we 5/22/2023
9:00 PM
> OUT U NE 但在构建训练循环之前, 让我先定义这个配置, 因为它
> WSL Ubuntu > TIME LNE build the training loop let me define this configuration because it keeps coming and /22/2023
9:00 PM
所以让我们创建一个名为 config ·p y 的新文件, 在其中定义
> OUTU NE
> WSL : Ubuntu > TIMELINE it's better to define the structure now so let's create a new file called config. py. 人
/22/2023
9:00 PM
所以让我们创建一个名为 config ·p y 的新文件, 在其中定义
> OUTUI NE
WSL : Ubuntu > TIME UNE
Pytho
9:00 PM
/22/2023
所以让我们创建一个名为 config. p y 的新文件, 在其中定义
> OUTU NE
> WSL : Ubuntu > TIME LIN I
. o in which we define two methods, one is called get Config and one is. to get the path. 通
两个方法. 一个是get Config, 另一个是获取我们将保存模型权重的
> OUT UNE TIME LINI where we will save the weights of the model ok, let's define the batch size 3. o6tra
2 ENG
9:00 PM
> OUT UNE 路径. 好的, 让我们定义批量大小.
> WSL: Ubuntu10
> TIMELINI
where we will save the weights of the model ok, let's define the batch size 3. 10. 6t
2 ENG
9:00 PM
> OUTUNE
WSL: Ubuntu00
> TIME LNI
n3, Col 15
Spaces4 UTF-8 LFPython 3. 10. 6(tra
9:00 PM
P
ENG
5/22/2023
> OUT UNE 我选择8, 如果你的电脑允许, 你可以选择更大的值.
> WSL: Ubuntu00
> TIMEUINI
Ichoose 8, you can choose something bigger if your computer allows it.
4. 10. 6
9:01 PM
> OUT U NE
WSL : Ubuntu 10
> TIMELNE
n4. Col13
4 UIF-8 LFPytho 3. 10. 6 tra
9:01 PM
P
2 ENG
5/22/2023
> OUTUNE 我们将训练的轮数, 我认为20轮就足够了.
WSL: Ubuntu
TIMELNI
The number of epochs for which we will be training, I would say 20 is enough. 人
P
ENG
9:01 PM
> OUT U NE
WSL : Ubuntu 00
> TIMELINI
n5, Col9
UIF-8 LFPyth o
4. 10. 6(tra
9:01 PM
ENG
5/22/2023
学习率, 我使用的是 10 的负 4 次方.
> OUTUNE
WSL: Ubuntu > TIME LINI The learning rate, l am using 10 to the power of minus 4. urs u
pton 口 通
Z
2 ENG
9:01 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln5, Col 21
UTF-8 LF{ Pytho
4. 10. 6 (tra
9:01 PM
2 ENG
5/22/2023
> OUTUNE 你可以使用其他值.
WSL: Ubuntu00
> TIMELNI
You can use other values. 通
Z
P
4. 01 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
n6, Col 9
UTF-8 LFPytho
9:01 PM
2 ENG
5/22/2023
> OUT UNE 我认为这个学习率是合理的.
WSL: Ubuntu00
> TIMELNI
I thought this learning rate is reasonable. cou 9 spues4 urs u o ython 口 通
2 ENG
9:01 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
ENG
5/22/2023
9:01 PM
在训练过程中改变学习率是可能的.
> OUTUNE
WSL: Ubuntu10
> TIMELNI
It's possible to change the learning rate during training. us u yto n
10. 10. 6(tra
Z
ENG
5/22/2023
9:01 PM
> OUT U NE
> WSL : Ubuntu 10
> TIMELNE
n6. Col10
UIF-8 LF{ Pytho
4. 10. 6 (tr
9:01 PM
ENG
5/22/2023
实际上, 给予一个非常高的学习率, 然后随着每个轮次逐渐
> OUTUNE
> WSL: Ubuntu10
> TIMEUNE
Actually, it's quite common to give a very high learning rate and. then reduce it 办 ENG
9:01 PM
降低它, 这是相当常见的做法.
> OUTUNE
WSL: Ubuntu10
> TIMELNE
gradually with every epoch.
n6, Col10
9:01 PM 通
P
ENG
> OUT U NE
WSL : Ubuntu 10
> TIMEUNE
n6, Col10
es4 UIF-8 LFPytho 3. 10. 6 (tra
9:01 PM
P
ENG
5/22/2023
> OUTUNE 我们不会使用这种方法, 因为它会让代码稍微复杂一些, 而
WSL: Ubuntuβ10
> TIMEUNE
We will not be using it because it wil I just complicate the code a little more and 人
2 ENG
9:01 PM
> OUTUNE 这并不是本视频的真正目的.
WSL: Ubuntu10
> TIMELNE
this is not actually the goal of this video. 6. coli1o spaes. a uir& u i mthon
4. 10. 6(t
9:01 PM 通
2 ENG
> OUT UNE 本视频的目的是教授 Transformer 的工作原理.
WSL: Ubuntu
> TIMELINI
1 A0
The goal of this video is to teach how the transformer. works.
LF( Python
2 ENG
9:01 PM
> OUT U NE
WSL : Ubuntu > TIMELINE 1 A0
Ln6, Col10
4 UIF-8 LF Pytho
9:01 PM
P
ENG
5/22/2023
> OUT UNE 我已经检查过, 对于这个特定的从英语到意大利语的数据集
WSL: Ubuntu
> TIMELINE
2 A0
Ihave already checked the sequence length that we need for this particular dataset
> OUT UNE 我已经检查过, 对于这个特定的从英语到意大利语的数据集
WSL: Ubxuntu20
> TIMELIN
from English to Italian, which is 350 is more than enough. msu y tho
> OUT UNE 序列长度为350已经足够了.
WSL: Ubuntu20
> TIMELIN
from English to Italian, which is 350 is more than enough. mrs u 4 Aython
ENG
9:01 PM
> OUT U NE
WSL : Ubuntu > TIMELINE n6, Col24
Spaces:4 UTF-8 LFPytho
4. 10. 6 (t
9:01 PM
2 ENG
5/22/2023
> OUTUNE 我们将使用的 D模型是默认的512.
WSL: Ubuntu10
> TIMELINI And the D model that we will be using is the default of 512rsu4phom
4. 10. 6(tr
ENG
9:01 PM
> OUT UNE
> WSL: Ubuntu20
> TIMELNE
Ln 7, Col22
Pytho
9:02 PM
P
5/22/2023
> OUTUNE 源语言是英语.
> WSL: Ubuntu00
> TIMELNE
The language source is English.
Ln8. Col9 Spaces4 UTF-8 LFPython 3. 10. 6 (t
9:02 PM
Z
ENG
所以我们是从英语开始的.
> OUTUNE
WSL: Ubuntu00
> TIMELNE
So we are going from English. 通
9:02 PM
> OUT U NE
> WSL : Ubuntu > TIME UNE
n9. Col9
UIF-8 LF{ Pytho
9:02 PM
> OUT U NE 目标语言是意大利语.
> WSL: Ubuntu
> TIMELNE
001fle to analyze The language target is Italian.
n9, Col16
4. 10. 6 (tr
9:02 PM
P
ENG
> OUTUNE
> WSL: Ubuntu00
> TIMEUNE
Ln9. Col24
9:02 PM
ENG
5/22/2023
> OUT U NE 我们将翻译成意大利语.
> WSL : Ubuntu 00
> TIMEUNE
ENG
5/22/2023
9:02 PM
> OUT U NE
> WSL: Ubuntu00
> TIMEUNE
Ln10. Col9
4. 10. 6 (tra
9:02 PM
ENG
5/22/2023
我们将把模型保存到名为 weights 的文件夹中, 模型文件名将是
> OUTUNE
WSL: Ubuntuβ10
> TIMEUNE
we will save the model into the folder called weights and the file. nameof which 口
2 ENG
> OUT UI NE
tmodel, 即 Transformer 模型. 我还编写了代码, 以便在可能需要
WSL : Ubuntu > TIME UNE i model will be t model so transform model l also build the code i to. preload the. model. in 2 ENG
9:02 PM
> OUT U NE 重新启动训练时预加载模型, 例如在模型崩溃后.
WSL : Ubuntu > TIME UNE o model will be t model so transform model l also build the codeto. preload. the. model. in.
9:02 PM
重新启动训练时预加载模型, 例如在模型崩溃后.
> OUTUNE
WSL: Ubuntu10
> TIMEUNE
case we want to restart the training after maybe its crash mrs u
yton
2 ENG
5/22/2023
9:02 PM
> OUT UNE > TIME LNE
Ln13, Col9 Spaces4 UTF-8 LF { Python 3. 10. 6(tra
9:02 PM
ENG
5/22/2023
> OUT UNE 这是tokenizer 文件.
> TIMELINE and this is the tokenizer file so it will be saved like this so tokenizer n and
4. 10. 6 Ctra
ENG
9:02 PM
所以它将这样保存:根据语言分别保存为tokenizer _en 和tokenizer _it,
WSL : Ubuntu and this is the tokenizer file so it will be saved like this so tokenizer n and
5. 10. 6 (tra
ENG
9:02 PM
> OUT UNE 这是用于 Tensor Board 的实验名称, 我们将在训练过程中保存
WSL: Ubuntu00
> TIMELNE
tokenizer it according to the language and this is the experiment name. for a. osctra 人
2 ENG
9:03 PM
> OUT UNE 这是用于 Tensor Board 的实验名称, 我们将在训练过程中保存
> WSL : Ubuntu > TIMELINE tensor board on which we Ln 14, Col2
Pytho
4. 03 PM
2 ENG
/22/2023
> OUT UNE 这是用于 Tensor Board 的实验名称, 我们将在训练过程中保存
WSL : Ubuntu > TIME LNE will save the losses while training I think there is a comma. here, ok. yho
10. 10. 6 C
9:03 PM
> OUT UNE 损失. 我想这里有一个逗号.
> WSL: Ubuntu
> TIMELNE
3 A0
will save the losses while training I think there is a comma. here, ok. pth on
4. 10. 6 Car
ENG
5/22/2023
9:03 PM
> OUT U NE 好的, 现在让我们定义另一个方法, 用于找到保存权重的
> WSL : Ubuntu > TIMELINE will save the losses while training l think there is a comma. here, ok. ython 办 ENG
9:03 PM
> OUT U NE 好的, 现在让我们定义另一个方法, 用于找到保存权重的
> WSL : Ubuntu > TIMELINE Ln17, Col1
9:03 PM
> OUT U NE 好的, 现在让我们定义另一个方法, 用于找到保存权重的
> WSL : Ubuntu > TIMELINE Now, let's define another method that allows to find the path where we need to save 9:03 PM
> OUT U NE
> WSL : Ubuntu > TIMELINE 10
Ln17, Col5 Spaces4 UTF-8 LF Python ENG
5/22/2023
9:03 PM
> OUT UNE 我创建如此复杂的结构是因为我还将提供在 Google Cola b上运行此
WSL: Ubuntu10
> TIMELNE
Why I'm creating such a complicated structure is because. I will provide also. o ctan
ENG
9:03 PM
训练的笔记本.
> OUT UNE
> WSL: Ubuntu
> TIMELINE
10
notebooks to run this training on Google Colab.
Spaces4 UTF-8 LFPython 3. 10. 6(trar
9:03 PM 通
5/22/2023
> OUTUNE
WSL: Ubuntu
> TIMELNE
1 A0
4. 10. 6 (tra
9:03 PM
2 ENG
5/22/2023
> OUT UNE 因此, 我们只需更改这些参数, 使其在 Google Colab 上运行
> TIME UNE
并直接将权重保存到您的 Google Drive 中
> OUT UNE > TIME L NE
i So we just need to change these parameters to make it work on Google Colab and save > WSL: Ubuntu
5/22/2023
9:03 PM
并直接将权重保存到您的 Google Drive 中.
> OUT UNE
> WSL: Ubuntu10
> TIMELINE the weights directly on your Google Drive. co2s spaces. 4
Pytho
4. 10. 6(
9:03 PM 口
5/22/2023
> OUT U NE
> WSL : Ubuntu 10
> TIMEL INE
Ln 17, Col 28
4 UTF-8 LFPytho
4. 10. 6 (tr
9:03 PM
5/22/2023
> OUTUNE 我已经编写了这段代码, 它将在 Git Hub上提供, 我也会在
> WSL: Ubuntu10
> TIME UNE I have already created actually this code and it will be provided on Git Hub andl 人
2 ENG
/22/2023
9:03 PM
> OUT UN E 我已经编写了这段代码, 它将在 Git Hub上提供, 我也会在
> WSL: Ubuntu
> TIMELNE
1 A0
will also provide the link in the video. n1z. cd2s spaces4 ufs u (0 python 5/22/2023
9:03 PM
视频中提供链接.
> OUTUNE
> WSL: Ubuntu
> TIMEUNE
1 A0
4. 10. 6(ta
ENG
9:03 PM
Z
5/22/2023
> OUT UNE 好的, 文件是根据模型基本名称构建的, 然后是 epoch. pt.
> WSL: Ubuntu
> TIMEUNE
0 A1
okay the file is built according to model base name then. the epoch dotpt
9:04 PM 口
2 ENG
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME UNE Ln 20. Col50 Spaces4 UTF-8 LF ( Python 3. 10. 6(tran
ENG
5/22/2023
9:04 PM
> OUT U NE 让我们在这里也导入路径库. 好的, 现在让我们回到训练
> WSL : Ubuntu > TIME UNE let's import also here the path library okay now let's go back. to our. training loop. 口
5/22/2023
9:04 PM
循环. 好的, 我们终于可以构建训练循环了. 所以,
> OUT U NE
> WSL : Ubuntu > TIME LNE okay we can build the training loop now finally so train model given the 3. 10. 6(trar 人
ENG
5/22/2023
9:04 PM
> OUT UNE 根据配置, 首先我们需要定义将所有张量放置在哪个设备上.
> WSL: Ubuntu
> TIMELNE
1 A0
okay we can build the training loop now finally so train. model given the 口 人
/22/2023
9:05 PM
> OUT U NE 根据配置, 首先我们需要定义将所有张量放置在哪个设备上.
> WSL : Ubuntu > TIME LNE configuration okay
( Pytho
9:05 PM
P
/22/2023
69
> OUTUNE 根据配置, 首先我们需要定义将所有张量放置在哪个设备上
> WSL: Ubuntu > TIME LNE
1 A0
first we need to define which device on which we will put all the tensor sso define 人
/22/2023
9:05 PM
69
# Define tt
> OUTUNE 所以, 定义设备.
> WSL: Ubuntu > TIME LNE the device Ln69. Col16 Spa
UTF-8 LFPython 3. 10. 6(trans
2 ENG
5/22/2023
9:05 PM
69
Define the d
I
> OUTUNE
> WSL: Ubuntu > TIME LN E
1 A0
Ln 69, Col20 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
9:05 PM
5/22/2023
69
70
device = torch. device('cudsa # Define the device > OUT U NE 如果我的电脑有冷却时间, 我们也会打印出来
> WSL: Ubuntu
> TIMELNE
001fleto analyze If l have cool down on my computer...
Then we also print. urs u ython
3. 10. 6(tra
9:05 PM 人
5/22/2023
print (
device =
> OUT U NE
> WSL : Ubuntu > TIME UNE Ln 71, Col13 Spaces4 UTF-8 LF { Python 3. 10. 6(transfo
ENG
5/22/2023
9:05 PM
69
70
# Define the device 71
print(f Usiqg device (device }’) 我们确保创建了 weights 文件夹, 然后加载我们的数据集.
> OUT U NE
> WSL : Ubuntu > TIMELINE We make sure that the weights folder is created /22/2023
9:06 PM
69
70
device = torch. device ('cuda ′if torch. cuda. is _a = False )-> None # Define the device (mode:int=511, parents :bool = False, exist _ok:bool
71
print(f'Usipg device {device }')
Creat at this given path > OUT UNE 我们可以直接使用这里的值, 并将其设置为get DS
> WSL: Ubuntu
> TIMEUNE
/22/2023
9:06 PM
70 device = torch. device ('cuda ′if torch. cuda. is _available ()else‘cpu')
71
print(f'Using device {device }’) 我们可以直接使用这里的值, 并将其设置为get DS > OUT U NE
> WSL : Ubuntu > TIME UNE and then we load our data set we can just take these values here and say it's. equal to /22/2023
9:06 PM
print(f'Using device [device } 有一个名为get Vocab Size 的方法, 我想我们没有其他参数了
Path (config ['model _folder']).
> OUTUNE
> WSL: Ubuntu01
> TIMELINE get Ds. of config we create also the model to get the vocabulary size. there is a method 人
5/22/2023
9:06 PM
print(f'Using device [device }')
encode 72
Path(config ['model _folder']). mkdir (par > OUT UNE 有一个名为?
get Vocab Size 的方法, t我想我们没有其他参数了
WSL: Ubuntu
> TIMELINE
2 A0
called get Vocab Size
/22/2023
9:06 PM
print(f'Using device {device }’)
Path (config ['model _folder']). mkdir (parents = True, exist _ok = True )
> OUT UNE train _data loader, val _data loader, tokenizer _src, tokenizer _tgt =get _ds (config )
> TIME UNE model = get _model (config, tokenizer _src. get _vocab _size (), tokenizer _tgt. get _vocab _size ()
WSL: Ubuntu
Ln76. Col93 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :co 办 ENG
5/22/2023
9:07 PM
71
72
print(f'Using device {device }’) 最后, 我们将模型转移到我们的设备上
73
Path(config['
del_folder']). mkdir (parents = True, exist _ok = True )
> OUT U NE
WSL : Ubuntu > TIME UNE And finally we transfer the model to our device.
ENG
5/22/2023
9:07 PN
print(f'Using device {device }')
divmod dir Path (config ['model _folder']). mkdir (parents = True, exist _ok = True )
Data loader
s Data set > OUT UNE train _data loader, val _data loader, tokenizer _src, tokenizer _tgt = get_ds(config)
> TIMEUNE
76
model = get _model (config. L tokenizer _src. get _vocab _size (), tokenizer _tgt. get _vocab _size ()). to ()
> WSL: Ubuntu
Ln 76. Col99 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :co
ENG
5/22/2023
9:07 PM
Path (config ['model _folder']). mkdir (parents = True, exist _ok = True ) 我们还启动了 Tensor Board ab _size ()). to (device )
> OUTUNE
77
> WSLUbuntu > TIMELINE We also start Tensor Board.
Ln77, Col7 Spaces4 UTF-8 LF Python 3. 10. 6(trans fo
ENG
9:07 PM 通
5/22/2023
Path (config ['model _folder']). mkdir (parents = True, exist _ok = True )
train _data loader, val _data loader, tokenizer _src, tokenizer _tgt =get _ds (config )
model = get _model (config, tokenizer _src. get _vocab _size (), tokenizer _tgt. get _vocab _size ()). to (device )
> OUT UNE # Tensor board I > TIMELINE 78
> WSLUbuntu
Ln78. Col5 Spaces :4 UTF-8 LF Python 3. 10. 6trans fo
ENG
5/22/2023
9:07 PM
Path (config ['model _folder']). mkdir (parents = True, exist _ok= True)
74
get_ds(config)
> OUTUNE
> WSL Ubuntu > TIMELINE Tensor Board allows to visualize the loss, the graphic s, cthe. charts. yth on 3. 1o trns
9:07 PM 人
P
ENG
5/22/2023
Path (config ['model _folder ′]). mkdir (parents = True, exist _ok = True )
train _data loader, val _data loader, tokenizer _src, tokenizer _tgt = get _ds (config )
model =get _model (config, tokenizer _src. get _vocab _size (), tokenizer _tgt. get _vocab _size ()). to (device )
> OUT UNE # Tensor board I > TIME UNE
78
> WSL: Ubuntu
Ln78, Col5 Spaces 4 UTF-8 LFPython 3. 10. 6(trans 人
ENG
5/22/2023
9:07 PN
Path (config ['model _folder']). mkdir (parents = True, exist _ok = True )
get _ds (config )
ab _size ()). to (device )
> OUT UNE 让我们也导入 Tensor Board > WSL: Ubuntu
> TIMEUNE
78
Let'salso import Tensor Board.
3. 10. 6 (trans
ENG
9:07 PM
Path (config ['model _folder ′]). mkdir (parents = True, exist _ok = True )
model =get _mode train _data loader, val _data loa
nizer _tgt. get _vocab _size ()). to (device )
> OUT UNE # Tensor board > TIME UNE
78
writer = Summaryrxter
Okay, let's go back.
WSL : Ubuntu Ln78. Col5 Spaces:4 UTF-8 LFPython 3. 10. 6(transfo 人
ENG
5/22/2023
9:07 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln 14, Col50 Spaces:4 UTF-8 LF ( Python3. 10. 6(trans
ENG
9:07 PM
> OUT U NE 让我们也创建优化器.
> WSL : Ubuntu > TIME LNE Let's also create the optimizer.
Ln14, Col50 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
5/22/2023
9:08 PM
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 82, Col5 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
9:08 PM
我将使用 Adam 优化器.
> OUT U NE
WSL : Ubuntu > TIME L NE
I will be using the Adam optimizer.
Ln 82, Col5 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
9:08 PM
Z
P
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LNE
Ln 82. Col13 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans 办 ENG
5/22/2023
9:08 PM
> OUTUNE 好的, 由于我们还有配置允许在模型崩溃或某些东西崩溃时
WSL: Ubuntu00
> TIMEUNE
Okay, since we also have the configuration that allow us to resume the. training in. 人
ENG
/22/2023
> OUTUNE 好的, 由于我们还有配置允许在模型崩溃或某些东西崩溃时
> WSL: Ubuntu00
> TIMEUNE
case the model crashes or something crashes, let'simplement that. one rand that will ENG
/22/2023
9:08 PM
> OUT U NE 恢复训练, 让我们实现这一点, 这将允许我们恢复模型的
WSL : Ubuntu > TIMELINE allow us to Ln84, Col 5
Pytl
/22/2023
> OUT U NE 恢复训练, 让我们实现这一点, 这将允许我们恢复模型的
WSL : Ubuntu > TIMELINE restore the state of the model and the state of the optimizer. uwthon
/22/2023
9:08 PM
> OUTUNE 状态和优化器的状态.
WSL: Ubuntu00
> TIMELNE
restore the state of the model and the state of the optimizer. um on 3. o6tasto 办 ENG
9:08 PM
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIME LNE
Ln84, Col5 Spaces4 UIF-8 LF Python 3. 10. 6(trans 办 ENG
9:08 PM
5/22/2023
> OUT U NE 让我们导入我们在数据集中定义的方法. 我们加载文件.
> WSL : Ubuntu > TIME LNE let'simport this method we defined in the dataset /22/2023
9:09 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln91. Col28 Spaces 4 UTF-8 LF { Python 3. 10. 6(trans fo
ENG
9:09 PM
5/22/2023
96
> OUTUNE 这里有一个拼写错误. 好的, 我们将使用的损失函数是交叉
> WSLUbuntu
> TIMEUNE
here we have a typo okay the loss function we will be using is the cross ientropy loss
9:10 PM 人
ENG
> OUT U NE 熵损失. 我们需要告诉他忽略索引是什么.
> WSL : Ubuntu > TIMELINE here we have a typo okay the loss function we will be using is. the cross ientropy loss
5/22/2023
9:10 PM
> OUT U NE 熵损失. 我们需要告诉他忽略索引是什么.
> WSL : Ubuntu > TIMELINE ENG
5/22/2023
9:10 PM
所以我们不希望他忽略填充标记.
> OUT U NE
WSL : Ubuntu > TIME UNE we need to tell him what is the ignore index so we don't we want him to ignore the ENG
5/22/2023
9:10 PM
96 所以我们不希望他忽略填充标记.
> OUTUNE
WSL: Ubuntu
> TIMELNE
padding es4 UTF-8 LFPython3. 10. 6(trans f
Ln 96, Col 35 S
ENG
9:10 PM
5/22/2023
96
> OUTUNE 基本上, 我们不希望填充标记的损失对总损失有贡献, 我们
WSL: Ubuntu00
> TIME UNE token basically we don't want the loss to the padding token to contribute ito the loss 人
9:10 PM
> OUT UI NE 基本上, 我们不希望填充标记的损失对总损失有贡献, 我们
WSL : Ubuntu > TIME LNE
Pytho
9:10 PM 人
5/22/2023
loss _f n = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id [ PAD ]) 还将使用标签平滑.
> OUT U NE
WSL: Ubuntu
> TIMELNE
Ln 96, Col80 Spaces 4 UTF-8 LF ( Python 3. 10. 6(transformer :conda) 人
P
ENG
5/22/2023
9:11 PM
> OUT U NE 还将使用标签平滑.
> WSL: Ubuntu
> TIMEUNE
5/22/2023
9:11 PM
loss _f n = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]’)]
> OUT U NE
> WSLUbuntu
> TIMEUNE
Ln96, Col82 Spaces 4 UTF-8 LF Python3. 10. 6(transformer :co
9:11 PM 通
5/22/2023
96 标签平滑基本上允许我们的模型对其决策不那么自信.
> OUTUNE
WSL: Ubuntu
> TIMEUNE
Label smoothing basically allows us, our model, to be less confident about. it s 人
5/22/2023
9:11 PM
96
_to_id([ PAD]'),) 标签平滑基本上允许我们的模型对其决策不那么自信.
> OUTUNE
> WSLUbuntu > TIME LNE
decision.
Ln96, Col84 S
Python 3. 10. 6 (tra
9:11 PM
5/22/2023
loss _f n =nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]’),
> OUT U NE
> WSLUbuntu
> TIMEUNE
Ln96, Col84 Space :4 UF-8 IF( Python3. 10. 6 (transformer :cond)
9:11 PM
5/22/2023
> OUT U NE 所以, 打个比方, 想象一下我们的模型告诉我们选择第三个
WSL : Ubuntu > TIME LNE So, how to say, imagine our model is telling us to choose the word number three, 人
5/22/2023
9:11 PM
loss _f n = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id([ PAD]'),
> OUTUNE 词, 并且概率非常高.
WSL: Ubuntu00
> TIME LNE
Ln96. Col84 Spaces4 UTF-8 LF( Python 3. 10. 6(transfo
9:11 PM
5/22/2023
> OUT U NE 词, 并且概率非常高.
> WSL : Ubuntu > TIMELINE And with a very high probability.
Ln 96. Col84 Spaces4 UTF-8 LF ( Python 3. 10. 6(transformer :co
2
9:11 PM
5/22/2023
96
> OUTUNE 因此, 我们将通过标签平滑来取走一小部分那个概率, 并将
WSL: Ubuntu00
> TIMELINE So what we will do with labels booting is take a little percentage of that 3. 10. 6(tr
/22/2023
9:11 PM
96
> OUTUNE 因此, 我们将通过标签平滑来取走一小部分那个概率, 并将
> WSL: Ubuntu00
> TIME UNE probability and distribute to the other tokens so that our model becomes less sure of.
5/22/2023
9:11 PM
96
loss_fn = nn. Cross Entrc 其分配给其他标记, 从而使我们的模型对其选择不那么确定.
> OUTUNE
> WSL: Ubuntu00
> TIMEUNE
probability and distribute to the other tokens so that our model becomes less sure of.
9:11 PM 口 人
5/22/2023
96
_to_id([ PAD]'), D 其分配给其他标记, 从而使我们的模型对其选择不那么确定.
> OUTUNE
> WSL: Ubuntu > TIME UNE its choices.
Ln 96, Col 84
Pytho
9:11 PM
5/22/2023
loss _f n =nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]’),
> OUT U NE
> WSL: Ubuntu00
> TIMELNE
Ln96, Col 84 Spaces 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
9:11 PM
5/22/2023
> OUT U NE 这样就不会过度拟合.
> WSL : Ubuntu > TIME LNE So kind of less over fit.
Ln 96, Col84 Spaces 4 UTF-8 LF (↓ Python3. 10. 6(transformer :conda) 通
5/22/2023
9:11 PM
loss _f n = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id([ PAD]'), D 这实际上提高了模型的准确性.
> OUT U NE
WSL : Ubuntu > TIME LNE
9:11 PM 口
5/22/2023
loss _f n =nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ([ PAD ]’),
> OUT U NE
> WSLUbuntu
> TIMELNE
Ln96, Col84 Spaces 4 UTF-8 LF( Python3. 10. 6 (transformer :conda)
9:11 PM 通
5/22/2023
96
> OUTUNE 因此, 我们将使用0. 1的标签平滑, 这意味着从每个最高
> WSL: Ubuntu > TIMELINE 2 A0
So we will use a label spootingofo. 1, which means from every highest probability 9:11 PM 人
P
5/22/2023
96
> OUTUNE 因此, 我们将使用0. 1的标签平滑, 这意味着从每个最高
WSL: Ubuntu > TIME UNE
token, take 0. 1% of score and give it to the other s.
5/22/2023
9:11 PM
> OUTUNE 概率的标记中, 取0. 1
> WSL: Ubuntu > TIME LNE
token, take 0. 1% of score and give it to the other Soces. 4 ufrs u ( ython 3. o6 tansiomer cond)
9:11 PM 口
5/22/2023
loss _f n = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]'), label _smoothing =e. 1)
> OUTUNE
> WSL: Ubuntu > TIME UNE
Ln96, Col103 Space :4 UTF-8 IF( Python3. 10. 6 (transformer :conda)
9:11 PM 口
5/22/2023
thing =0. 1). to(device)
98
> OUTUNE 好的, 最后让我们构建训练循环,
WSL: Ubuntu > TIME LNE 通 办 ENG
5/22/2023
9:12 PM
loss _f n =nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]’), label _smoothing =o. 1). to(device)
98
> OUTUNE
WSLUbuntu > TIME UNE
Ln98, Col5 Spaces4 UTF-8 LF Python 3. 10. 6(trans
Z
ENG
5/22/2023
9:12 PM
loss _f n =
[e]decoder double oothing=o. 1). to(device )
For [e ]dump _patches model 我们告诉模型进行训练.
> OUT U NE
> WSL : Ubuntu > TIME UNE We tell the model to train.
Ln 99. Col15 Spaces:4 UTF-8 LFPython 3. 10. 6(transformer :co 办 ENG
9:12 PM
5/22/2023
1oss_fn = nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id([ PAD]’), label_smoothing=o. 1). to(device )
for epoch in range (initial _epoch, config [num _epochs ′]):
model. train train > OUT UNE [e )training > WSL: Ubuntu
> TIMEUNE
Ln 99. Col20 Spaces:4 UTF-8 LFPython3. 10. 6(transformer :co
ENG
9:12 PM
5/22/2023
97 我使用 TKODM为数据加载器构建了一个批次迭代器, 这将显示
> OUTUNE
> WSL: Ubuntu > TIME L NE
Ibuild a batch iterator for the data loader using TKo DM, which will show a very nice 9:12 PM
ENG
5/22/2023
othing =0. 1). to(device)
89
For
chs']):
100
model. train ()
batch... 个非常漂亮的进度条.
> OUT U NE
WSL : Ubuntu > TIMELINE I build a batch iterator for the data loader using TKo DM, which will show a very nice 9:12 PM
5/22/2023
loss _f n =nn. Cross Entropy Loss (ignore _index =tokenizer _src. token _to _id ('[ PAD ]’), label _smoothing =o. 1). to(device)
89
for
epoch in range (initial _epoch, config [num _epochs ‘]):
batch _iterator =|
model. train ()
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 100. Col26 Spaces4 UTF-8 LF{ Python 3. 10. 6(trans
ENG
5/22/2023
9:12 PM
97 我们需要导入 TQDM, 好的最后我们得到了张量, 编码器
98
> OUTUNE
WSL: Ubuntu > TIME UNE and we need to import T QDM ok, finally we get the tensors the encoder input octr
5/22/2023
9:12 PM
我们需要导入 T QDM, 好的, 最后我们得到了张量, 编码器
> OUT U NE
> WSLUbuntu
> TIMEUNE
1 A0
UTF-8 LFPython ENG
9:13 PM
5/22/2023
> OUT U NE 输入, 这个张量的大小是多少?
> WSL : Ubuntu > TIME UNE
1 A0
ENG
5/22/2023
9:13 PM
> OUT U NE 输入, 这个张量的大小是多少?
> WSL : Ubuntu > TIME UNE what is the size of this tensor?
4. 10. 6(trar
ENG
9:13 PM
5/22/2023
它是批次到序列长度. 解码器输入是解码器输入的批次
> OUT U NE
> WSL : Ubuntu > TIMELINE it's batch to sequence length the decoder input is batch of decoder input and we also 9:13 PM
5/22/2023
> OUT U NE 我们也将其移动到我们的设备上. 批次到序列长度. 我们还
> WSL : Ubuntu > TIMELINE 5/22/2023
9:13 PM
> OUTUNE 我们也将其移动到我们的设备上. 批次到序列长度. 我们还
> WSL: Ubuntu00
> TIMELINE move it to our device batch to sequence length we get the two. masks also ENG
5/22/2023
9:13 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln 107. Col 16 Spaces 4 UTF-8 LF( Python3. 10. 6(trans fc
ENG
9:13 PM
5/22/2023
> OUT U NE 这是大小, 然后是解码器掩码. 好的, 为什么这两个掩码
WSL : Ubuntu > TIME UNE this is the size and then the decoder mask okay why these two masks are different 5/22/2023
9:14 PM
不同?
> OUT U NE
> WSL : Ubuntu > TIME LNE because in the one case we are only telling him to hide UTF-8 LFPython 3. 10. 6(transfo
9:14 PM 口
5/22/2023
> OUT U NE 因为在一种情况下, 我们只告诉他隐藏填充标记. 在另一种
> WSL : Ubuntu > TIME LNE because in the one case we are only telling him to hide uf& u ython 办 ENG
5/22/2023
9:14 PM
> OUT U NE 情况下, 我们还告诉他隐藏每个词的所有后续词, 以隐藏
> WSL : Ubuntu > TIMELINE Pytho
9:14 PM
5/22/2023
> OUT U NE 情况下, 我们还告诉他隐藏每个词的所有后续词, 以隐藏
> WSL : Ubuntu > TIME UNE only the padding tokens in the other case we are also telling him. to hide all this 人
5/22/2023
9:14 PM
> OUT UNE 好的, 现在我们运行. 让我们让张量通过 Transformer.
WSL : Ubuntu > TIMELINE subsequent words for each word to hide all the subsequent words. to mask them out okay. 人
5/22/2023
9:14 PM
好的, 现在我们运行. 让我们让张量通过 Transformer.
> OUT U NE
WSL : Ubuntu > TIMELINE now werun
Ln 110. Col 13 S
4 UTF-8 LFPython 9:14 PM
5/22/2023
> OUT UNE 好的, 现在我们运行. 让我们让张量通过 Transformer.
WSL : Ubuntu > TIMELINE the um let'smake some run the tensors through the transformer so first we calculate 5/22/2023
9:14 PM
> OUT U NE 因此, 首先我们计算编码器的输出, 并使用编码器输入和
> WSL : Ubuntu > TIMELINE the um let'smake some run the tensors through the transformer so first we calculate 9:14 PM
5/22/2023
> OUTUNE 因此, 首先我们计算编码器的输出, 并使用编码器输入和
WSL: Ubuntu00
> TIMELNE
the output of the encoder Ln111. Col13 Spac
UTF-8 LFPytho
9:14 PM
Z
5/22/2023
> OUT U NE 因此, 首先我们计算编码器的输出, 并使用编码器输入和
> WSL : Ubuntu > TIME LNE and we encode using the encoder input and the mask of the encoder. ytho 人
ENG
5/22/2023
9:14 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln 111, Col64 Spaces :4 UTF-8 LFPython3. 10. 6(transformer :co
ENG
5/22/2023
9:15 PM
> OUT U NE 然后我们使用编码器输出、编码器掩码、解码器输入和
> WSL : Ubuntu > TIME LNE Then we calculate the decoder output using the encoder output, the mask of. the 口 人
P
5/22/2023
9:15 PM
> OUT U NE 然后我们使用编码器输出、编码器掩码、解码器输入和
WSL : Ubuntu > TIME LNE encoder, then the decoder input and the decoder amask. urs u 4 yhon
4. 10. 6 (tra
9:15 PM
2 ENG
5/22/2023
> OUT U NE
WSL : Ubuntu > TIME LN E
Ln 112. Col74 Spaces:4 UTF-8 LF( Python 3. 10. 6(transt
ENG
5/22/2023
9:15 PM
> OUTUNE 好的, 正如我们所知, 这个的结果. 因此, 模型编码的
> WSL: Ubuntu00
> TIMELNE
okay as we know the result of this so the output of the model. encode will be a. batch. 人
ENG
5/22/2023
9:15 PM
> OUTUNE 输出将是批次序列长度-d模型. 同样, 解码器的输出
xwu o△ Ssequence length d model also the output of the decoder will be batch sequence. length.
> TIMELINE 5/22/2023
9:15 PM
将是批次序列长度, d 模型
> OUTUNE
> TIMEUNE
5/22/2023
9:15 PM
> OUT UNE 将是批次序列长度, d模型
> WSL Ubuntu > TIME LNE
dmodelbut
Ln 112, Col124 Spaces 4 UTF-8 LF Python3. 10. 6(trans fc
ENG
9:15 PM
5/22/2023
> OUT U NE 但我们想将其映射回词汇表, 所以我们需要投影.
> WSL Ubuntu > TIME L NE
dmodel but
Ln 112, Col125 S
4. 10. 6 (tra
Python 9:15 PM
Z
ENG
5/22/2023
> OUT UNE 所以让我们获取投影输出. 这将产生一个, B, 即批次序列
> WSL: Ubuntu00
> TIMELNE
we want to map it back to the vocabulary so we need the projection so let's get the ENG
5/22/2023
9:15 PM
> OUTUNE 所以让我们获取投影输出. 这将产生一个, B, 即批次序列
WSL: Ubuntu
> TIMELNE
projection output P
5/22/2023
9:15 PM
> OUT UNE 所以让我们获取投影输出. 这将产生一个, B, 即批次序列
WSL : Ubuntu > TIMELINE And this will produce a B, so batch sequence length and target vocabulary size. ctr
5/22/2023
9:15 PM
> OUT U NE
WSL : Ubuntu > TIME UNE Ln 113. Col75 Spaces4 UTF-8 LF Python 3. 10. 6(transfo 口
5/22/2023
> OUT U NE 好的, 现在我们有了模型的输出, 我们想将其与我们的标签
> WSL Ubuntu > TIME UNE
9:16 PM 人
5/22/2023
进行比较.
> OUT U NE
> WSL : Ubuntu > TIMELINE 5/22/2023
9:16 PM
> OUT U NE 所以首先让我们从批次中提取标签.
> WSL : Ubuntu > TIMELINE So first let'sextract the label from the batch.
Spaces 4 UIF-8 LF( Python3. 10. 6 Ctransformer :co
ENG
5/22/2023
9:16 PM
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 115, Col19 Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :conda)
ENG
9:16 PM
5/22/2023
> OUT U NE 我们也将其放在我们的设备上.
WSL : Ubuntu > TIME LNE and we also put it on our device.
Ln 115, Col37 Spaces4 UTF-8 LFPython 3. 10. 6 (trans fo
ENG
9:16 PM 口 通
5/22/2023
> OUT U NE 那么标签是什么?
WSL : Ubuntu > TIMELINE So what is the label?
Ln 115, Col47 Spaces4 UTF-8 LF( Python 3. 10. 6(transformer :conda)
9:16 PM 通
Z
P
5/22/2023
> OUT UNE 因此, 标签已经为每个b和序列长度, 因此对于每个
WSL: Ubuntu
> TIMEUNE
It's B, so batch to sequence length, in which each position tell, so the label is uctar
9:16 PM 人
5/22/2023
> OUT UNE 因此, 标签已经为每个b和序列长度, 因此对于每个
WSL : Ubuntu > TIME UNE already,
Ln 115, Col 18 (5 selected)
UTF-8 LFPytho 3. 10. 6 (tra
9:16 PM 通
P
ENG
5/22/2023
> OUTUNE 因此, 标签已经为每个b和序列长度, 因此对于每个
WSL: Ubuntu00
> TIMELINE for each b and sequence length so for each dimension tells us what is. the position in 通
5/22/2023
9:16 PM
维度, 告诉我们该特定词在词汇表中的位置.
> OUT U NE
> WSL : Ubuntu > TIME UNE for each b and sequence length so for each dimension tells us what is. the position in 9:16 PM
5/22/2023
> OUT U NE 维度, 告诉我们该特定词在词汇表中的位置.
> WSL : Ubuntu > TIME UNE. the vocabulary of that particular word and we want these two. to. be comparable so we
9:16 PM
5/22/2023
> OUT UNE 我们希望这两者是可比较的, 所以我们首先需要计算损失到
> TIMEUNE
5/22/2023
9:16 PM
> OUT U NE 我们希望这两者是可比较的, 所以我们首先需要计算损失到
> WSL : Ubuntu > TIME LNE
first
n 117, Col 13
UTF-8 LFPytho
9:16 PM
5/22/2023
> OUT U NE 我们希望这两者是可比较的, 所以我们首先需要计算损失到
> WSL : Ubuntu > TIME LNE need to compute the loss into this i show you now projection output view minus. one 5/22/2023
9:16 PM
> OUT U NE 这个. 我现在展示给你投影输出视图减一,
> WSL : Ubuntu > TIME LNE need to compute the loss into this i show you now projection output view minus one 9:17 PM 口 办 ENG
5/22/2023
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 117, Col47 Spaces4 UIF-8 LF { Python 3. 10. 6(transfo
9:17 PM
5/22/2023
> OUT U NE 好的, 这是做什么的?
> WSL : Ubuntu > TIME UNE Okay, what does this do?
Ln 117. Col79 Spaces4 UTF-8 LFPython 3. 10. 6(transfo
ENG
5/22/2023
9:17 PM
这基本上转换了... 我会在这里展示给你.
> OUTUNE
> WSL: Ubuntu00
> TIMEUNE
This basically transforms the... Il'll showyou here.
4 UTF-8 LFPython 3. 10. 6(trans
5/22/2023
9:17 PM
> OUT U NE 这个大小变成这个大小.
> WSL : Ubuntu > TIME UNE This size into this size.
Ln 113, Col59 Spaces4 UTF-8 LF Python 3. 10. 6(trans fo
ENG
9:17 PM 通
5/22/2023
> OUT UNE
P乘以序列长度, 然后是目标词汇表大小.
WSL : Ubuntu > TIME UNE
P multiplied by sequence length and then target vocabulary size. pto 3. ios traobon
ENG
9:17 PM 人
5/22/2023
词汇表大小.
> OUT U NE
> WSL : Ubuntu > TIMELINE Vocabulary size.
Ln 117, Col 73 Spaces4 UTF-8 LF Python 3. 10. 6(transformer :cond
ENG
9:18 PM
5/22/2023
> OUTUNE 好的, 因为我们想将其与这个进行比较.
> WSL: Ubuntu00
> TIMELINE Okay, because we want to compare it with. this.
ENG
9:18 PM 通
5/22/2023
> OUT U NE
> WSL Ubuntu > TIME LNE
Ln 115, Col47 (29 selected ) Spaces : 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
ENG
9:18 PM 口
Z
5/22/2023
> OUT U NE 这就是交叉希望张量的样子.
> WSL Ubuntu > TIME LNE
ENG
4. 18 PM 口
5/22/2023
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln 118. Col 82 Spaces 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
9:18 PM
5/22/2023
> OUT U NE 我们可以更新我们的进度条, 这个用我们计算的损失, 这将
> WSL : Ubuntu > TIME LNE and also the label okay now we can we have calculated the loss we can update our ENG
5/22/2023
9:18 PM
> OUT U NE 我们可以更新我们的进度条, 这个用我们计算的损失, 这将
> WSL : Ubuntu > TIME LNE progress bar this one with the loss we have calculated.
5/22/2023
9:18 PM
> OUT U NE 我们可以更新我们的进度条, 这个用我们计算的损失, 这将
> WSL : Ubuntu > TIME UNE UIF-8 LFPytho
9:18 PM
5/22/2023
False File Exists Erro > OUT UNE 在我们的进度条上显示损失. 我们也可以在 Tensor Board 上记录它.
> WSL : Ubuntu > TIME UNE 3 A0
Ln 119, Col51 Spaces4 UTF-8 LF Python 3. 10. 6 (tra
9:18 PM
ENG
5/22/2023
在我们的进度条上显示损失. 我们也可以在 Tensor Board 上记录它.
> OUT U NE
> WSL : Ubuntu > TIME LNE and this will show the loss on our progress bar we can also log it on. tensor board 人
P
ENG
5/22/2023
9:18 PM
让我们也刷新它.
> OUT U NE
WSL : Ubuntu > TIMELINE let's also flush it Ln 122, Col56 Spaces:4 UTF-8 LFPython 3. 10. 6(transfo 通
5/22/2023
9:19 PM
> OUT U NE
WSL: Ubuntu
> TIMELNE
Ln 123, Col 24 Spaces : 4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/22/2023
9:19 PM
> OUT UNE 好的, 现在我们可以反向传播损失, 所以loss backward, 最后我们
> WSL : Ubuntu > TIME LNE okay now we can back propagate the loss so loss. backward and finally we update. the 人
ENG
5/22/2023
9:19 PM
129
> OUTUNE 更新模型的权重. 这就是优化器的工作.
WSL: Ubuntu > TIME UNE okay now we can back propagate the loss so loss. backward and finally we update the 9:19 PM 人
5/22/2023
129 optimizer. step ()
> OUT U NE 最后我们可以将梯度清零, 并将全局步数加一. 全局步数
WSL : Ubuntu > TIME UNE weights of the model so that is the job of the optimizer and finally we can zero out
9:19 PM 人
ENG
5/22/2023
129 optimizer. step ()
> OUT U NE 最后我们可以将梯度清零, 并将全局步数加一. 全局步数
> WSL : Ubuntu > TIME UNE the grad Ln 130, Col32 S
5/22/2023
9:19 PM
129 optimizer. step ()
> OUT UNE 最后我们可以将梯度清零, 并将全局步数加一. 全局步数
> WSL: Ubuntu10
> TIMEUNE
and we move the global step by one the global step is being used mostly for. tensor 9:19 PM
ENG
5/22/2023
128
# Update the weights
130
optimizer. step () 主要用于 Tensor Board 来跟踪损失.
> OUT U NE
> WSL : Ubuntu > TIME UNE and we move the global step by one the global step is being used mostly for. tensor ENG
9:19 PM
5/22/2023
# Update the weights 130 optimizer. step () 主要用于 Tensor Board 来跟踪损失.
> OUT U NE
WSL : Ubuntu > TIME UNE board to keep track of the loss we can save the model UTF-8 LFPython3. 10. 6(trans fc 办 ENG
5/22/2023
9:19 PM
130
131
optimizer. zero _grad() 我们可以在每个 epoch 保存模型, 好的, 模型文件名
global_step +=1
> OUTUNE
WSL: Ubuntu > TIME UNE board to keep track of the loss we can save the model UTF-8 LF{ Pyhc
3. 10. 6(tra
9:20 PM
5/22/2023
130
131
optimizer. zero _grad() 我们可以在每个 epoch保存模型, 好的, 模型文件名
132
gl obal_step +=1
> OUTUNE
WSL: Ubuntu > TIME UNE
4. 10. 6(t
9:20 PM
5/22/2023
13
132
odelfile
> OUTUNE 我们从我们的特殊方法中获取, 这个.
WSLUbuntu > TIME UNE
5/22/2023
9:21 PM
131
130
optimizer. zero _grad() 我们告诉他我们的配置和我们文件的名称;这是epoch, 但前面
3
global _step > OUT UNE > TIMELINE 9:21 PM
ENG
5/22/2023
optimizer. zero _grad ()
133
global_ste
134 有零, 然后我们保存我们的模型
h:str)->str
> OUTUNE 135
> WSL: Ubuntu
> TIMEUNE
134
with zeros in front Ln 135, Col69
P
5/22/2023
9:21 PM
global _step += 1
134
135 有零然后我们保存我们的模型
> OUTUNE
136
> WSL: Ubuntu > TIME LNE and we save our model it is very good idea when we want to. be able to resume the 9:21 PM 口
5/22/2023
136
137
torch. save 当我们希望能够恢复训练时, 保存不仅模型的状态, 还有
> OUTUNE
WSL: Ubuntu00
> TIMELNE
training to also save not only the state of the model but also the state of the ostra 人
5/22/2023
9:21 PM
137 优化器的状态, 这是一个非常好的主意, 因为优化器也会
> OUTUNE
WSL: Ubuntu > TIMELINE optimizer because Ln 137, Col 13
Pytho
9:21 PM
5/22/2023
136
137
torch. save 优化器的状态, 这是一个非常好的主意, 因为优化器也会
> OUTUNE
WSL: Ubuntu > TIMELINE the optimizer also keep tracks of some statistics one for each weight pyu
4. 10. 6(ta
5/22/2023
9:21 PM
137
torch. save 跟踪一些统计数据, 每个权重一个, 以理解如何独立移动
> OUTUNE
WSL: Ubuntu > TIME UNE the optimizer also keep tracks of some statistics one for each weight rha
4. 10. 6 C
ENG
/22/2023
9:21 PM
137
136
torch. save
> OUTUNE 跟踪一些统计数据, 每个权重一个, 以理解如何独立移动
> WSL: Ubuntu > TIME UNE
Pytho
9:21 PM
5/22/2023
136
137
torch. save 跟踪一些统计数据, 每个权重一个, 以理解如何独立移动
> OUTUNE
WSL: Ubuntu > TIME UNE to understand how to move each weight independently and usually actually l saw. that
/22/2023
9:21 PM
137
> OUTUNE 每个权重, 通常实际上, 我发现优化器字典相当大.
WSL: Ubuntu
> TIMEUNE
oto understand how to move each weight independently and usually actually I saw. that.
5/22/2023
9:21 PM
137
> OUTUNE 每个权重, 通常实际上, 我发现优化器字典相当大.
WSL: Ubuntu
> TIMELNE
the optimizer dictionary is quite big so even if it's big if you want your training cm
ENG
5/22/2023
9:21 PM
137
> OUTUNE 所以即使它很大, 如果你想让你的训练可恢复, 你需要保存
WSL: Ubuntu > TIME UNE the optimizer dictionary is quite big so even if it's big if you want your training sm
9:21 PM
ENG
/22/2023
136
137
torch. save
> OUTUNE 所以即使它很大, 如果你想让你的训练可恢复, 你需要保存
> WSL: Ubuntu > TIME UNE
to be
n137. Col13
( Pytho
9:21 PM
5/22/2023
136
137
torch. save 工
> OUTUNE 所以即使它很大, 如果你想让你的训练可恢复, 你需要保存
xwsuu nu oresum able you need to save it otherwise the optimizer will always start from zero. and > TIME LNE
5/22/2023
9:22 PM
136
137
torch. save 它, 否则优化器总是会从零开始, 并且必须从零开始
> OUTUNE
xwum oresum able you need to save it otherwise the optimizer will always start from zero and.
> TIME LNE
9:22 PM 人
5/22/2023
136
137
torch. save 它, 否则优化器总是会从零开始, 并且必须从零开始
> OUTUNE
> TIME LNE
9:22 PM 口
ENG
5/22/2023
136
137
torch. save 即使你从个之前的epoch开始一一女 如何移动每个权重.
> OUTUNE
> TIMELNE
9:22 PM
5/22/2023
136
137
torch. save 即使你从个之前的 epoch开始 一一 如何移动每个权重.
> OUTUNE
> WSL : Ubuntu > TIME LNE
each weight
Ln137. Col13 S
Pytho
9:22 PM
5/22/2023
torch. save
D
> OUTUNE
140
> WSL: Ubuntu
> TIMELNE
Ln 137. Col 13 Spaces:4 UTF-8 LF Python 3. 10. 6(trans
9:22 PM
5/22/2023
137 所以每次我们保存一些快照时, 我总是包含它.
> OUTUNE
> WSL: Ubuntu
> TIMELNE
So every time we save some snapshot, I always include it. ns u
pthon
4. 10. 6(tra
9:22 PM 人
5/22/2023
torch. save I
D
'epoch':d > OUT UNE
140
> WSL: Ubuntu
> TIMELNE
3 A0
Ln 137, Col 23 Spaces4 UTF-8 LF ( Python 3. 10. 6 (translo
9:22 PM
5/22/2023
13
torch. save
138
model _state l'epoch':epoch,
D 模型的状态.
> OU TUNE > WSL: Ubuntu
> TIMELNE
101fleto analyze The state of the model.
Ln 138, Col26 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
9:22 PM 中
P
5/22/2023
13
torch. save
I
138
model _state _dict'
epoch':epoch,
> OUTUNE
140
141
D
> WSL: Ubuntu > TIME LNE
Ln 138, Col31 Spaces :4 UTF-8 LFPython3. 10. 6(trans fo
ENG
9:22 PM
5/22/2023
13
epoch':epoch,
model _state _dict':
model. state I > OUT UNE 这是模型的所有权重.
WSL: Ubuntu
> TIMELNE
3 A0
This is all the weights of the model.
Ln 138. Col44 Spaces:4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/22/2023
9:22 PM
to re See also : saving-loading-tensors 13
> OUTUNE
141
140
WSL: Ubuntu > TIME LNE 142
Ln 139, Col13 Spaces4 UTF-8 LF Python 3. 10. 6(trans
ENG
9:22 PM
5/22/2023
torch. save (
13
epoch':epoch,
odel
dict':
odel. state _dict()
> OUTUNE
141
140 我们也想保存优化器.
WSL: Ubuntu > TIME LNE
142
ENG
9:22 PM
5/22/2023
13'model _state _dict':model. state _dict (),
epoch':epoch,
I
140
'optil
> OUTUNE
141
> WSLUbuntu > TIMELINE 1 A0
142
Ln 139, Col 18 Spaces :4 UTF-8 LFPython3. 10. 6(transformer :co
ENG
9:22 PM
Z
5/22/2023
137
aving-load
> OUTUNE 让我们也保存全局步数, 我们想把所有这些保存到文件名中.
WSL: Ubuntu > TIME UNE let's do also the global step and we want to save all this into the file name soocm
ENG
5/22/2023
9:22 PM
136
137
use_new_zipfl_serializ a
n= True )
Saves an object to a dis fe > OUT U NE 所以模型, 文件名, 就这样. 现在让我们编写代码来
WSL : Ubuntu > TIMELINE let's do also the global step and we want to save all this into the file name soostar
ENG
9:22 PM
> OUT U NE 所以如果名字是, 我真的发现警告很烦人, 所以我想过滤掉
WSL : Ubuntu > TIME LNE model file name and that's it now let'sbuild the code to run this so if name is octr
ENG
5/22/2023
9:22 PM
> OUT U NE 所以如果名字是, 我真的发现警告很烦人, 月 所以我想过滤掉
WSL : Ubuntu > TIME UNE
1 A0
Pytho
9:23 PM
5/22/2023
> OUT UNE 所以如果名字是, 我真的发现警告很烦人, 所以我想过滤掉
WSL: Ubuntu
> TIMELINE
1 A0
Ireally find the warnings frustrating so I want to filter them out because lhave. a
5/22/2023
9:23 PM
max_len_src =
max_l en_tgt
max(max_len_src, 1en(src_ids))
x_len_tgt, len(tgt_ids) 它们因为我有很多库尤其是 CUDA.
> OUTUNE
WSL: Ubuntu > TIME UNE lot of libraries, especially Cu DA, I already know what's the content and so. ldon't
9:23 PM
5/22/2023
42
tokenizer_tgt 我已经知道内容是什么所以我不希望每次都看到它们. 但
> OUTUNE
WSL: Ubuntu > TIME LNE
10
lot of libraries, especially cu DA, i already know what's the content and so. l don't.
9:23 PM
5/22/2023
我已经知道内容是什么, 所以我不希望每次都看到它们. 但
tokenizer _tgt > OUT U NE
WSL : Ubuntu > TIME LNE
10
wantto
Ln19. Col1 S
Pytho
4. 10. 6(ta
9:23 PM
5/22/2023
42
tokenizer_tgt
config ['lang _src']) 我已经知道内容是什么, 所以我不希望每次都看到它们. 但
> OUT U NE
> WSL : Ubuntu > TIME LNE
1 A0
5/22/2023
9:23 PM
41
# Build tokenizer s
42
43
tokenizer _src = get_or_build_to kenizer(config, ds_raw
tokenizer_tgt
'lang_tgt']
> OUTUNE 对你来说, 我建议至少看一次以了解是否有任何大问题.
> WSL: Ubuntu
> TIMEUNE
1 A0
visualize them every time but for sure for you guys I suggest watching them at least.
5/22/2023
9:23 PM
> OUT UNE 否则它们只是在抱怨 CUDA.
> WSL: Ubuntu
> TIMELNE
1 A0
once to understand if there is any big problem otherwise they're just complaining.
5/22/2023
9:23 PM
> OUT UNE 否则它们只是在抱怨 CUDA.
> WSL: Ubuntu
> TIMELNE
1 A0
from CUDA.
Ln146, Col1 Spaces4 UTF-8 LFPython 3. 10. 6(tran 通
5/22/2023
9:23 PM
> OUT U NE
> WSL : Ubuntu > TIME LNE
1 A0
Ln146, Col1 Spaces4 UTF-8 LF↓ Python
ENG
5/22/2023
9:23 PM
> OUT U NE 好的, 让我们尝试运行这段代码, 看看是否一切正常,
> WSL : Ubuntu > TIMELINE okay let's try to run this code and see if everything is working fine we should what 5/22/2023
9:23 PM
max_len_src=0
55
54
max_len_tgt=0
578
for item in ds_raw: 应该没问题
> OUTUNE
tgt_ids= tokenizer src _ids = tokenizer n fig [*lang _tgt']]). ids
fig['lang_src']]). ids
> WSL: Ubuntu > TIME UNE okay let's try to run this code and'see if everything is working fine we should what 5/22/2023
9:23 PM
108
109
> OUTUNE 我们期望的是,? 代码应该在第一次下载数据集, 然后创建
WSL: Ubuntu00
> TIMELINE okay let's try to run this code and see if everything is working fine we should what ENG
5/22/2023
9:23 PM
129 optimizer. step ()
> OUT UNE tokenizer 并将其保存到文件中, 并且它还应该开始训练模型30
> WSL: Ubuntu
> TIMEUNE
create Ln 146, Col 24
9:23 PM
5/22/2023
129 optimizer. step ()
> OUT UNE tokenizer 并将其保存到文件中, 并且它还应该开始训练模型30
WSL: Ubuntu
> TIMEUNE
the tokenizer and save it into its file and it should also start training the model ENG
5/22/2023
9:23 PM
> OUT UNE tokenizer 并将其保存到文件中, 并且它还应该开始训练模型30
WSL: Ubuntu
> TIMEUNE
for30 epochs of course it will never finish. but
ENG
5/22/2023
9:24 PM
当然, 它永远不会完成, 但让我们开始吧.
> OUT UNE
> WSL: Ubuntu
> TIMEUNE
for30 epochs of course it will never finish. but
9:24 PM 通
5/22/2023
120
121
2
# Log the loss
lar('tra
> OUT U NE 当然, 它永远不会完成, 但让我们开始吧
WSL : Ubuntu > TIME UNE Loss. backward ()
Ln 146, Col 24 S
UTF-8 LFPython 3. 10. 6(trans
ENG
5/22/2023
9:24 PM
# Log the loss writer. add _scalar ('train loss', loss. item (), global _step )
writer. flush ()
> OUT UNE
125
# Back propagate the loss > WSL : Ubuntu > TIME UNE 126
loss. backward()
Ln 146, Col24 Spaces:4 UTF-8 LF( Python 3. 10. 6 (transformer : conda)
9:24 PM 通
5/22/2023
> OUT U NE 让我再检查一下配置.
WSL Ubuntu > TIMELINE Let me check again the configuration. Ln. col25s spaces4 urs u ython 3. 10. 6tan
ENG
9:24 PM
P
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Lnit6: Gol41(351 selected ) Spaces 4 UTF-8 LF( Python ENG
5/22/2023
9:24 PM
> OUT U NE 好的, 让我们运行它.
> WSL : Ubuntu > TIMELINE Okay, let's run it.
Ln 16. Col41 Spaces 4 UTF-8 LF { Python3. 10. 6 Ctrans fo
9:24 PM
P
5/22/2023
> OUT U NE
> WSL : Ubuntu > TIMELINE Ln 16. Col41 Spaces4 UTF-8 LF { Python 3. 10. 6 Ctrans fo
ENG
5/22/2023
9:24 PM
AK POINTS Using device dataset opus _books (/home /kira /. cache/
d39eb594746af2daf )
Raised Exceptions Uncaught Exceptions 好的, 它在构建tokenizer, 我们这里有一些问题
WSL : Ubuntu User Uncaught Exceptions ENG
5/22/2023
9:24 PM
AK POINT S
ound
Uncaught Exceptions Raised Exceptions > WSL : Ubunt
User Uncaught Exceptions Ln 50, Col1 Spaces4 UTF·8 LFPython 3. 10. 6 (transformer conda ) A 口
AK POINTS ging face /dat
en-it/1. 0. 0/e8f958a4f32dc39b7f9888988216cd2d7e21ac35f8 93d84d39eb594746af2daf)
Uncaught Exceptions Raised Exceptions Be:00:ee] Pre-processing sequences WSL : Ubunt
User Uncaught Exceptions Sequence length.
Ln 50, Col 1i Spaces:4 UTF-8 LFPython 3. 10. 6 (transformer :conda ) A 口
AK POINT S
ound
books /en-it/1. 0. 0/e8f958a4f32dc39b7f9888988216cd2d7e21ac35f 893d84d39eb594746af2daf )
Raised Exceptions Uncaught Exceptions > WSL : Ubunt
User Uncaught Exceptions Ln 50, Col1 Spaces:4 UTF-8 LF( Python 3. 10. 6(transformer conda) A 口
NG5
9:24 PM
Using > OUTUNE
lax
> WSL: Ubunt
> TIMELINE Processing epoc l Okay, finally the model is training.
1/3638[00:01<48:30, 1. 25it/s, 1oss=9. 800]
9:29 PM
5/22/2023
-from-scratch ;/us r/bin/env/h
> OUTUNE
axler
engthof so
argee sentence :30
WSL: Ubunt
> TIMELINE Processing epoch B0:
Ln46. Col31 Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :conda)
5/3638[00:03<33:21, 1. 82it/s, 1oss=9. 456]
9:29 PM
5/22/2023
kira /
scratch /train.
iles /lib /python /de > OUT U NE 我给你们回顾一下我犯的错误
WSL: Ubunt
> TIMELINE
7/3638[00:04<28:32, 2. 12it/s, 1oss=9. 472]
9:29 PM
> OUT UNE ength of tp
d
cee sentence:27
> WSL: Ubunt
> TIME UNE Processing epoch e e:e%
Ln46, Col31 Spaces4 UTF-8 LFPython 3. 10. 6 (transf
15/3638[00:07<27:20, 2. 21it/s, 1oss=9. 225]
9:29 PM
ects /ti device cud
> OUTUNE
lax
Length
WSL: Ubunt > TIME UNE First of all,
16/3638[00:07<30:50, 1. 96it/s, 1oss=9. 126]
ENG
9:29 PM
5/22/2023
K78 DPGG :~/pro
jects/t
> OUTUNE source > W SL: Ubunt > TIME UNE Processing epoch Be :
There was a capital L here.
Ln46. Col31 Spaces4 UTF-8 LF Python 3. 10. 6 (transf
22/3638[00:11<34:30, 1. 75it/s, 1oss=8. 906]
ENG
9:29 PM
> OUT U NE
> WSL: Ubunt > TIME UNE Processing epoc h ee:
Ln&. Col15 Spaces:4 UTF-8 LFPython3. 10. 6 Ctrans
24/3638[00:13<48:00, 1. 51it/s, 1oss=8. 767]
ENG
5/22/2023
9:30 PM
K78 DPGG:~/pn
/kira/
Using device 还有在数据集中
_books /en-it/1. 0. 0/e8f95ea4f32dc39b7f9088908216cd2d7e21ac35f 893d84d39eb594746af2daf )
> OUT UNE length of > WSL: Ubunt
> TIMELNE
ENG
9:30 PM
ects/t 我忘了在这里保存它, 这里我也写成了大写, 所以是
kira
> OUTUIN E
WSL: Ubunb
> TI MEL INE
and also in the dataseti forgot to save it here and here l had it also written 1. 38it/s, 1oss=8. 57e
ENG
9:30 PM
ects/t 大写的现在训练正在进行中你可以看到训练相当
> OUTUINE
WSL: Ubunt
> TIMELINE 97it/s, 1oss=8. 024] 办 ENG
9:30 PM
t ra
PGG:~/pr
cts/ti
> OUTLINE 快, 至少在我的电脑上是这样. 实际上并不那么快2
WSL: Ubunt
> TIME UNE training is
Ln 55, Col 1
58/3638[00:28<38:18,
1. 57it/s, 1oss=7. 755
Python 3. 10. 6 (tra
ENG
9:30 PM
5/22/2023
tra
PGG :~/pro
ject s/t
-scratch ;/usr /bin /env /
> OUTLINE 快, 至少在我的电脑上是这样. 实际上并不那么快但
WSL: Ubunt
> TIMEUNE
T53it/s, loss=7. 571
ENG
5/22/2023
9:30 PM
tra-scratch ;/usr /bin /env /
> OUT UINE 由于我选择了8的批量大小我可以尝试增加它
WSL: Ubunt
> TIMELINE
2. 95it/s, 1oss=7. 862
ENG
9:30 PM
5/22/2023
(transform e
jects/tr
e/kira /.
Using cac
ooks/en-it/1. 0. 0/e8f958a4f32dc39b7f9888
988216cd2d7e21ac35f893d84d39eb594746 af2daf)
> OUTUINE
lax
I
WSL: Ubunt
> TIMELNE
ENG
9:30 PM
5/22/2023
cuda
> OUTUINE WSL: Ubunt
> TIMELNE
Processing epoch 00:
2%
Ln55, Col1 Spaces:4 UTF-8 LF Python 3. 10. 6(transf
71/3638[00:40<29:17, 2. 03it/s, 1oss=7. 288] 通
(tran
PGG:~/pr
ects/ti
ansformer-from-scratch ;/usr /bin /en v/h
kira 损失在减少权重将保存在这里
> OUTUINE
/e8f958a4f32dc39b7f9888988216cd2d7e2 1ac35f893d84d39eb594746a f2daf)
WSL: Ubunt
> TIMEUNE
73/3638[00:41<29:24, 2. 02it/s, 1oss=7. 325]
ENG
3
01:03
> OUTUINE
01:03
01:64
WSL: Ubun b > TIMELINE Processing rocessing Ln55. Col1
[01:09<39:58, 1. 46it/s, 1oss=7. 143]
UTF-8 LFPython 3. 10. 6(trar
9:30 PM
5/22/2023
所以如果我们到达epoch 的末尾, 它将在这里创建第一个
> OUTUINE
WSL: Ubunb
> TIMELNE
Soif we reach the end of the epoch, it will create the first weight here.
2. 46it/s, 1oss=7. 143
5/22/2023
9:30 PM
01:02
01:03
91:02
01:03
01:03 权重.
01:03
> OUT UI NE
WSL: Ubunb > TIMELINE So if we reach the end of the epoch, it will create the first. weight here. to n
0:17, 1. 45it/s, 1oss=7. 115]
2. 10. 6 (ta
9:31 PM
> OUTUINE 所以让我们等到
epoch结束, 看看权重是否真的创建了
WSL: Ubunb
> TIMELINE So let'swait until the end of the epoch and see if the weight is actual s, 1oss=6. 858]
p
ENG
5/22/2023
9:31 PM
3
01:03
01:03
01:03
> OUTUINE
53
01:03
01:04
WSL: Ubunt
> TIMELNE
Processing Ln 55. Col1
[01:18<23:35, 2. 47it/s, 1oss=6. 733]
( Python 3. 10. 6 (trar
9:31 PM
5/22/2023
53
max_len_src=0
55
max_len_tgt=
> OUT UI NE 在实际完成模型训练之前, 让我们再做一件事
WSL : Ubuntu > TIME LNE Before actually finishing the training of the model, let's do another thing.
4. 10. 6(tran
6:41 PM 口
ENG
5/23/2023
54
55
max_len_tgt =θ
56
57
for item in ds_raw :
src _ids = tokenizer _src. encode (item [translation'][config ['lang _src']]). ids
> OUTUINE
58
tgt_ids= tokenizer _src. encode (item ['translation ’][config ['lang _tgt']]). ids > WSL : Ubuntu > TIMELINE 59
max_len_src = max(max_len_src, len(src_ids))
Ln24. Col40 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :conda
ENG
6:41 PM 口
5/23/2023
53
max_len_src=0
55
54
nax_len_tgt= 我们还希望在训练过程中可视化模型的输出.
> OUTUI NE
WSL : Ubuntu > TIMELINE We also would like to Visualize the output of the model while we are training. otas
6:41 PM
5/23/2023
54
55
max_len_tgt =0
56
57
for item in ds_raw :
src _ids = tokenizer _src. encode (item ['translation'][config ['lang_snc']]). ids
> OUTUNE
58
tgt_ids= tokenizer _src. encode (item ['translation'][config ['lang _tgt']]). ids
> WSL: Ubuntu
> TIMELNE
59
max_len_src = max (max_len_src, len(src_ids))
Ln 24. Col40 Spaces 4 UTF-8 LF { Python3. 10. 6 Ctrans fo
ENG
6:41 PM
5/23/2023
54
55
max_len_tgt =0
56
57
for item in ds_raw: 这被称为验证
> OUTUNE
58
src_ids = tokenizer _src. enc > WSL : Ubuntu > TIME LNE
59
max_len_src = max(max_le
And this is called validation.
Ln 24. Col40 Spaces4 UTF-8 LFPython 3. 10. 6 Ctrans fc
ENG
6:41 PM
5/23/2023
53
max_len_src=0
nax_len_tgt=
> OU TUNE 所以我们想看看我们的模型在训练过程中是如何演变的.
> WSL : Ubuntu > TIME LNE So we want'to check how our model is evolving while it is. getting trained.
4. 10. 6 Ctrar
6:41 PM
5/23/2023
53
max_len_s rc =0
54
55
max_len_tgt =θ
56
57
for item in ds_raw:
src_ids = tokenizer _src. encode (item ['translation'][config ['lang _src']]). ids
> OUTUINE
58
tgt_ids= tokenizer _src. encode (item [*translation'][config [‘lang _tgt']]). ids > WSL : Ubuntu > TIMELINE 59
max_len_src= max(max_len_src, len(src_ids))
Ln24, Col40 Spaces 4 UTF-8 LFPython3. 10. 6(transformer :con
> ENG
6:41 PM
5/23/2023
53
max_len_src =0
nax_len_tgt = 所以我们想要构建的是一个验证循环, 它将允许我们评估
> OUTUI NE
WSL : Ubuntu > TIMELINE so what we want'to build is a validation loop which will allow us to evaluate the Z
ENG
5/23/2023
6:41 PM
54
max_len_tgt=
> OUTUINE 句子, 看看它们是如何被翻译的:所以让我们开始构建验证
> WSL: Ubuntu > TIMELINE =max (max_len_src, len (
"sample sentences Ln 24, Col 40
UTF-8 LFPython 6:41 PM
5/23/2023
54
55
max_len_tgt =0
56
for item in ds_raw:
> OUTUINE
src_ids = tokeni
izer_src. enc
g_src']1). ids
> WSL: Ubuntu > TIME UNE and see how they get translated so let's start building the validation loop the. first ENG
6:41 PM
5/23/2023
53
max_len_src=0
max_len_tgt= 我们做的第一件事是构建个名为run Validation 的新方法, 这个方法
> OUT UI NE
> WSL : Ubuntu > TIME LNE and see how they get translated so let's start building the validation loop the. first.
5/23/2023
6:41 PM
将接受一些我们将使用的参数. 现在, 我只是把它们都写
> OUT UI NE
> WSL Ubuntu > TIMELINE src _ids =tokenizer _src. encode (item ['translation'][config ['lang
src'11). ids
Ln22, Col8
Pytho
6:41 PM
5/23/2023
54
55
nax_len_src
56 下来, 稍后我会解释它们将如何使用.
> OUTUNE
> WSL: Ubuntu > TIME LNE
10
and this method will accept some parameters that we will use for now I just. write. all 5/23/2023
6:42 PM
54
55
nax_len_src
56
nax_len 下来, 稍后我会解释它们将如何使用.
> OUTUNE
WSL: Ubuntu
> TIMELNE
10
of them and later T explain how they will be used s
ces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:42 PM
54
55
max_len_src=θ
56
57
max_len_tgt =0
> OUTUNE
58
for item in ds_raw:
> WSL: Ubuntu
> TIMELNE
1 A0
59
src_ids = tokenizer _src. encode (item ['translation'][config ['lang _src']]). i ds
Ln 22, Col20 Spaces:4 UTF-8 LF {↓ Python 3. 10. 6(trans
ENG
5/23/2023
6:42 PM
54
val_ds = Bilingual Dataset (val _ds _raw,
tokenizer _tgt, config ['lang _src'], config l'lang_tgt'], configl'seq_len'])
> OUTUNE 好的, 我们运行验证的第一件事是将我们的模型设置为评估
> WSL: Ubuntu
> TIMEUNE
1 A0
Okay, the first thing we do to run the validation is we put our model into evaluation 5/23/2023
6:42 PM
54
55
val_ds = Bilingual Dataset (val _ds _raw, tokenizer _s
kenizer_tgt, config [‘lang _src'], config["lang_tgt ‘], config ['seq _len′])
56
57
max_len_src= 模式.
> OUTUNE
58
max_len_tgt=θ
> WSL: Ubuntu
> TIMEUNE
1 A0
59
for item in ds _raw :
mode.
Ln23, Col5 Spaces:4 UTF-8 UFPython 3. 10. 6(trans
ENG
6:42 PM 口
5/23/2023
val _ds = Bilingual Dataset (val _ds _raw, tokenizer _src,
okenizer_tgt, config [lang _src "], config ['lang _tgt "], config ['seq _len'])
> OUT UNE 所以我们执行model. eval, 这意味着这告诉 Py To rch我们将要评估
> WSL: Ubuntu
> TIMEUNE
1 A0
So we do model. eval and this means that this tell s Py Torch that we are going to 口
5/23/2023
5:42 PM
54
55
val_ds = Bilingual Dataset (val _ds _raw, tokenizer config ['lang _src'], config ['lang_tgt'], config[‘seq_len*])
56
max_len_src=0 我们的模型
> OUTUNE
max_len_tgt=θ
> WSL: Ubuntu > TIME UNE So we do model. eval and this means that this tell s Py Torch. that we are going to 6:42 PM
ENG
5/23/2023
54
55
train_ds = Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config [lang _src'], config [lang_tgt'], config['seq_len])
val_ds = Bilingual Dataset (val _ds _raw, tokenizer _src, tokenizer _tgt, config [lang _src'], config [‘lang_tgt'], config['seq_len'])
56
57
max_len_src=θ
> OUTUNE
58
max_len_tgt=e
> WSL: Ubuntu > TIME UNE 59
Ln24. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(trans
6:42 PM
ENG
5/23/2023
53
train_ds
= Biling
nfig['seq_len'])
eq_len']) 然后我们将推理两个句子, 看看模型的输出是什么.
> OUT U NE
> WSL : Ubuntu > TIME UNE And then what we will do, we will inference two sentences and see what is the output 口
5/23/2023
6:42 PM
54
train_ds
= Bilingual Dataset (train _ds_r
nfig['seq_l en'])
eq_len']) 然后我们将推理两个句子, 看看模型的输出是什么.
> OUTUNE
> WSL: Ubuntu > TIMELINE of the model.
Ln 24, Col5 S
Pytho
6:43 PM 口
5/23/2023
54
55
train_ds = Bilingual Dataset (train _ds _raw, tokenizer _src, tokenizer _tgt, config [‘lang _src'], config [‘lang_tgt'], config[‘seq_len])
val_ds = Bilingual Dataset (val _ds _raw, tokenizer _src, tokenizer _tgt, config [lang _src'], config [‘lang _tgt'], config['seq_len‘])
56
57
max_len_src=θ
> OUTUNE
58
max_len_tgt =θ
> WSL: Ubuntu > TIMELINE 59
Ln 24, Col5 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
S
ENG
5/23/2023
6:43 PM
53
ds_raw=load_dataset('opus_books', f'{config ["lang _src "]}-{config ["lang_tgt"]}’, split='train')
54
Build tokenizer > OUT UNE 所以通过torch. No Grad我们禁用了在这个宽度块内运行的每个张量
> WSL: Ubuntu
> TIMEUNE
10
So with torch. Not Grad we are disabling the gradient calculation for every tensor that 5/23/2023
6:44 PM
53
ds_raw =load_dataset('opus_books', f'{config ["lang _src "]}-{config ["lang_tgt"]}’, split='train')
54
> OUTUNE 所以通过torch No Grad我们禁用了在这个宽度块内运行的每个张量
> WSL: Ubuntu
> TIMEUNE
10
Will run inside this width block.
Ln34. Col9 S
Pytho
4. 10. 6(t
5:44 PM 口
5/23/2023
54
55
# Build tokenizer s
56
57
tokenizer_src =get _or _build _tokenizer (
tokenizer _tgt =get _or _build _tokenizer 的梯度计算.
> OUTUNE
58
> WSL: Ubuntu > TIME UNE
1 A0
59
# Keep 9e% for training Will'run in'side this width block.
Ln34. Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
6:44 PM 口
5/23/2023
55
54
# Build tokenizer s
56
57
tokenizer_tgt =get _or _build _tokenizer (config, ds _raw, config ['lang _tgt'])
tokenizer _src =get _or _build _tokenizer (config, ds _raw, config [lang_src'])
> OUTUNE
58
> WSL: Ubuntu > TIME UNE
1 A0
59
# Keep 90% for training and 1e% for validation Ln34, Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:44 PM
55
54
# Build tokenizer s
56
57
tokenizer_src =get _or _build tokenizer _tgt = get_or_build 而这正是我们想要的.
> OUTUNE
58
> WSL: Ubuntu > TIME UNE
1 A0
59
# Keep 9e% for train in And this is exactly what we want.
Ln34, Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:44 PM
53
ds_raw =load_dataset('opus_books', f'{config ["lang _src "]}-{config ["lang _tgt "]}’, split ='train') 我们只是想从模型中推理, 我们不希望在这个循环中训练它.
> OUT UNE
> WSL: Ubuntu10
> TIMEUNE
We just want to Tn ference from the model, we don't want to train. it during this loop.
5/23/2023
6:44 PM
54
55
# Build tokenizer s
56
57
tokenizer_tgt =get _or _build _tokenizer (config, ds _raw, config [lang _tgt'])
tokenizer _src = get _or _build _tokenizer (config, ds _raw, config ['lang_src'])
> OUTUNE
58
WSL: Ubuntu > TIME UNE
1 A0
59
# Keep 90% for training and 1e% for validation Ln34, Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
6:44 PM 口
5/23/2023
54
55 所以让我们从验证数据集中获取一个批次.
> OUTUNE
WSL: Ubuntu30
> TIME UNE So let's get a batch from the validation dataset. s
UTF-8 LF{ Python 3. 10. 6(trans
ENG
6:44 PM 口
5/23/2023
54
55
# Build tokenizer s
56
57
tokenizer_tgt =get _or _build _tokenizer (config, ds _raw, config [lang _tgt'])
tokenizer _src = get _or _build _tokenizer (config, ds _raw, config [‘lang _src'])
> OUTUNE
58
> WSL: Ubuntu
> TIMELNE
59
# Keep 90% for training and 1e% for validation Ln 34. Col35 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
ENG
5/23/2023
6:44 PM
def ds _raw =load _dataset ('opus _books', f'{config ["lar
c"]}-(config ["1
g_tgt"])', split='train') 因为我们只想推理两个所以我们记录我们已经处理了多少
> OUTUNE
> WSL: Ubuntu > TIME L NE
ibecause we want to inference only two so we keep a count of how many we have. already 5/23/2023
6:44 PM
55
54
def
get_ds(config):
ds_raw data set (
56
Build 并从这个当前批次中获取输入.
> OUTUNE
> WSL: Ubuntu > TIME UNE processed and we get the input from this current batch I want to remind you that for.
ENG
5/23/2023
6:44 PM
get _ds (config ):
> OUT UNE 我想提醒你, 对于验证ds, 我们只有一个批次大小. 这是
WSL: Ubuntu00
> TIMELNE
processed and we get the input from this current batch I want to remind you. that for.
ENG
5/23/2023
6:44 PM
get _ds (config ):
> OUTUNE 我想提醒你, 对于验证ds, 我们只有一个批次大小. 这是
WSL: Ubuntu
> TIMELNE
tokenizer _tgt get _or _build _tokenizer (config, ds_raw
n36. Col24 Spaces:4 UTF-8 LFPython
3. 10. 6(
6:44 PM
5/23/2023
get _ds (config ):
, f'(config ["lang _src "]}-(config ["
> OUTUNE 我想提醒你, 对于验证ds, 我们只有一个批次大小. 这是
> WSL: Ubuntu
> TIMELNE
validation ds we only have a batch size of one this is the encoder input and. we can 口
ENG
5/23/2023
6:45 PM
53
55
54
get_ds(config):
plit='train')
> OUTUNE
56 编码器输入, 我们也可以获取编码器掩码.
> WSL: Ubuntu
> TIMELNE
validation ds we only have a batch size of one this is the encoder input and. we can ENG
5/23/2023
6:45 PM
55
ds(config):
> OUTUNE
5 编码器输入,, 我们也可以获取编码器掩码.
> WSL: Ubuntu > TIME UNE
4. 10. 6(trar
Ln 37. Col 13
UTF-8 LFPython
ENG
6:45 PM
Z
5/23/2023
54
55
def
get_ds(config):
56
57
ds_raw =load _dataset (‘opus _books', f'{config ["lang _src "]}-{config ["lang _tgt"]}’, split='train′)
> OUTUNE
58
# Build tokenizer s
WSL : Ubuntu > TIME UNE tokenizer _src = get _or _build _tokenizer (config, ds _raw, config ['lang_src'])
Ln37. Col26 Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
6:45 PM 口
5/23/2023
53
else:
54
Tokenizer. from _file (str (tokenizer _path )) 让我们确认一下批次的实际大小确实是一个. 现在让我们进入
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
let'sjust verify that the size of the batch is actually one and now let's go to the.
6:45 PM
5/23/2023
54
else:
tokenizer = Tokenizer. from _file(str(tokenizer_path))
55
56
return tokenizer > OUT UNE get _ds (config ): 有趣的部分
> TIME LNE
58
tgt"]}', split='train')
WSL: Ubuntu let'sjust verify that the size of the batch is actually one and now let's go to. the. can
ENG
5/23/2023
6:45 PM
54
else:
tokenizer = Tokenizer. from _file(str(tokenizer_path))
55
return tokenizer > OUTUNE
57
def
get_ds(config ): 有趣的部分
> TIME LNE
58
tgt"]}’, split='train')
WSL: Ubuntu interesting part so as you remember when. we Spaces 4 UTF-8 LFPython3. 10. 6(trans fo
ENG
5/23/2023
6:45 PM
53 tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
tokenizer. save (str (tokenizer _path )) 所以, 如你所记得的, 当我们想要推理模型时, 我们需要
> OUT U NE
> WSL : Ubuntu > TIMELINE ef get _ds(con
3. 10. 6 (tr
6:45 PM
ENG
5/23/2023
tokenizer. save (str (tokenizer _path )) 所以, 如你所记得的, 当我们想要推理模型时, 我们需要
> OUT U NE
WSL: Ubuntu
> TIMELNE
lef get_ds(config ):
4. Col13 S
es4 UTF-8 LFPython 3. 10. 6 (tr
6:45 PM
5/23/2023
53 tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
tokenizer. save (str (tokenizer _path )) 只计算一次编码器输出, 并将其重用于我们将要处理的每个token.
> OUT U NE
> WSL Ubuntu > TIME LNE calculate the when we want to inference the model we need to calculate the encoder ENG
5/23/2023
6:45 PM
53 tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
54 tokenizer. save (str (tokenizer _path )) 只计算一次编码器输出, 并将其重用于我们将要处理的每个token.
> OUTUNE
WSL: Ubuntu00
> TIMEUNE
output only once and reuse it for every token that we will the model will output from ENG
5/23/2023
6:45 PM
53 tokenizer. train _from _iterator (get _all _sentences (ds, lang), traine
54
lse
tokenizer. save (str (tokenizer _path )) 模型将从解码器输出:所以让我们创建另一个函数, 它将在
> OUT U NE
WSL: Ubuntu
> TIMELNE
lef get_ds(config ):
the
n41, Col 13
(. Pytho
5:46 PM
5/23/2023
53 tokenizer. train _from _iterator (get _all _sentences (d s, lang), traine
r=trainer lse tokenizer. save (str (tokenizer _path )) 模型将从解码器输出:所以让我们创建另一个函数, 它将在
> OUT U NE
WSL : Ubuntu > TIME LNE decoder so let'screate another function that will run the greedy decoding. on our 口
ENG
5/23/2023
5:46 PM
53 tokenizer. train _from _iterator (get _all _sentences (ds, lang), trainer
r=trainer)
54
tokenizer. save (str (tokenizer _path ))
> OUT U NE 我们的模型上运行贪婪解码, 我们将看到它只会运行一次
WSL : Ubuntu > TIMELINE model and we'll use and we will see that it will run the encoder only once so let's
5/23/2023
6:46 PM
53
54
okenizer. train _frc
"[sos]","[ Eos]"], min_frequency =2)
> OUTUNE 编码器. 所以让我们称这个函数为greedy_decode.
> WSL: Ubuntu > TIME LNE model and we'll use and we will see that it will run the encoder only once so let's.
5/23/2023
6:46 PM
53
54
trainer = Word Level Trainer (special _token s=["[u Nk]"
okenizer. train _frc
itera
"[sos]","[ Eos]"], min_frequency =2)
> OUTUNE 编码器. 所以让我们称这个函数为greedy_decode.
> WSL: Ubuntu > TIME LN E
urn tokenizer call this function greedy decode Ln22, Col1 Spaces 4
Pytho
. 10. 6
6:46 PM
5/23/2023
53 tokenizer. pre _tokenizer = Whitespace ()
55
54
trainer = Word Level Trainer (special _token s=["[u NK],"[ PAD]","[sos]","[ EOs]"], min_frequency=2)
tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
56
57
tokenizer. save (str(tokenizer_path))
> OUTUNE
58
else :
tokenizer = Tokenizer. from _file (str (tokenizer _path ))
WSL : Ubuntu > TIME LNE return tokenizer Ln22, Col2 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:46 PM
55
54
tokenizer. pre _tokenizer = Whitespace ()
rair
["[ UNK 好的, 让我们创建一些我们将需要的token.
> OUTUNE
> WSL: Ubuntu > TIMELINE 1 A0
Okay, let'screate some tokens that we will need.
UTF-8 LFPython3. 10. 6 Ctrans
ENG
6:46 PM 口
5/23/2023
54 tokenizer. pre _tokenizer = Whitespace ()
pecial _t 所以 Sostoken 即句子的开始, 我们可以从任一tokenizer 中
> OUT UNE
> WSL: Ubuntu10
> TIMEUNE
So the Sos token, which is the start of sentence, we can get it from either 5:46 PM
54 tokenizer. pre _tokenizer = Whitespace ()
trainer pecial _t 所以 Sostoken 即句子的开始, 我们可以从任一tokenizer 中
> OUT U NE
WSL : Ubuntu > TIMELINE tokenizer, doesn't matter if it's the target or the source, they both have it. so tars
5/23/2023
6:46 PM
55
54
trainer = Word Level Trainer (special _token s=["[u NK]","[ PAD]","[ Sos]","[ Eos]"], min_frequency=2)
tokenizer. pre _tokenizer = Whitespace ()
56
57
tokenizer. save (str (tokenizer _path ))
tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer a trainer )
> OUTUNE
58
else:
WSL: Ubuntu > TIMELINE 59 tokenizer = Tokenizer. from _file(str(tokenizer_path))
Ln 23, Col28 Spaces 4 UTF-8 LF ( Python3. 10. 6(trans fc
ENG
6:46 PM 口
5/23/2023
45
> OUTUINE
eos好的然后我们做的是我们预先计算编码器输出, 并将
> WSL: Ubuntu > TIMELINE eos okay and then we what we do is we pre-compute the encoder output and reuse. it for Z
5/23/2023
6:47 PM
assert encoder _input. size (e)== 1," Batch size
> OUTUNE
eos好的, 然后我们做的是我们预先计算编码器输出, 并将
WSL: Ubuntu > TIME LNE
42. 10. 6(t
6:47 PM
5/23/2023
45 assert encoder _input. size (e)== 1," Batch size
validatio
> OUTUNE 其重用于我们从解码器获得的每个token. 月 所以我们只给出源和源
> WSL : Ubuntu > TIME LNE
5/23/2023
6:47 PM
45
44
encoder
=batch['enc
_mask']. to (device )
> OUTUNE
47 掩码, 即编码器输入和编码器掩码,
> WSL: Ubuntu > TIMELINE 3. 10. 6(tra
ENG
5/23/2023
6:47 PM
44
45
encoder_n
mask']. to(device )
47
46 掩码, 即编码器输入和编码器掩码.
> OUTUNE
> WSL: Ubuntu > TIMELINE We just give the source and the source mask, which is the encoder input and. the ENG
5/23/2023
6:47 PM
44
count+=1
encoder _input =batch ['encoder _input']. to (device )
45
47
46 掩码, 即编码器输人和编码器掩码.
> OUTUNE
> WSL: Ubuntu > TIMELINE encoder mask.
Ln 22, Col45 (11 selected )
Spaces 4 UTF-8 LFPython 3. 10. 6(ta
ENG
6:47 PM
5/23/2023
44 encoder _input = batch ['encoder _input']. to (device )
count +=1
45
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (o)== 1," Batch size must be 1 for validation "
> WSL : Ubuntu > TIMELINE Ln28. Col5 Spaces4 UTF-8 LFPython 3. 10. 6(trans 口
ENG
5/23/2023
6:47 PM
43
count+=1
encoder _input = batch ['encoder _input']. to (device )
49 encoder _mask =batch ['encoder _mask']. to (device )
> OUT U NE 我们也可以称之为编码器输人和编码器掩码.
> WSL : Ubuntu > TIMELINE We can also call it encoder input and encoder mask. s urs u 4 ython
4. 10. 6(tr
ENG
6:47 PM
5/23/2023
44
45
encoder _input =batch ['encoder _input']. to (device )
encoder _mask = batch ['encoder _mask']. to (device )
46
> OUTUNE
47
48
assert encoder _input. size (0)== 1," Batch size must be 1 for validation "
> WSL : Ubuntu > TIME UNE 49
Ln 27, Col28 Spaces:4 UTF-8 LF↓ Python 3. 10. 6(trans
Z
ENG
5/23/2023
6:47 PM
44
45
encoder _input =batch['enc
encoder_mask =batch ['encoder _mask']. to (device )
oder _input']. to (device )
> OUT UNE
47
48 然后我们:好的, 我们如何进行推理?
> WSL: Ubuntu
> TIMEUNE
then we get the then we okay how do we do the inferencing the first thing we do is we 口
5/23/2023
6:47 PM
44
45
encoder _input =batch ['encoder _input']. to (device )
> OUT UNE 我们做的第一件事是给解码器句子的开始token, 以便解码器将
> TIME LNE
5/23/2023
6:47 PM
44
45
encoder _input =batch ['encoder _input']. to (device )
> OUT UNE 我们做的第一件事是给解码器句子的开始to ken, L 以便解码器将
WSL: Ubuntu00
> TIMELNE
give to the decoder the start of sentence token so that the decoder will output the ENG
5/23/2023
6:47 PM
44
45
encoder _input =batch ['encoder _input']. to (device )
> OUT UNE 我们做的第一件事是给解码器句子的开始token, 以便解码器将
> WSL : Ubuntu > TIME UNE
first
Ln28, Col 5
5:48 PM 口
44
45
encoder _input = batch ['encoder _input ’]. to (device )
46 输出翻译句子的第一个token.
> OUTUNE
47
48
asserte
> WSL : Ubuntu > TIME UNE token of the sentence of the translated sentence then at every iteration just like we
5/23/2023
6:48 PM
44 encoder _input =batch ['en encoder _mask = batch ['encoder _m
oder_in p
ask']. to (device )
ut']. to (device )
> OUTUNE 然后在每次迭代中,? 就像我在幻灯片中展示的那样, 每次
WSL: Ubuntu00
> TIMEUNE
token of the sentence of the translated sentence then at every iteration just like we
5/23/2023
6:48 PM
encoder _input =batch ['en sk']. to (device )
ut']. to (device )
> OUT U NE 然后在每次迭代中, 就像我在幻灯片中展示的那样, 每次
WSL : Ubuntu > TIME UNE saw in my slides at every iteration we add the previous token to. the python 3. 10. 6(tr
6:48 PM
5/23/2023
45 encoder _input =batch['enc
oder_input']. to (device )
> OUT UNE 选代我们都将前个token 添加到解码器输入中, 以便解码器
WSL : Ubuntu > TIME UNE saw in my slides at every iteration we add the previous token to. the python 5/23/2023
5:48 PM
45
44
encoder _in put =batch['en
oder_in
ut']. to (device ) 迭代我们都将前个token 添加到解码器输入中, 以便解码器
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
5/23/2023
5:48 PM
45
44
encoder _input =batch['enc
oder_input']. to (device ) 迭代我们都将前个token 添加到解码器输入中, 以便解码器
> OUT UNE > TIME UNE to the decoder input and so that the decoder can output the next to ken
5/23/2023
5:48 PM
44
count +=1
encoder _input =batch ['encoder _input']. to (device )
45
46
> OUTUNE
47
assert encoder _i 可以输出下一个token.
WSL : Ubuntu > TIME LNE to the decoder input and so that the decoder can output the next token.
4. 10. 6(trans
ENG
5/23/2023
6:48 PM
44
count +=1
encoder _input =batch ['encoder _input']. to (device )
45
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (0)== 1," Batch size must be 1 for validation "
WSL : Ubuntu > TIME LN E
49
Ln28. Col5 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
ENG
6:48 PM
5/23/2023
44
count +=1
encoder _input =batch['enc
oder_input']. to (device )
45
encoder_mas
> OUTUNE 然后我们取下=个-token;再次将其放在解码器输入的前面, 并
> WSL : Ubuntu > TIME LNE Then we take the next token, we put it again in front of the input to. the decoder and. 口
5/23/2023
5:48 PM
45
44 然后我们取下一个-token;°再次将其放在解码器输入的前面, 并
> OUTUNE
WSL: Ubuntu > TIME LNE we get the successive to ken.
Pytho
5:48 PM
5/23/2023
44
45
encoder _input =batch ['encoder _input']. to (device )
46
> OUTUNE
47
48
assert encoder _input. s 获取后续的token.
WSL : Ubuntu > TIME LNE we get the successive to ken.
Ln28, Col 5 Spaces4 UTF-8 LFPython 3. 10. 6(trans 口
ENG
5/23/2023
6:48 PM
44
count+=1
encoder _input =batch ['encoder _input']. to (device )
45
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (0) =m 1," Batch size must be 1 for validation "
WSL : Ubuntu > TIME LNE
Ln28. Col SSpaces4 UTF-8 LF Python 3. 10. 6(transfo
ENG
5/23/2023
6:48 PM
encoder _input =batch ['en oder _in ask `]. to (device )
ut `]. to (device )
> OUT U NE 所以让我们为第一次送代构建一个解码器输入, 它只包含句子
WSL : Ubuntu > TIMELINE So let'sbuild a decoder input for the first iteration, which is only the start. of 5/23/2023
5:48 PM
45
encoder _mask = batch['encod
encoder_input =batch ['enc code r_in
der_mask']. to (device )
ut `]. to (device )
46 的开始tokento
> OUTUNE
47
assert encoder _input. size (0)
WSL: Ubuntu > TIMELINE So let'sbuild a decoder input for the first iteration, which is only the start. of ost
ENG
5/23/2023
6:48 PM
44 encoder _input =batch ['encoder _input']. to (device )
count +=1
45
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (0)== 1," Batch size must be 1 for validation "
> WSL : Ubuntu > TIMELINE 45
Ln 28. Col5 Spaces4 UTF-8 LF↓ Python 3. 10. 6(trans
ENG
6:48 PM 口
5/23/2023
44
45
count+=1
encoder _input =batch ['encoder _input']. to (device )
47
46 我们用句子的开始token填充这个.
> OUTUNE
> WSL: Ubuntu > TIME LNE
00 Q1fleto analyze We fill this one with the start of sentence token. spaces 4 urs u ython 3. 1o6(tran
ENG
6:48 PM
5/23/2023
44
45
encoder _input = batch ['encoder _input']. to (device )
count +=1
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (e)1," Batch size must be 1for validation > WSL : Ubuntu > TIME LNE 49
Ln 29. Col51 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
6:48 PM
ENG
5/23/2023
44
45
encoder _input = batch ['encoder _input']. to (device )
count +=1
46
encoder 它与编码器输入具有相同类型
> OUTUNE
47
48
WSL: Ubuntu > TIME LNE And it has the same type as the encoder input.
Spaces 4 UIF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:49 PM
44
45
encoder _input =batch ['encoder _input']. to (device )
count +=1
46
encoder _mask = batch ['encoder _mask']. to (device )
> OUTUNE
47
48
assert encoder _input. size (e) 1," Batch size must be 1 for validation WSL : Ubuntu > TIMELINE 49
Ln 29, Col70 Spaces4 UTF-8 LF{ Python 3. 10. 6(trans
ENG
5/23/2023
6:49 PM
45 For batch in validation_ds:
count +=1 现在, 我们将不断要求解码器输出下一个token, 直到我们达到
> OUTUNE
> WSL: Ubuntu > TIMELINE Now, we will keep asking the decoder to output the next token until we reach either 5/23/2023
5:49 PM
44
45
for batch in validation_ds:
count+=1
46
encoder _mask = batch ['encoder _mask']. to (device )
encoder _input = batch ['encoder _input']. to (device )
> OUTUNE
47
48
> TIMELNE
49
assert encoder _input. size (e) == 1," Batch size must be 1 for validation > WSL: Ubuntu
Ln22, Col 84(7 selected ) Spaces 4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:49 PM
43
with torch. no_grad():
For batch in validation_ds:
45
count+=1
47
46 所以我们可以做个while True 循环.
> OUT UNE
48
e(e)
> WSL: Ubuntu
> TIMEUNE
So we can do a while true.
In 30, Col 11 S
UTF-8 LFPytho
4. 10. 6(tr
6:49 PM
ENG
5/23/2023
44
45
with torch. no_grad():
for batch in validation_ds:
46
encoder _input = batch ['encoder _input']. to (device )
count +=1
> OUTUNE
47
48
encoder _mask =batch ['encoder _mask']. to (device )
WSL : Ubuntu > TIME UNE 49
encoder innut size(e)mn1" Batch size Ln31. Col9 Spaces4 UTF-8 LF Python 3. 10. 6(trans
6:49 PM
ENG
5/23/2023
vith torch.
grad ():
> OUT U NE 然后我们的第个停止条件是, 如果解码器输出, 即下一步
> WSL : Ubuntu > TIME UNE And then our first stopping condition is if the decoder output, which becomes the 5/23/2023
6:49 PM
vith torch.
grad ():
validation _ds :
> OUT U NE 然后我们的第个停止条件是, 如果解码器输出, 目 即下一步
WSL : Ubuntu > TIMELINE input of the next step, becomes larger than maxlength or reaches maxlength. som
5/23/2023
6:49 PM
45
with torch. no_grad():
for batch in validation_ds:
46
count 的输人大于或达到最大长度.
> OUTUNE
47
encoder > WSL: Ubuntu
> TIMEUNE
10
input of the next step, becomes larger than maxlength or reaches maxlength. c
Z
ENG
5/23/2023
6:49 PM
44
45
with torch. no_grad():
for batch in validation_ds:
46
count+=1
encoder _input =batch ['encoder _input']. to (device )
> OUTUNE
47
48
encoder _mask =batch ['encoder _mask']. to (device )
> WSL : Ubuntu > TIME UNE 49
Ln31, Col9 Spaces4 UTF-8 LFPython 3. 10. 6(trar
ENG
5/23/2023
6:49 PM
44
45
with torch. no_grad():
46
for batch in > OUT UNE
47
48
count+
encoder 这里, 为什么我们有两个维度?
WSL: Ubuntu > TIMELINE 49
ENG
6:49 PM
Z
5/23/2023
45
44
with to rch. no_grad(): 个是用于批次, 另一个是用于解码器输入的token.
> OUTUNE
> WSL: Ubuntu01
> TIMELINE One is for the batch and one is for the tokens of the decoder input. pth on
4. 10. 6(tra
6:49 PM
ENG
5/23/2023
44
45
with torch. no_grad():
46
for batch in validation_ds:
> OUTUNE
47
48
count +=1
encoder _input = batch ['encoder _input']. to (device )
> TIMELINE 49 encoder _mask =batch ['encoder _mask']. to (device )
> WSL: Ubuntu
Ln 29, Col18 (13 selected ) Spaces :4 UIF-8 LF( Python3. 10. 6(trans fc
ENG
5/23/2023
6:49 PM
44
45
# Size of the control window (just use a default value )
console _width =80
46
> OUTUNE
47
48
with torch. 现在我们也需要为此创建一个掩码.
> WSL : Ubuntu > TIME UNE
Z
ENG
5/23/2023
6:49 PM
44
45
# Size of the control window (just use a default value )
console _width = 80
46
with torch. no_grad():
> OUTUINE
47
48
for batch in validation_ds:
> TIMEUNE
49
count+=1
> WSL: Ubuntu Ln 34. Col11 Spaces4 UTF-8 LF { Python 3. 10. 6(trans
ENG
5/23/2023
6:49 PM
expected =[]
predicted =[]
> OUT UNE 我们可以使用我们的函数causal mask 来表示我们不希望输入看到
> WSL : Ubuntu > TIME UNE We can use our function causal mask to say that we don't want the input to watch 5/23/2023
5:50 PM
44
predicted=[]
expected =[]
45
46
# Size of the control win > OUT UNE
47
48
console_width = 80 未来的词.
> WSL: Ubuntu > TIME UNE We can use our function causal mask to say that we don't want the input to watch.
with torch. no_grad():
5/23/2023
6:50 PM
44 predicted =[]
expected =[]
45
46
> OUTUNE
47
48
console _width = 80
> TIMEUNE
49
with torch. no_grad ():
> WSL : Ubuntu Ln35, Col36 Spaces4 UTF-8 LF Python 3. 10. 6(trar
ENG
5/23/2023
6:50 PM
45 predicted =[]
> OUT UNE 我们不需要另一个掩码, 因为如你所见, 这里没有任何填充token.
WSL : Ubuntu > TIME L NE
o And we don't need the other mask because here we don't have any padding token as you 5/23/2023
6:50 PM
43 predicted =[]
expected =[]
45
> OUTUNE 我们不需要另个掩码, 因为如你所见, 这里没有任何填充token.
WSL: Ubuntu > TIME UNE can see.
Ln 35, Col. 66
Pytho
5:50 PM
5/23/2023
44
predicted=[]
expected =[]
45
46
# Size of the control window (just use a default value )
> OUTUNE
47
48
console_width = 80
> TIME UNE
49
with torch. no_grad():
WSL: Ubuntu 3. 10. 6(tra
ENG
5/23/2023
6:50 PM
44
45
expected =[]
source _text s =[]
46
predicted=[]
> OUTUINE
47
48
# Size of the control wind c 现在我们计算输出.
> TIMELINE 49
console_width=8e
> WSL: Ubuntu Now we calculate the output.
Ln37. Col9 Spaces4 UTF-8 LF Python 3. 10. 6(trans
Z
ENG
5/23/2023
6:50 PM
count =θ
> OUT U NE 我们为循环的每一次送代重用编码器的输出. 我们重用源
> WSL Ubuntu > TIMELINE we reuse the output of the encoder for every iteration of the loop we reuse the s
6:50 PM
5/23/2023
44
45
source_texts=[]
46
expected =[] 掩码, 即编码器的输入和掩码.
> OUTUNE
47
48
predicted=[]
> WSLUbuntu > TIME UNE o source mask so the input the mask of the encoder then we give the decoder input and 5/23/2023
6:50 PM
count =θ 然后我们提供解码器输入, 以及它的掩码, 民
> OUT U NE 即解码器掩码
> WSL : Ubuntu > TIME UNE
6:51 PM
5/23/2023
然后我们提供解码器输入, 以
> OUT U NE 以及它的掩码, 即解码器掩码
> WSL : Ubuntu > TIMELINE along with its 5/23/2023
6:51 PM
count =θ
> OUT U NE 然后我们提供解码器输入, 以及它的掩码, 民 即解码器掩码
> WSL : Ubuntu > TIMELINE 5/23/2023
6:51 PM
45
count=θ
46
source _text s = 然后我们得到下一个 token.
> OUTUNE
47
expected =[]
predicted =[]
> WSL : Ubuntu > TIMELINE mask the decoder mask and then we get the next token so we get the probabilities of 5/23/2023
6:51 PM
43
model. eval()
ples=2)
count=θ
> OUTUNE 所以我们使用投影层获取下一个token 的概率, 但我们只想要
> TIME UNE
5/23/2023
6:51 PM
def run _validation (model, validation _ds, tokenizer _s
odel. eval ()
device examples =2):
> OUTUNE 所以我们使用投影层获取下一个token白 的概率, 但我们只想要
WSL: Ubuntu > TIMELINE the next token using the projection layer ucos
5/23/2023
6:51 PM
43
def run_validation (model, validation _ds, tokenize device nsg, global _state,
examples =2)
45
model. eval() 最后一个token的投影, 即我们在编码器之后给出的最后一个
> OUTUNE
WSL: Ubuntu > TIMELINE 01
n 41, Col29
Python 6:51 PM
5/23/2023
43
def
run_validation (model, validation _ds, tokenizer device _examples =2):
45
model. eval()
> OUTUNE 最后一个. token的投影, 即我们在编码器之后给出的最后一个
WSL Ubuntu > TIMELINE but we only want the projection of the last token so the next token after the last we
5/23/2023
6:51 PM
next 得到具有最大概率的 token. 这就是贪婪搜索.
> OUT U NE
WSL Ubuntu > TIME LNE. have given to the encoder now we can use the max so we get the. token with the maximum ond)
5/23/2023
6:51 PM
les=2) 得到具有最大概率的token. 这就是贪婪搜索.
> OUTUNE
> WSLUbuntu > TIME LNE probability this is the greedy search Ln42. Col34 S
( Pyth
10. 6(t
6:51 PM
ENG
5/23/2023
44
[0]prob
46
45
def run_validation (model, validation _ds, tokenizer _src, tokenizer _tgt, max_len, device, print _msg,
model. eval ()
> OUT UNE
47
48
count=θ
> TIMELINE
49
source _texts =[]
WSL : Ubuntu Ln42. Col38 Spaces4 UTF-8 LF { Python 3. 10. 6 (tra
5/23/2023
6:51 PM
43
44
-next _word = to rch. max(prob, dim=1) 然后我们得到这个词, 并将其追加回来, 因为它将成为下
> OUTUINE
> WSL: Ubuntu > TIMELINE and then we get this word and we append it back to this one because it will become 5/23/2023
6:52 PM
44
45
46
> OUTUNE
47
_validation 次送代的输入我们进行连接
obal _state, writer ples=2):
> WSL: Ubuntu
> TIMELNE
and then we get this word and we append it back to this one because it will become.
odel. eval(
6:52 PM
ENG
5/23/2023
43
44
decoder_input F torch. ca 所以我们取解码器输入并追加下个, token, 所以我们创建另一个
> OUTUNE
> WSL: Ubuntu10
> TIMEUNE
the input of the next iteration and we concat so we take the decoder input and we
5/23/2023
6:52 PM
44
decoder _input F to rch. cat([decoder_input, torch. empty(1, 1). type_as(source), fi) 张量, 应该是正确的好的如果下个词或token 等于
> OUT UNE
WSL: Ubuntu
> TIMEUNE
3 A0
Ln 44, Col 85
Pytho
6:52 PM
5/23/2023
43
44
decoder_input F torch. cat ([decoder _input, to rch. empty(1, 1). type, next _word =to rch. max(prob, dim=1)
m()). to(device )]) 句子的结束token 那么我们也停止循环, 这就是我们的贪婪
> OUT UNE el. eval ()
WSL: Ubuntu
> TIMEUNE
( Pytho
5:52 PM
43
decoder _input F torch. cat([decoder_input, torch. empty (1, 1). type
em()). to(device)], dim=1)
45
> OUTUNE 句子的结束token, 那么我们也停止循环, 这就是我们的贪婪
WSL : Ubuntu > TIME UNE should be correct okay if the next token so if the next word or token is equal equal ENG
5/23/2023
6:53 PM
43
next_word=toexit
. fill_(nex
d. item ()). to (device )], dim=1)
Encoding warning 句子的结束token, 那么我们也停止循环, 这就是我们的贪婪
> OUTUNE
WSL: Ubuntu20
> TIMELNE
to the end of sentence token then we also stop the loop and this is our greedy search.
5/23/2023
6:53 PM
45
decoder _input F torch. cat([decoder_input, torch. empty (1, 1). type_as(source). fil1_(next _word. item ()). to (device )], dim=1)
47
46
if next_word == eos_idx:
break 搜索.
> OUTUNE
48
> WSL: Ubuntu
> TIMEUNE
to the end of sentence token then we also stop the loop and this is our greedy search 5/23/2023
6:53 PM
44
45
decoder_input F torch. cat ([decoder _input, torch. empty (1, 1). type_as(source). fil1_(ne
rd. item ()). to (device )], dim =1)
46
fnext_word
> OUTUNE
47
48
break 现在我们可以直接返回输出
WSL : Ubuntu > TIME UNE now we
Ln48. Col5 Spaces4 UTF-8 LFPython 3. 10. 6 Ctra
6:53 PM
5/23/2023
45
decoder _input F to rch. cat([decoder_input, torch. empty(1, 1). type_as(source ). fill _(next rd. item()). to(device)], dim=1)
47
46
if next _word break 现在我们可以直接返回输出
> OUTUNE
48
> WSL: Ubuntu
> TIMEUNE
can just return the output so the output is basically the decoder input because every.
ENG
5/23/2023
6:53 PM
45
idx
> OUTUNE 所以输出基本上就是解码器输入, 因为你每次都将其追加下
> WSLUbuntu > TIME LNE
1
can just return the output so the output is basically the decoder. input because every.
5/23/2023
6:53 PM
44
input
ce). fill_
> OUTUNE 所以输出基本上就是解码器输入, 国 因为你每次都将其追加下
> WSL: Ubuntu > TIME L NE
itime you are appending the next token to it and we remove the batch dimension so we 口
5/23/2023
6:53 PM
f next eos_idx:
> OUTUNE
token,. 并且我们移除了批次维度, 户 所以进行了压缩, 这
> WSL: Ubuntu > TIMELINE i time you are appending the next token to it and we remove the batch dimension so we
5/23/2023
6:53 PM
square > OUT UNE token, 并且我们移除了批次维度,! 所以进行了压缩, 这
> WSL : Ubuntu > TIMELINE squeeze it and 49, Col28
( Pytho
6:53 PM
5/23/2023
44
45
46
if next _word == eos_idx:
47
48
break 就是我们的贪婪解码.
> OUTUNE
49
return decoder _input. squeeze > WSL: Ubuntu
> TIMELNE
00 O1fle to analyze Ln 49. Col36 Spaces4 UTF-8 LFPython 3. 10. 6(trans
ENG
5/23/2023
6:53 PM
60
61
# Size of the control window (just use a default value )
63
62
console _width =8e
> OUTUNE
64
65
with torch. no_grad ():
for batch in validation_ds :
> WSL : Ubuntu > TIME LNE
count+=1
Ln 22. Col 18(13 selected ) Spaces 4 UTF-8 LF ( Python 3. 10. 6(trans
6:53 PM
ENG
5/23/2023
73
> OUTUNE
75
foritem
yiel d 现在我们可以在这个函数中使用它.
WSL: Ubuntu 00
> TIMELNE
ENG
6:53 PM
5/23/2023
80
81
ifnot Path. exists (tokenizer _path ):
tokenizer = Tokenizer ( Word Level (unk_token='[ UNk]‘))
82
83
tokenizer. pre _tokenizer = Whitespace ()
trainer =word Level Trainer (special _token s=["[ UNK]","[ PAD]","[ Sos ]","[ EOS ]"], min _frequency =2)
> OUTUINE
84
85
tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
tokenizer. save (str (tokenizer _path )
WSL: Ubuntu
> TIMELNE
Ln71. Col1 Spaces 4 UTF-8 LF { Python3. 10. 6(trans fo
ENG
6:53 PM
5/23/2023
if not Path. exists (tokenizer _path ):
> OUT U NE 所以在验证函数中, 我们终于可以得到模型输出, 等于贪婪
WSL : Ubuntu > TIME LNE So in the validation function, so we can finally get the model output is equal to. m 口
5/23/2023
6:53 PM
80
79
def
tokenizer _path = Path (config ['tokenizer file']. fc
81
if not Path. exists (tokenizer _path ):
tokenizer = To ker
83
82
tokenizer. pre _to k 解码, 我们给它所有参数
> OUTUNE
84
trainer=
toker
> WSL: Ubuntu > TIMELINE 01 O1fle to anayze
ENG
5/23/2023
6:53 PM
79
80
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang))
81
if not Path. exists (tokenizer _path ):
tokenizer = Tokenizer ( Word Level (unk_token='[ UNk]))
83
82
tokenizer. pre _tokenizer = Whitespace ()
> OUTUNE
84
85
trainer = Word Level Trainer (special _token s=[[ UNK]","[ PAD]","[ SOs]","[ EOs]], min_frequency=2)
tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer =trainer )
WSL : Ubuntu > TIME UNE Ln 72. Col39 Spaces:4 UTF-8 LF( Python 3. 10. 6(trans
ENG
6:53 PM
5/23/2023
_build _tokenizer (config, d s, lang) 然后我们想要将这个模型输出与我们预期的结果进行比较, 即
> OUT U NE
> WSL : Ubuntu > TIME LNE And then we want to compare this model output with what we expected, so. with the.
ENG
5/23/2023
6:54 PM
79
80
yield item [‘translation ′][lang]
81
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['to ke
82
83
if not Path. exists (tokenizer _path) 与标签进行比较.
> OUTUNE
84
85
tokenizer = Tokenizer ( Word Le
> WSL : Ubuntu > TIMELINE And then we want to compare this model output with what we expected, so. with the 6:54 PM
5/23/2023
79
80
81
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang))
82
83
if not Path. exists (tokenizer _path):
> OUTUNE
84
85
tokenizer = Tokenizer ( Word Level (unk _token'[ UNK ]))
tokenizer. pre _tokenizer = Whitespace ()
WSL : Ubuntu > TIMELINE n 74, Col13 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
6:54 PM
ENG
5/23/2023
79
80
yield item [translation *][lang]
83
81
def
get_or_build _tokenizer (config, ds, la tokenizer _path
> OUTUNE
84
ifnot Path.
tokenizer 所以让我们把这些全部追加进去.
WSL: Ubuntu
> TIMELNE
85
Solet's append all of these.
6:54 PM
ENG
5/23/2023
79
80
yield item ['translation'][lang]
83
81
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
if not Path. exists (tokenizer _path):
> OUTUNE
84
85
tokenizer = Tokenizer ( Word Level (unk _token ='[ UNK ]))
> WSL : Ubuntu > TIME UNE Ln 74. Col13 Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
5/23/2023
6:54 PM
81
def
get_or_build _tokenizer (config, d s, lang)
82
83
tokenizer 所以我们给输入的, 我们给了模型,
> OUTUNE
84
> WSL: Ubuntu > TIME UNE
ENG
5/23/2023
6:54 PM
79
80
yield item ['translation'][lang]
83
81
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
if not Path. exists (tokenizer _path):
> OUTUNE
84
85
tokenizer = Tokenizer ( Word Level (unk _token a'[ UNK]'))
> WSL: Ubuntu
> TIMEUNE
Ln57, Col. 17 (12 selec
ENG
6:54 PM
5/23/2023
yield item [*translation'][lang ]
get _or _build _tokenizer (config, d s, lang) 模型的输出, 即预测的输出, 以及我们预期的输出, 我们
> OUT U NE
WSL : Ubuntu > TIME UNE What the model output, the output of the model, so the predicted, and what we 6:54 PM
5/23/2023
79 yield item [*translation'][lang]
81
def
get_or_build _tokenizer (config, d s, lang)
tokenizer _path 82
83
if not Path. exi 将所有这些保存在这些列表中,
> OUTUNE
84
tokenizer > WSL : Ubuntu > TIME UNE expected as'ou it put, we'save all of this in this list s. csa urs u
ython 3. io trans t
ENG
5/23/2023
6:54 PM
79
80
81
def get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang))
83
82
if not Path. exists (tokenizer _path):
> OUTUNE
84
tokenizer = Tokenizer ( Word Level (unk _token'[ UNK ]'))
> WSL : Ubuntu > TIME UNE
n58. Col1 S(24selected ) Spaces 4 UTF-8 LF {↓ Python 3. 10. 6(tran
ENG
6:54 PM
5/23/2023
yield item [*translation'][lang ]
get _or _build _tokenizer (config, ds, lang) 然后在循环结束时, 我们将在控制台上打印它们.
> OUTUNE
WSL: Ubuntu00
> TIME UNE And then'at the end of the loop, we will print them on the console. yton
6:54 PM
ENG
5/23/2023
79
80
yield item ['translation'][lang]
81
def get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang))
83
82
if not Path. exists (tokenizer _path):
> OUTUNE
84
85
tokenizer = Tokenizer ( Word Level (unk _token ='[ UNK ]))
tokenizer. pre _tokenizer =whitespace ()
WSL : Ubuntu > TIME UNE Ln 74. Col 13 Spaces4 UTF-8 LF( Python 3. 10. 6 (trans
ENG
5/23/2023
6:54 PM
80
def
get_al1_sentences (d s, lang):
for item in ds :
yield item ['translation ′][lang ] 为了获取模型输出的文本我们需要再次使用tokenizer 将token > OUT U NE
WSL : Ubuntu > TIME LNE to get the text of the output of the model we need to use the tokenizer again to 5/23/2023
5:54 PM
for item in ds :
yield item ['translation'][lang ] 转换回文本当然我们使用目标tokenizer, 因为这是
> OUTUNE
WSL: Ubuntu10
> TIMEUNE
convert the tokens back into text and we use of course the target tokenizer because r 口
ENG
5/23/2023
6:55 PM
def get _all _sentences (ds, 1ang):
for item in ds:
yield item ['translation'][lang ] 转换回文本当然我们使用目标tokenizer, 因为这是
> OUT U NE
WSL : Ubuntu > TIMELINE this is the target language Ln 76, Col 43
Pytho
6:55 PM
5/23/2023
80
79
def
get _all _sentences (ds, lang):
for item in ds
81
yield item ['translation'][lang]
82
83
def get_or_build _tokenizer (config, ds, lang): 目标语言
> OUTUNE
84
85
ifnot Path. exists (tokenizer _path ):
tokenizer _path = Path (config ['tokenizer _fil WSL : Ubuntu > TIMELINE this is the target language Ln 76, Col43 Spaces4 UTF-8 LF{ Python 3. 10. 6(trans
ENG
6:55 PM 口
5/23/2023
80
79
def
get _all _sentences (ds, lang):
for item in ds:
81
yield item ['translation'][lang]
> OUTUNE
83
84
def get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
> TIMELINE 85
if not Path. exists (tokenizer _path ):
WSL : Ubuntu Ln 76, Col44 Spaces4 UTF-8 LF( Python 3. 10. 6(trans
ENG
5/23/2023
6:55 PM
好的, 现在我们将所有这些保存到各自的列表中, 我们也
> OUT U NE
WSL : Ubuntu > TIME LNE okay and now we save them all of this into their respective lists and we can also ENG
5/23/2023
6:55 PM
80
e.]_path_
prl
82
83
def
get _all _sentences (d s, 可以在控制台上打印出来.
> OUTUNE
84
For
item in ds
ield item['trans l a
> WSLUbuntu > TIME UNE okay and now we save them all of this into their respective lists and we can also 6:55 PM 口
ENG
5/23/2023
80 predicted. append )
expected. append (t
81
82
83
def
get _all _sentences (ds, 可以在控制台上打印出来.
> OUTUNE
84
85
for
item in ds:
yield item ['translation'][lang ]
WSL : Ubuntu > TIME UNE print it on the console Ln 80. Col30 Spaces4 UTF-8 LF ( Python 3. 10. 6(tran
6:55 PM
ENG
5/23/2023
80 predicted. append (model _out _text )
expected. append (target _text )
81
82
83
> OUTUNE
84
def get_al1_sentences (d s, lang):
WSL : Ubuntu > TIME LNE
Ln 82, Col16 Spaces4 UTF-8 LFPython
6:55 PM
ENG
5/23/2023
predicted. app expected. append (target _text )
d (model _out _text ) 为什么我们使用这个名为print Message 的函数, 而不是直接使用 Python > WSL : Ubuntu > TIME LNE Why are we using this function called print Message, and why not just use the print. of
5/23/2023
6:56 PM
80
81
predicted. append (model _out _text )
expected. append (target _text )
82
83
# Print to the console print _msg 的 print?
> OUT UNE
84
> WSL: Ubuntu
> TIMEUNE
Why are we using this function called print Message, and why not just use the print of
6:56 PM 口
5/23/2023
80 predicted. appe
nd(model _out _text )
> OUTUNE 因为我们在主循环, 即训练循环中使用这里, TKODM, 这是一个
WSL: Ubuntu00
> TIME UNE Because we are using here in the main loop, in the training loop, we are using here,
6:56 PM
ENG
5/23/2023
194 optimizer. zero _grad () 非常漂亮的进度条, 但在进度条运行时不建议直接在控制台上
global _step > OUT UNE
WSL: Ubuntu
> TIMELNE
TKo DM, which is our really nice looking progress bar, but it is not suggested to. cm.
5/23/2023
6:56 PM
optimizer. zero _grad ()
global _step += 1
> OUTUNE
198
> TIMEUNE
199
model _filename = get _weights _file _path(conf
WSL: Ubuntu
ENG
5/23/2023
6:56 PM
optimizer. zero _grad ()
global _step += 1
> OUTUNE
198
199
# Save the model at the end of every epoch > TIME UNE
200
model _filename = get _weights _file _path (config, f'{epoch:02d}')
torch. save((
WSL: Ubuntu Ln 83. Col22 Spaces:4 UTF-8 LF( Python 3. 10. 6(transformer :con
5/23/2023
6:56 PM
194
195
optimizer. zero_grad()
9
global _step > OUT UNE 所以在控制台上打印 TKODM提供了一个名为print的方法
WSL: Ubuntu > TIME UNE So to print on the console, there is one method called print, pton
6:56 PM
5/23/2023
194
195
optimizer. zero _grad () 所以在控制台上打印 TKODM提供了一个名为print的方法
9
gl obal_ste
> OUTUNE
WSL: Ubuntu > TIME LNE torch. save (
83, Col 22
Pytho
6:56 PM
5/23/2023
195
194
optimizer. zero _grad ()
L96
global_step
> OUTUNE 所以在控制台上打印 TKODM提供了一个名为print 的方法,
WSL : Ubuntu > TIME LNE provided by Tko DM and we will give this method to this function so. that the output.
ENG
5/23/2023
6:56 PM
yield item ['translation'][lang ] 我们将这个方法传递给这个函数, 以确保输出不会干扰进度
> OUT UI NE
WSL : Ubuntu > TIMELINE does not interfere with the progress part printing..
ENG
5/23/2023
6:56 PM
88
89
yield item [‘translation'][lang]
91
90
def
get_or_build _tokenizer (config, d s, lang)
tokenizer _path = Path (config ['tokenize 部分的打印.
> OUTUINE
92
if not Path. exists (tokenizer _path ):
> WSL : Ubuntu > TIMELINE izer = Tokenizer (
ENG
6:56 PM
5/23/2023
88
89
yield item ['translation'][lang]
90
def
get_or_build _tokenizer (config, ds, lang):
> OUTUINE
91
92
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
if not Path. exists (tokenizer _path):
> WSL: Ubuntu
> TIMELINE
93
tokenizer = Tokenizer ( Word Level (unk _token ='[ UN K]*))
Ln83, Col22 Spaces:4 UTF-8 LFPython 3. 10. 6(trans
ENG
6:56 PM
5/23/2023
88
89
for item in ds:
yield item ['translation'][lang]
90
91 我们打印一些条形图, 然后打印所有消息.
> OUTUNE
92
WSL: Ubuntu > TIME L NE
We print'some bars and then we print all the messages :
UTF-8 LF Python3. 10. 6(trans fo
ENG
6:56 PM
5/23/2023
88
89
for item in ds:
yield item [‘translation'][lang]
90
91
def
get_or_build _tokenizer (config, ds, lang):
> OUTUNE
92
if not Path. exists (tokenizer _path ):
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
WSL : Ubuntu > TIME UNE Ln 84. Col25 Spaces:4 UTF-8 LF( Python 3. 10. 6(transfo
ENG
5/23/2023
6:56 PM
88 如果我们已经处理了一定数量的示例, 那么我们就中断.
> OUTUNE
> WSL: Ubuntu
> TIMELNE
and if we have already processed number of examples then we just break so why we. have ond
n`][lang
5/23/2023
6:57 PM
87
88
break
> OUTUNE 嗯, 实际上我们也可以将所有这些发送到,, tensor board, 所以
WSL: Ubuntu00
> TIMEUNE
created these lists uh actually we can also send all of this to um to uh tensor board ENG
5/23/2023
6:57 PM
9
96
yield item ['translation'][lang ] 我们可以. 例如-如果我们启用了tensor board, 我们可以将所有
> OUTUNE
> WSL: Ubuntu00
> TIMELINE created these lists uh actually we can also send all of this to um to uh tensorboard.
5/23/2023
6:57 PM
96
for
item in ds
yield item ['translation'][lang ]
98
97
> OUTUNE
100
99
def
get_or_build _token iz 这些发送到tensor board.
> TIME LNE
cokenizer_path = Pati
WSL: Ubuntu can so for example if we have tensor board enabled we can send all of this. to. the ENG
5/23/2023
6:57 PM
96
ror
item in ds:
yield item ['translation'][lang ]
98
97
> OUTUNE
100
99
def
get_or_build _token iz 这些发送到tensorboard.
> TIMELNE
101
if not Path. exists (tokenizer _path ):
tokenizer _path = Pat
WSL: Ubuntu
1 A0
tensor board Ln93, Col9 Spaces:4 UTF-8 LF Python 3. 10. 6 (trans
ENG
5/23/2023
6:57 PM
96
ror
item in ds:
yield item ['translation'][lang ]
98
97
> OUTUNE
100
99
def
get_or_build _tokenizer (config, d s, lang):
tokenizer _path = Path (config ['tokenizer _file']. format (lang ))
WSL: Ubuntu
> TIMELNE
1 A0
101
if not Path. exists (tokenizer _path):
Ln93, Col9 Spaces :4 UTF-8 LFPython3. 10. 6(trans fo
ENG
6:57 PM
5/23/2023
96
7
for item in ds:
yield item ['translation'][lang ]
> OUTUNE 为此 实际上我们需要另一个库, 它允许我们计算一些指标.
WSL: Ubuntu10
> TIMELNE
and to do that actually we need another library that allow us to calculate. some 6:57 PM
95
def
get_al1_sentences (d s, lang):
item in ds or yield item ['tra ans lation'][lang ]
> OUT UNE 我想我们可以跳过这部分, 但如果你真的很感兴趣, 我在我
WSL: Ubuntu
> TIMELNE
1 A0
metrics i think we can skip this part but if you're really interested iii n myinucta
ENG
5/23/2023
6:58 PM
我想我们可以跳过这部分,
> OUT U NE 但如果你真的很感兴趣, 我在我
WSL : Ubuntu > TIMELINE Path (config ['tokenizer _file']
the codel
Ln93, Col 11
Pyti
6:58 PM
5/23/2023
发布在 Git Hub上的代码中,
> OUTUNE 你会发现我使用了这个名为torch
WSL: Ubuntu > TIMELINE published on git hub you will find that i use this library called the torch matrix. m 口
5/23/2023
5:58 PM
det
or
iter 发布在 Git Hub上的代码中, 你会发现我使用了这个名为torch
> OUTUNE
WSL : Ubuntu > TIME LNE that allows to calculate the char error rate coo
Pytho
5:58 PM
5/23/2023
95
7
det
or
matrix的库, 它允许计算字符错误率、 BL EU指标, 这对翻译
> OUTUNE
WSL: Ubuntu > TIME UNE that allows to calculate the char error rate co2 spaces 4ursu
Pytho
5:58 PM
5/23/2023
96
def
get _all _sentences (d s, lang):
for item in ds :
98
97
yield item[ 任务非常有用, 还有词错误率
> OUTUNE
100
99
def get _or _build _tol
WSL: Ubuntu10
> TIMEUNE
and the bleu the bleu metric which is really useful for translation. task stand. the ENG
5/23/2023
6:58 PM
lang ) 所以如果你真的感兴趣, 你可以在 Git Hub上找到代码, 但
> OUTUNE
WSL: Ubuntu10
> TIME UNE word error rate so if you really interested you can find the code on the githubbut
?
ENG
5/23/2023
6:58 PM
所以如果你真的感兴趣, 你可以在 Git Hub 上找到代码, 但
> OUT U NE
> WSL : Ubuntu > TIME LNE Path (config ['token iz
forour
n93, Col 58
UTF-8 LF{ Python 5:58 PM
5/23/2023
95
> OUTUNE 对于我们的演示, 我认为这并不必要. 所以, 实际上
> WSL: Ubuntu
> TIMELNE
Path (config ['t ken izer _file']. f
forour
3. 10. 6(t
6:58 PM
5/23/2023
95
s(ds, lang) 对于我们的演示,. 我认为这并不必要. 所以, 实际上
> OUTUNE
> WSL: Ubuntu > TIME LNE
1 A0
demonstration i think it's not necessary so and actually this we can also remove it ENG
5/23/2023
6:58 PM
9
94 我们可以移除它, 国 因为我们没有做这部分.
> OUTUNE
9
> WSL: Ubuntu > TIMELINE demonstration i think it's not necessary so and actually this we can also remove it 6:58 PM
5/23/2023
9 我们可以移除它, 因为我们没有做这部分.
> OUTUNE
> WSL: Ubuntu > TIMELINE given that we are not doing this part i(aisckted
4. 10. 6(trans
UTF-8 LFPython ENG
6:58 PM
5/23/2023
92
if not Path. exists (tokenizer _path ):
tokenizer = Tokenizer ( Word Level (unk _token ='[u Nk ]))
tokenizer. pre _tokenizer = Whitespace ()
> OUTUNE
95
96
trainer = Word Level Trainer (special _token s=["[ UNK]","[ PAD]","[sos]","[ EOs]], min_frequency=2)
tokenizer. train _from _iterator (get _all _sentences (d s, lang), trainer a trainer )
WSL : Ubuntu > TIMELINE tokenizer. save (str (tokenizer _path ))
Ln57, Col1: Spaces4 UTF-8 LF { Python 3. 10. 6(transfo
ENG
5/23/2023
6:58 PM
91 tokenizer = Tokenizer ( Word Level (unk _token =′[ UNk ])) 好的, 现在我们有了run Validation 方法, 我们可以直接调用它.
trainer d Level Trainer (special _token s=["[u Nl
os]"]. min _frequency =2)
> OUTUNE
WSL: Ubuntu00
> TIMELINE Okay, so now that we have our run Validation method, we can just call. it..
4. 10. 6 C
5/23/2023
6:58 PM
train _data loader = Data Loader (train _ds, batch _size =config ['batch _size'], shuffle = True )
val _data loader = Data Loader (val_ds, batch_size=1, shuffle = True )
return train _data loader, val _data loader, tokenizer _src, tokenizer _tgt
> OUTUNE
132
133
def get_model (config, vocab _src _ Len, vocab _t gt_len):
> WSL: Ubuntu
> TIMEUNE
133
model = build _transformer (vocab_src_ Len, vocab_tgt_len, config [seq len'], config [seq len'], config['d
Ln82, Col1 Spaces 4 UIF-8 LF{ Python3. 10. 6(trans odel'1)
ENG
5/23/2023
6:58 PM
194
195
gl obal_step +=1
196
> OUTUNE 我通常的做法是在每隔几步后运行验证, 但由于我们希望尽快
WSL: Ubuntu > TIME UNE what I usually do is l run the validation at every few steps but because we want to 5/23/2023
6:58 PM
194
gl obal_step +=1
196
> OUTUNE 看到结果我们将首先在每次送代时运行它, 并且我们还把
WSL: Ubuntu > TIMELINE see it as soon as possible what we will do is we will first run it at every iteration 5/23/2023
6:58 PM
194
195
gl obal_step +=1
196
> OUTUNE 看到结果我们将首先在每次送代时运行它, 并且我们还把
WSL: Ubuntu > TIME UNE del _state _dict':model. state _dict (),
and we Ln 163, Col9 (13 se
5/23/2023
6:59 PM
195
194
global _step +=1 这个modeltrain放在这个循环中, 以便每次运行验证后, 模型都
4. 96
> OUTUNE
WSL: Ubuntu > TIME UNE and we Ln 163, Col9 (13 sek
5/23/2023
6:59 PM
196 # Save the model at the end global _step +=1 会回到训练模式.
> OUTUNE
198
model _filename =get _weights _
> TIME LNE validation the model'is back into its training mode so now we can just run validation.
199
torch. save((
> WSL: Ubuntu 5/23/2023
6:59 PM
195
2. 96
> OUTUNE 所以现在我们可以直接运行验证, 并给它所有运行验证所需的
WSL: Ubuntu > TIME LNE validation the model is back into its training mode so now we can just run validation ENG
5/23/2023
6:59 PM
195
194
run_val idatior
> OUTUNE 所以现在我们可以直接运行验证, 并给它所有运行验证所需的
WSL: Ubuntu > TIMELINE lel _filename = get _weights _file _path (config,
5/23/2023
6:59 PM
194
195
num_examples:int=2)-> None run _validation ()
> OUT U NE 所以现在我们可以直接运行验证, 并给它所有运行验证所需的
WSL : Ubuntu > TIMELINE and we give it all'the parameters that it needs to run the validation so give. it os trn 口
ENG
5/23/2023
6:59 PM
num _examples :in t=2)-> None
run_validation (model, val _data loader x enizer_srd)
198 参数."给它模型. 好的, 用于打印消息.
> OUTUNE
199
WSL: Ubuntu > TIMELINE 0 A1 and we give it all the parameters that it needs to run the validation so give. it os tan
6:59 PM
5/23/2023
examples :in t=2)-> None
run_validation (model, val _data loade
_tgt, con 参数. 给它模型. 好的, 用于打印消息.
> OUTUNE
WSL: Ubuntu > TIME UNE no del _filename =get model okay for printing message Ln195. Col84 Spaces4 UTF-8 LF Python 3. 10. 6(transfo
ENG
5/23/2023
6:59 PM
num _examples :in t =2)-> None
run_validation (model, val _data loader, tokenizer _src, tokenizer _tgt, config ['seq _len'], device, D
global_step +=1
> OUTUNE
199
# Save the model at the end of every epoch WSL : Ubuntu > TIME UNE model _filename = get _weights _file _path (config, f'{epoch:02d}')
Ln195 Col108 Spaces :4 UTF-8 LF( Python3. 10. 6(transfomer:conda)
> ENG
6:59 PM 口
5/23/2023
178
179
label =batch [label']. to (device )#( B, Seq_ Len)
18
batch_iterator loss = loss _f n(pr 我们在打印任何消息吗?
> OUT UNE
WSL: Ubuntu
> TIMELINE
184
are we printing any message?
Ln 195, Col108 Spaces:4 UTF-8 LF Python 3. 10. 6(transfo
6:59 PM 口
Z
5/23/2023
print(f'Max length of source sentence :{max_len_src )')
print(f Max length of target sentence :{max_len_tgt))
> OUTUNE
122
train _data loader = Data Loader (train _ds, batch _size =config ['batch _size'], shuffle = True )
> TIMELINE 126
129
val_data loader = Data loader (val_ds, batch_size=1, shuffle = True )
> WSL : Ubuntu Ln 195, Col108 Spaces:4 UTF-8 LF ( Python 3. 10. 6 (trans
ENG
5/23/2023
6:59 PM
116
115
max_len src =θ
117
max_len_tgt=a
> OUT UNE 是的, 我们在打印所以让我们创建一个lambda, 我们只需
WSL : Ubuntu > TIMELINE we are so let's create a lambda and we just do and this is the message to write with.
ENG
5/23/2023
6:59 PM
203
model_state_dict':
te_dict()
[]disable > OUTUNE 这是要写的消息与 TQDM. 然后我们需要给出全局步骤和
WSL: Ubuntu20
> TIME UNE we are so let'screate a lambda and we just do and this is the message to write with.
/23/2023
7:00 PM
202
'epoch':epoch,
203
> OUTUNE 做, 这是要写的消息与 TQDM. 然后我们需要给出全局步骤和
WSL: Ubuntu
> TIMEUNE
othe TQDM then we need to give the global step and the writer which we will not use 口
ENG
/23/2023
7:00 PM
model _state _dict':model. state _dict (),
optimizer _state _dict':
dict()
286
global _step writer, 我们不会使用它.
> OUT UNE ), model _filename WSLUbuntu
> TIMELNE
the TQDM then we need to give the global step and the writer which we will not use or 口
ENG
5/23/2023
7:00 PM
203
_dict()
> OUTUNE 但好的现在我想我们可以再次运行训练, 看看验证是否
WSL: Ubuntu > TIME UNE
butokay
Z
/23/2023
7:00 PM
203
> OUTUNE 但好的, 现在我想我们可以再次运行训练, 看看验证是否
WSL: Ubuntu00
> TIME UNE now i think we can run the training again and see if the validation works.
3. 10. 6 (tra
7:00 PM
5/23/2023
config =get_config()
211 train _model (config ) 有效.
> OUT UNE
WSL: Ubuntu00
> TI MEUNE
now i think we can run the training again and see if the validation works 3. 10. 6 (transfo
ENG
5/23/2023
7:00 PM
config =get_config()
211
train_model(config)
> OUTUNE
WSL: Ubuntu
> TIMEUNE
Ln 195, Col 1 Spaces :4 UTF-8 LF( Python3. 10. 6(transformer :conda)
7:00 PM 口
5/23/2023
AR hay K POINTS Raised Exceptions Uncaught Exceptions 好的, 看起来它在工作
User Uncaught Exceptions All right, looks like it is working.
0/363800:03?, it/s, 1oss=10. 051]
> WSL: Ubunt
Ln 195, Col1 Spaces :4 UTF-8 LF( Python3. 10. 6 Ctrans f
mer:conda)
ENG
7:00 PM
And you est iti!
Raised Exceptions K POINTS 所以模型没问题
Uncaught Exceptions > WSL : Ubunt
User Uncaught Exceptions So the model is okay.
Pyth
7:01 PM
ENG
K POINTS Raised Exceptions Uncaught Exceptions 它在每一步都运行验证, 这完全不是我们想要的
> WSL : Ubunt
User Uncaught Exceptions It's running the validation at every step, which is not desirable atal
ENG
7:01 PM
5/23/2023
Raised Exceptions K POINTS fet to!
Uncaught Exceptions WSL : Ubunt
User Uncaught Exceptions 7:01 PM
Raised Exceptions Uncaught Exceptions 但至少我们知道贪婪搜索在工作, 而直它不是. 至少看起来
W SL: Ubun User Uncaught Excep but at least we know that the greedy search is working and it's i not at least look /23/2023
7:01 PM
Raised Exceptions K POINTS 但至少我们知道贪婪搜索在工作:而直它不是.:至少看起来
Uncaught Exceptions > WSL: Ubun
User Uncaught Exceg
like it is working and the model is not predicting anything useful actua
7:01 PM
all'id e
ndo il vasto pet to, sorrise sprezzante deld
Raised Exceptions K POINTS 它在工作, 模型没有预测任何有用的东西
Uncaught Exceptions > W SL: Ubun User Uncaught Exce like it is working and the model is not predicting anything useful actu
UIST
7s/it,
7:01 PM
Raised Exceptions K POINTS 实际上它只是在预测一堆逗号, 因为它完全没有训练.:但
ai gioielli!
Uncaught Exceptions WSL : Ubunt
User Uncaught Exceptions predicting 7:01 PM
AK POINTS 如果我们训练模型一段时间后;我们应该会看到在几个epoch Raised Exceptions Uncaught Exceptions > W SL: Ubun User Uncaught Excep a bunch of commas because it's not trained at all but if we train the model after a
Z
ENG
7:01 PM
80alla virtu attribuire, non mi pare da lasciarli indrieto, ancora Raised Exceptions POINTS 之后:模型会变得越来越好
> WSL: Ubun
better
582] 口
ENG
7:01 PM
HOU HAS T HID hile talking Raised Exceptions POINTS 之后,:模型会变得越来越好
Uncaught Exceptions > W SL: Ubun User Uncaught Exceptions and better 72s/it,
5821
195, Col. 1
Pyth
7:01 PM
ENG
s wife that night.
Raised Exceptions POINTS Uncaught Exceptions > W SL: Ubun User Uncaught Exceptions 72s/it,
5821
7:01 PM
Raised Exceptions K POINTS 所以让我们停止这个训练;:并把这个放回它该在的地方
Uncaught Exceptions W SL: Ubun User Uncaught Except so let's stop this training and let'sput this one back to where it belongs so atth
7:01 PM
210 config = get_config()
AKPOINTS
211
train _model (config )
Raised Exceptions Uncaught Exceptions 每个 epoch 的末尾. 是的, 这个我们可以留在这里, 没
WSL : Ubuntu User Uncaught Except so let's stop this training and let's put this one back to where it belongs so at the 中
5/23/2023
7:01 PM
AK POINTS 每个 epoch 的末尾. 是的, 这个我们可以留在这里, 没
Raised Exceptions Uncaught Exceptions WSL : Ubuntu User Uncaught Exceptions 0 A0 Q1fletoanaly end of every epoch yeah and this one we can keep it here no problem ythoe
4. 10. 6(t
7:01 PM
5/23/2023
191 # Update the weights AKPOINTS
193
192
optimizer. zero _grad ()
optimizer. step () 问题.
Raised Exceptions Uncaught Exceptions 194
global_step +=1
User Uncaught Exceptions 195
WSL: Ubuntu 口
5/23/2023
7:01 PM
198 # Update the weights S REAK POINTS optimizer. zero _grad ()
optimizer. step ()
Uncaught Exceptions Raised Exceptions 194
195
global_step +=1
WSL: Ubuntu User Uncaught Exceptions 196
Ln 167, Col1 Spaces4 UTF-8 LF ( Python 3. 10. 6(trans
7:01 PM 口
5/23/2023
210 config = get_config()
AKPOINTS
211
train _model (config )
Raised Exceptions Uncaught Exceptions 我现在将快进到一个已经预训练好的模型
WSL : Ubuntu User Uncaught Exceptions I will now skip fast forward to a model that has been pre-trained.
( Python 3. 10. 6(transfo
5/23/2023
7:01 PM
config = get_config()
SREAKPOINTS
211
train _model (config )
Uncaught Exceptions Raised Exceptions WSL : Ubuntu User Uncaught Exceptions Ln195, Col29 Spaces4 UTF-8 LF { Python 3. 10. 6(transt
7:01 PM 口
5/23/2023
210 config = get_config()
SREAKPOINTS
211
train _model (config )
Raised Exceptions Uncaught Exceptions 我预先训练了它几个小时, 这样我们就可以进行推理并可视化
WSL : Ubuntu User Uncaught Exce
ENG
5/23/2023
7:01 PM
config =get_config()
SREAKPOINTS
211
train _model (config )
Uncaught Exceptions Raised Exceptions WSL : Ubuntu User Uncaught Exceptions Ln 195, Col29 Spaces:4 UTF-8 LF { Python 3. 10. 6(trans
7:01 PM
5/23/2023
model. py Inference ipy nb SOURCE :" Never in my life."
tokenizer _en json TARGET :-Mai.
PREDICTED :
(1tokenizeritjson 我已经复制了预先计算好的预训练权重
> OUTUNE
talk to his wife : he was ashamed to WSL: Ubuntu00
> TIMEUNE
Thave copied the pre-trained weights that I pre-calculated.
Ln5, Col52 Cell1of 3
ENG
5/23/2023
7:04 PM
model py Inference ipy nb SOURCE :" Never in my life.
1tokenizer enjson
(1 tokenizerjtj son TARGET : - Mai.
PREDICTED :-Mai.
> OUT UI NE
WSL : Ubuntu > TIME LNE PREDICTED : Senza parlare di cena, di son no, senza rif lettere a quello che avrebbe fatto, egli non poteva neanche parlare con sua moglie:si vergog
Ln5, Col52 Cell 1of 3
ENG
5/23/2023
7:05 PM
Inference. ipy nb model. py SOURCE :" Never in my life."
( tokenizer _enjson {1tokenizerjtjson
TARGET :-Mai.
> OUT UI NE 我还创建了这个笔记本,,·重用了我们在训练文件中之前定义的
WSL : Ubuntu > TIMELINE And I also created this notebook re using the functions that we have defined before in
7:05 PM
Inference. ipy nb model. py SOURCE :" Never in my life."
1 tokenizer itjson
1tokenizer_enjson
TARGET:-Mai. 实际上:我只是从训练文件中复制粘贴了代码
> OUTUNE
ld not even talk to his wife :he was ashamed to WSL: Ubuntu00
> TIMELINE Actually, T just copy and pasted'the code'from the train file.
Ln 4, Col 4 (101 selected ) Cell 2 of 3
ENG
7:05 PM
/23/2023
Inference. ipy nb model py SOURCE :" Never in my life."
tokenizer. enjson 1tolkenizeritjson
TARGET : - Mai.
PREDICTED :-Mai.
> OUT UNE SOURCE : Not only could he not think of supper, of getting ready for the night, of considering what they were to do, he could not even talk to his wife : he was ashamed to > WSL: Ubuntu00
> TIMELINE PREDICTED : Senza parlare di cena, di son no, senza rif lettere a quello che avrebbe fatto, egli non poteva neanche parlare con sua moglie:si vergog
In 5, Col 11 (5 selected ) Cell 2 of 3
7:05 PM
5/23/2023
In ference. ipy nb
SOURCE:" Never i n my life."
model py
tokenizer_enjson
TARGET:-Mai.
{1tokenizer_itjson 我只是加载模型并运行验证:使用我们刚刚写的方法
> OUTUNE
> WSL: Ubuntu00
> TIMELINE
Ijust load themodel and'runthe Validation,
thesamemethodthatwej
ustwrote.
/23/2023
7:05 PM
Inference. ipy nb model. py SOURCE :" Never in my life."
1 tokenizer. enjson
1 tokenizerit json PREDICTED :-Mai.
TARGET : - Mai.
> OUT UI NE
WSL : Ubuntu > TIME LNE PREDICTED : Senza parlare di cena, di so no, senza rif lettere a quello che avrebbe fatto, egli non poteva neanche parlare con sua moglie : si vergog
Ln 5, Col11(5 selected ) Cell 3 of 3
ENG
7:05 PM
5/23/2023
Inference. ipy nb model py SOURCE : It must have been most irk so stead of a parent to a strange child she could not love, and to see an
1tokenizer itjson
1 token izer. enjson PREDICTED : Era pent it a di esser TARGET : Era pen tita di essersi
poteva voler be ne e di vedere une strane a mescola ta al gruppo > OUTLINE SOURCE :" So mysterious!" 让我们再运行一次, 例如
WSL : Ubuntu > TIME LNE PREDICTED :-Veramente mister Let'srun it again, for example.
Ln5, Col11(5 selected ) Cel3of3 A(
ENG
7:05 PM
5/23/2023
Inference. ipy nb model py SOURCE : It must have been most irksome to find herself bound by a hard-w ung pledge to stand in the stead of a parent to a strange child she could not love, and to see an
1tokenizer. enjson
PREDICTED : E ra pentita di essersi impegnata con una pro mess a solenne a far da madre a una bam bina cui non poteva voler bene e di vedere unestranea mescola ta al gruppo TARGET : E ra pentita di essersi impegnata con una pro mess a solenn
afar da madre a una
(1tokenizer it json SOURCE :" So mysterious!" said Harris.
> OUT UNE TARGET :-Veramente misterio sal-disse Harris.
WSL : Ubuntu > TIMELINE PREDICTED :-Veramente misterio sa|-dis se Harris
Ln5, Col11(5 seleted) Cell 3of3 A
ENG
7:05 PM
5/23/2023
1 tokenizer enjson
{1tokenizer_it json model. p y
> OUTUNE 如你所见, 模型正在推理10个例子, 句子, 结果还
> WSL: Ubuntu
> TIMELINE
7:05 PM
Inference. ipy nb model. py TARGET : Saint-John par lava automaticamente ; lui solo sap eva quanto costa vagli quel rifiuto.
(1 tokenizer it json {tokenizer _enjson 不错. 我的意思是, 我们可以看到11个微笑, 11个
> OUTUINE
> TIMELNE
you can see the model is inferencing 10 examples sentences and the result is cel+o
7:05 PM
Inference. ipy nb model py PREDICTED : Ma
capisco
(1tokenizeritjson {tokenizer _en json > OUTUINE 不错. 我的意思是, 我们可以看到11个微笑, 11个
> TIMELNE
7:05 PM
Inference. ipy nb model py PREDICTED : Ma io non cap is co come egli abbia potuto dimenticare
(1tokenizer it json
{1tokenizer_enjson
> OUTUIN E 故事, 11个故事.
> WSL: Ubuntu
> TIMELINE
7:05 PM 口
Inference. ipy nb model py PREDICTED : Ma io
capisco com
egli abbia potuto diment i care mad re e fare di voi un ′in felice ;non ave va cuor
(1 tokenizer enjson
{1tokenizer_it json
> OUTUINE 它在匹配, 而且大多数都匹配. 实际上, 我们也可以说它
> TIMELINE
7:05 PM
In ference. ipynb
model. py
PREDICTED:-Si,
dei amici-
cosa di cui mi sono trovata piu di quelli amici, n
evo di che cosa volesse dire
(1 tokenizer enjson
(1tokenizeritjson 它在匹配, 而且大多数都匹配. 实际上:我们也可以说它
> OUT UI NE
WSL : Ubuntu > TIMELINE of the m
Ln 5, Col 17 Cell 3 of 4
7:05 PM
5/23/2023
Inference. ip ynb
modelpy
(1 tokenizerenjson satisfy hi quite {tokenizer it json 它在匹配, 而且大多数都匹配. 实际上, 我们也可以说它
> OUT U NE
WSL: Ubuntu
> TIMELNE
/23/2023
7:05 PM
Inference. ipy nb model. py {tokenizer itjson
{1tokenizer_enjson 几乎过度拟合了这个特定数据, 但这就是transformer 的力量.
nificat
> OUTUINE
WSL: Ubuntu > TIMELINE matching actually we could also say that it's nearly over fit for this particular co7 celi3oa 口
7:05 PM
Inference. ipy nb model py PREDICTED :-Ecco, ma poi, che cosa ne pensano icontadini?
TARGET :
-Ecco, ma di
tokenizer _enjson
(1 tokenizer. it json > OUT UI NE 我没有训练它很多天我只训练了几个小时, 如果我没记错
> TIMELINE data but this is the power of the transformer i didn't train it for many days ijust# cel3o
7:05 PM
/23/2023
Inference. ipynb
model. py
TARGET:
1 tokenizer it json
[1 tokenizer_en. json
> OUTUNE 我没有训练它很多天:我只训练了几个小时, 如果我没记错
WSL : Ubuntu > TIME LNE
trainedit
Ln5, Col 17 Cell 3of 4
7:05 PM 口
/23/2023
model. py
Inferenceipynb
TARGET:
(1tokenizer it json
{1tokenizer_enjson
> OUTUNE 我没有训练它很多天:我只训练了几个小时, 如果我没记错
WSL : Ubuntu > TIME LNE for a few hours if i remember correctly and the results are really really good Col17 Cell3of4 司
7:05 PM
model py Inference ipynb
TARGET:-Ecco, ma di’un po', che ne
PREDIc TED :-Ecco, ma poi, che cosa ne pensano i contadini?
pensano i contadini?
(1tokenizer jtjson
{1 tokenizer_en. json
URCE: Mr. St. John spoke al m 的话, 结果真的非常好.
im thus to refuse > OUT UNE TARGET : Saint-John par lava > TIME UNE PREDICTED : Saint-John par l a
WSL: Ubuntu for a few hours if i remember correctly and the results are really really good s
Col17 Cell3of4 口
ENG
5/23/2023
7:05 PM
model. py Inference ipy nb TARGET :- Eco, ma di’ un po', che ne pensano i conta dini?
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
(tokenizer. enjson
1 tokenizer. it json SOURCE : Mr. St. John spoke almost like an automaton :himself only knew the effort it cost him thus to refuse.
> OUT UNE TARGET : Saint-ohn par lava automaticamente ; lui solo sap eva quanto costa vagliquelrifiu to.
PREDICTED : Saint - John par lava automaticamente ;lui solo sapeva quanto quel rifiuto.
WSL: Ubuntu00
> TIMEUNE
Ln 5, Col17 Cell3of4
Z
ENG
5/23/2023
7:05 PM
Inference. ipy nb model. py PREDICTED :-Ecco, ma poi, che cosa ne pensano i contadini?
TARGET :
pensa no i contadini?
tokenizer _enjson
{1tokenizer_it json > OUT U NE 现在让我们制作一个笔记本:用来可视化这个预训练模型的
WSL : Ubuntu > TIME UNE 口
7:05 PM
/23/2023
PREDICTED :-Ecco, ma poi, che cosa ne pensano i contadini?
> OUT U NE 现在让我们制作一个笔记本::用来可视化这个预训练模型的
WSL : Ubuntu > TIMELINE pre-trained model.
Ln5, Col. 17 Cell3 of 4
7:05 PM
/23/2023
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
TARGET : Saint-John par la va
URCE: Mr. St. John spoke 根据我们之前构建的文件
im thus to refuse > OUT UNE PREDICTED : Saint-John par > WSL : Ubuntu > TIMELINE Given the file that we built before, so train. pi, you can also train your own model,
Cell 3of 4
/23/2023
7:06 PM
sano i contadini?
icontadini?
PREDICTED :-Ecco, ma poi, che cosa 所以, trainpi你也可以选择你喜欢的语言来训练你自己的模型
> OUTUNE
WSL : Ubuntu > TIMELINE Given the file that we built before, so train. pi, you can also train your own model,
Cell 3 of 4
7:06 PM
PREDICTED :-Ecco, ma poi, che cosa
ne pensano i contadini?
> OUT UI NE 我强烈建议你改变语言, 看看模型的表现如何, 并尝试
WSL : Ubuntu > TIMELINE choosing the language of your choice, which l highly recommend that you change the i:o
7:06 PM
> OUT UI NE 我强烈建议你改变语言, 看看模型的表现如何, 并尝试
WSL : Ubuntu > TIMELINE language L n5. Col17 Cell 3 of 4 口
/23/2023
7:06 PM
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini? 我强烈建议你改变语言, 看看模型的表现如何, 并尝试
> OUT UI NE
> WSL : Ubuntu > TIMELINE and try to see how the model is performing and try to diagnose why the model is Cell 3of 4
7:06 PM
/23/2023
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
SOURCE : Mr. St. John spoke 诊断为什么模型表现不佳.
thus to refuse > OUTLINE TARGET : Saint-John par lava > TIMELINE and try to see how the model is performing and try to diagnose why the model is cel3o PREDICTED : Saint-John par > WSL: Ubuntu
ENG
5/23/2023
7:06 PM
PREDICTED :-Ecco, ma poi, che cosa ne pensano i contadini?
thus to refuse > OUTLINE PREDICTED : Saint - John par TARGET : Saint-John par lava WSL : Ubuntu > TIMELINE performing bad.
Ln 5, Col 17 Cell3 of 4
ENG
5/23/2023
7:06 PM
PREDIc TED :- Ecco, ma poi, che cosa ne pensano i contadini? 如果表现不好或表现良好, 尝试理解如何进一步改进它.
> OUTLINE WSL : Ubuntu > TIMELINE If it's performing bad or if it's performing well, try to understand how can you a7 celio4
ENG
/23/2023
7:06 PM
> OUT UI NE 如果表现不好或表现良好尝试理解如何进一步改进它.
WSL : Ubuntu > TIMELINE improve it further.
Ln 5, Col 17 Cell3 of 4 口
5/23/2023
7:06 PM
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
TARGET : Saint-John par lava automaticamente ; lui solo sap eva quanto costa vagli quel rifiuto.
So URCE: Mr. St. John spoke almost like an automaton :himself only knew the effort it cost him thus to refuse.
> OUT UI NE PREDICTED : Saint-John par lava automaticamente ;lui solo sapeva quanto quel rifiuto.
> WSL: Ubuntu
> TIMELINE
Ln 5, Col17 Cell3of4
ENG
5/23/2023
7:06 PM
TARGET :-Ecco, ma diun po',
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
TARGET : Saint-John SOURCE : Mr. St. 所以让我们创建一个新的笔记本?
refuse > OUT UI NE PREDICTED : Saint -
WSL : Ubuntu > TIMELINE So let's create a new notebook.
Ln 5, Col 17 Cell 3 of 4
5/23/2023
7:06 PM
PREDICTED :- Ecco, ma poi, che cosa ne pensano i contadini?
TARGET : Saint-John par lava automaticamente ; lui solo sap eva quanto costa vagli quel rifiuto.
SOURCE: Mr. St. John spoke almost like an automaton :himself only knew the effort it cost him thus to refuse.
> OUT UI NE PREDICTED : Saint-John par lava automaticamente ;1ui solo sapeva quanto quel rif iuto.
> WSL: Ubuntu
> TIMELINE
Ln5, Co l 17 Cell3 of 4
ENG
5/23/2023
7:06 PM
PREDIc TED :- Ecco, ma poi, che cosa ne pensano i contadini? 让我们称之为比如说, 注意力可视化, 好的, 我们做
> OUT UI NE
WSL : Ubuntu > TIMELINE let's call it let's say attention visualization okay so the first thing we do wes. coi7 cel3o
7:06 PM
/23/2023
> OUT UI NE 让我们称之为, 比如说, 注意力可视化, 好的, 我们做
> WSL : Ubuntu > TIME UNE import all the libraries we will need Cell 1of 1
7:06 PM
/23/2023
> OUT U NE 让我们称之为, 比如说, 注意力可视化, 好的, 我们做
> WSL : Ubuntu > TIMELINE Ln1. Col12 LF
Cell1of 1
7:06 PM
5/23/2023
> OUT UNE 的第一件事是导入所有需要的库, 我还将使用一个叫做 Altair WSL: Ubuntu
> TIMELINE
Ln3, Col1# LFCell1of 1
5/23/2023
7:06 PM
的库.
> OUT U NE
WSL: Ubuntu
> TIMELNE
1 A0
Ln6, Col8 LFCell1of1
7:07 PM
5/23/2023
的库.
> OUT UNE
WSL: Ubuntu
> TIMELINE
1 A0
I will also be using this library called Altair.
Ln6, Col8 LFCell1of 1
5/23/2023
7:07 PM
> OUT UNE
WSL: Ubuntu
> TIMEUNE
Ln6, Col8 LFCell1of1
5/23/2023
7:07 PM
这是一个用于图表的可视化库.
> OUT U NE
> WSL : Ubuntu > TIME UNE It's a visualization library for charts.
Ln6, Col8 LFCell1of 1
5/23/2023
7:07 PM
> OUT UNE
> WSL: Ubuntu
> TIMEUNE
Ln6, Col8 LFCell1of1
5/23/2023
7:07 PM
> OUT U NE 它实际上与深度学习无关.
> WSL : Ubuntu > TIME UNE It's nothing related to deep learning, actually.
Ln6, Col8 LFCell1of 1
ENG
5/23/2023
7:07 PM
> OUT U NE 它只是一个可视化功能.
> WSL : Ubuntu > TIME UNE It's just a visualization function.
Ln6, Col8 LF Cell1of 1
ENG
5/23/2023
7:07 PM
> OUT UNE
WSL: Ubuntu10
> TIMELNE
Ln6. Col8 LFCell1of1
5/23/2023
7:07 PM
> OUTUNE 特别是可视化功能, 实际上, 我在网上找到的.
WSL: Ubuntu10
> TIMEUNE
In particular, the visualization function, actually, I found it online.
Ln6, Col8 LF Cell1of 1
5/23/2023
7:07 PM
不是我写的.
> OUTUNE
> WSL: Ubuntu10
> TIMEUNE
It's not written by me.
Ln6. Col8 LF Cell1of 1
5/23/2023
7:07 PM
> OUT U NE 就像大多数可视化功能一样, 如果你想构建一个图表或
WSL : Ubuntu > TIME LNE Just like most of the visualization functions, you can find easily on the internet if
Cell 1of1
ENG
7:07 PM
/23/2023
> OUT UNE
WSL: Ubuntu10
> TIMEUNE
Ln6, Col8 LFCell1of1
5/23/2023
7:07 PM
所以我主要使用这个库, 因为我从网上复制了代码来可视化它.
> OUT UNE
WSL: Ubuntu10
> TIMEUNE
So lam using this library mostly because l copied the code from the internet to
Cell1of1
ENG
/23/2023
7:07 PM
所以我主要使用这个库, 因为我从网上复制了代码来可视化它.
> OUTUNE
WSL: Ubuntu10
> TIMEUNE
visualize it.
Cell 1of1
n6, Col8 LF
7:07 PM
5/23/2023
> OUT UNE
> WSL: Ubuntu10
> TIMELNE
Ln6, Col8 LFCell1of1
5/23/2023
7:07 PM
> OUTUNE 但其余的都是我自己的代码.
> WSL: Ubuntu10
> TIMELNE
But all the rest is my own code.
Ln6, Col8 LF Cell1of 1
ENG
5/23/2023
7:07 PM
> OUT UNE
> WSL: Ubuntu10
> TIMELNE
Ln6, Col8 LFCell1of1
5/23/2023
7:07 PM
> OUT U NE 所以让我们导入它.
> WSL : Ubuntu > TIME LNE
1 A0
So let'simportit.
Ln6, Col 8 LF Cell1of1
ENG
5/23/2023
7:07 PM
> OUT U NE 好的, 让我们导入所有这些, 当然, 当你在电脑上运行
> WSL : Ubuntu > TIME LNE okay let'simport all of these and of course you will have to install this particular Cell 2of2
ENG
7:08 PM
/23/2023
代码时, 你需要安装这个特定的库. 我们还要定义设备.
> OUT U NE
> WSL : Ubuntu > TIME LNE okay let'simport all of these and of course you will have to install this particular Cell 2of2
ENG
/23/2023
7:08 PM
代码时, 你需要安装这个特定的库. 我们还要定义设备.
> OUT U NE
> WSL : Ubuntu > TIME LNE library when you run the code on your computer let'salso define the device you can a2ofz
7:08 PM
/23/2023
175 decoder _output =
model. decode (encoder _output,
deco
176
oroj_output =
> OUTUNE 你可以直接从这里复制代码, 然后我们加载模型, 我们可以
WSL: Ubunt00
> TIMELNE
library when you run the code on your computer let's also define the device you can ENG
5/23/2023
7:08 PM
174
175
encoder _output =model. encode (encoder _input, encoder _mask )#( B, Seq _ Len, d_model )
decoder _output =model. decode (encoder _output, encoder _mask, decoder _input,
176
oroj_output=
deco
> OUTUNE 你可以直接从这里复制代码, 然后我们加载模型, 我们可以
> TIME LNE just copy the code from here and then we load the model u
5/23/2023
7:08 PM
> OUT UI NE 你可以直接从这里复制代码, 然后我们加载模型, 我们可以
> WSL : Ubuntu > TIMELINE Cell 3 of 3
Ln3, Col 1
7:08 PM
/23/2023
> OUT UI NE 你可以直接从这里复制代码, 然后我们加载模型, 我们可以
> WSL: Ubuntu
> TIMEUNE
ENG
7:08 PM
PREDIc TED :-Ecco, ma poi, che cosa ne pensano i contadini?
> OUT UINE 像这样从这里复制好的·让我们粘贴到这里, 这个变成了
> WSL: Ubuntu
> TIMEUNE
7:08 PM
> OUT U NE 像这样从这里复制:好的, 让我们粘贴到这里, 这个变成了
> WSL : Ubuntu > TIMELINE vocabulary source and vocabulary target Cell 3of3
n3. Col35
7:11 PM
5/23/2023
> OUT U NE 词汇源和词汇目标.
> WSL : Ubuntu > TIMELINE 02 O4fles to analyze vocabulary source and vocabulary target Ln3, Col36 Spaces :4 LFCell3of3
5/23/2023
7:11 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
5/23/2023
7:11 PM
好的, 现在让我们创建一个加载批次的功能.
> OUT U NE
> WSL : Ubuntu > TIMELINE Okay, now let's make a function to load the batch.
es4 LFCell4of4
Ln8, Col49 Spac
ENG
7:11 PM
5/23/2023
> OUT UI NE
> WSL: Ubuntu
> TIMELNE
Ln8. Col49 Spaces :4 LFCell4of4
ENG
5/23/2023
7:11 PM
encoder Pytho r 我现在将使用tokenizer 将批次转换为标记.
> OUT U NE
> WSL : Ubuntu > TIME L NE
Iwill convert the batch into tokens now using the tokenizer.
ENG
5/23/2023
7:12 PM
encoder.
[]encoder _input []encoder _mask Python > OUT U NE
WSL: Ubuntu
> TIMELNE
Ln8, Col49 Spaces 4 LF Cell4of4
ENG
5/23/2023
7:12 PM
θ]. cpu (). 当然, 一对于解码器, 我们使用目标词汇, 民 即目标tokenizer.
Pytho r
> OU TUNE
> WSL Ubuntu > TIMELINE and of course for the decoder we use the target vocabulary so the target tokenizer souf4
7:13 PM
/23/2023
> OUT UNE 所以让我们使用我们的贪心解码算法进行推理
Pytho r
> WSL : Ubuntu > TIMELINE and of course for the decoder we use the target vocabulary so the target tokenizer souf4
7:13 PM
/23/2023
> OUT UNE 所以让我们使用我们的贪心解码算法进行推理
Pythor
WSL: Ubuntu
> TIMELNE
let's just infer using our greedy decode algorithm so we provide the model Cell 4of4
/23/2023
7:13 PM
(model : Any, 所以我们提供模型,
izer_tgt: Any,
> OUTUNE 我们返回所有这些信息.
WSL: Ubuntu > TIMELINE ec4 LFCell4of4
Pythc
Ln 8, Col49 Spa
7:14 PM
5/23/2023
er_input[o]. cpu().
> OUTUNE 所以我们提供模型, 我们返回所有这些信息.
WSL: Ubuntu > TIME LNE we return all this information o knowl will build the necessary functions to 4 LFCell4of4
5/23/2023
7:14 PM
好的, 现在我将构建必要的功能来可视化注意力.
> OUT U NE
WSL : Ubuntu > TIME LNE visualize the Cell 5of5
n8. Col49
7:17 PM
5/23/2023
> OUT UINE 我将从另一个文件复制一些函数, 因为实际上我们要构建的
WSL: Ubuntu00
> TIMELNE
visualize the attention i will copy some functions from another file because actually elsors
/23/2023
7:17 PM
> OUT UI NE 我将从另一个文件复制一些函数, 因为实际上我们要构建的
> WSL : Ubuntu > TIMELINE △what we are going to build is nothing interesting from a learning point of view for calsors
ENG
/23/2023
7:17 PM
> OUT UI NE 内容从学习角度来看并不有趣, 就深度学习而言, 它主要是
> WSL: Ubuntu
> TIMELNE
with
In8. Col49 S
Cell 5of 5
5/23/2023
7:17 PM
> OUT UI NE 内容从学习角度来看并不有趣, 就深度学习而言, 它主要是
> WSL : Ubuntu > TIME LNE regards to the deep learning it's mostly functions to visualize the data so i will Cell 5of 5
ENG
/23/2023
7:17 PM
> OUT UI NE 用于可视化数据的函数. 所以我会复制它, 因为写起来相当
> WSL : Ubuntu > TIMELINE copy it because it's quite long to write and the salient part i will explain of 4 LFCell5of5
/23/2023
7:17 PM
长, 当然, 我会解释其中的关键部分.
> OUT UI NE
> WSL : Ubuntu > TIMELINE course Ln8. Col49 Spaces 4 LFCell5of5
5/23/2023
7:18 PM
row _tokens,
col _tokens,
> OUT UI NE 这就是那个函数. 好的, 这个函数是做什么的呢?
> WSL : Ubuntu > TIMELINE and this is the function okay what does this function do basically we have the.
LFCell5ofs
5/23/2023
7:18 PM
return att n[o, head ]. data 基本上, 我们会从编码器那里得到注意力.
> OUT UI NE
WSL : Ubuntu > TIMELINE and this is the function okay what does this function do basically we have the LFCell5of5 口
ENG
5/23/2023
7:18 PM
get _attn _map (attn _type, layer, head ), 如何从编码器获取注意力,
> OUT UI NE 例如, 我们在三个位置有注意力
WSL : Ubuntu > TIME LNE for example Cell 5 of 5
n 8, Col49
7:18 PM
/23/2023
get _attn _map (attn _type, layer, head ), 如何从编码器获取注意力, 例如, 我们在三个位置有注意力
> OUT UI NE
> WSL : Ubuntu > TIME LNE the attention we have in three positions first is in the encoder the second one is in csors
/23/2023
7:18 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 第一个在编码器中, 第二个在解码器的开始, 即解码器的
> WSL : Ubuntu > TIME LNE the decoder at the beginning of the decoder so the self attention of the decoder and asots 口
Z
7:18 PM
/23/2023
get _attn _map (attn _type, layer, head ), 自注意力, 然后我们有编码器和解码器之间的交叉注意力.
> OUT U NE
WSL : Ubuntu > TIMELINE then we have the cross attention between the encoder and the decoder Cell 5of5
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
max _sentence _len,
max _sentence _len,
> OUT UI NE
row _tokens,
> TIMELINE col_tokens,
WSL: Ubuntu
Ln8. Col49 Spaces 4 LFCell5of 5
ENG
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 所以我们可以可视化三种注意力. 如何获取关于注意力的
> WSL : Ubuntu > TIMELINE so we can visualize three types of attention how to get the information about the Cell 5of 5
?
Z
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
max _sentence _len,
nax _sentence _len, 信息.
> OUT UI NE
row _tokens,
> TIMELINE col _tokens,
> WSL : Ubuntu so we can visualize three types of attention how to get the information about. the Cell 5of5
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 我们加载模型, 我们有编码器, 我们选择要从哪个层获取
> WSL : Ubuntu > TIMELINE /23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 注意力, 然后从每个层我们可以获取自注意力块及其注意力
> WSL : Ubuntu > TIMELINE attention from and then from each layer we can get the self-attention block. and then srs
7:19 PM
/23/2023
get _attn _map (attn _type, layer, head ), 然后从每个层我们可以获取自注意力块及其注意力
> OUT UI NE 注意力,
WSL : Ubuntu > TIMELINE its attention scores where does this variable come from?
Cell5of 5
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
max _sentence _len,
max _sentence _len > OUT UI NE
row _tokens, 分数, 这个变量从哪里来?
> TIMELINE col _tokens,
WSL : Ubuntu its attention scores where does this variable come from?
Ln8. Col49 Spaces 4 LFCell5of 5
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 如果你记得当我们在这里定义注意力计算时, 当我们计算
WSL : Ubuntu > TIMELINE if you remember when we defined the attention calculation here here Cell 5of5
5/23/2023
7:19 PM
108
107
value = value. view (value. shape [e], value. shape [1], self. h, self. d_k). transpose(1, 2)
> OUT UI NE 注意力时我们不仅返回输出到下一层我们还给出这个
> WSL : Ubuntu > TIMELINE if you remember when we defined the attention calculation. here here o
4. 10. 6 (tra
5/23/2023
7:19 PM
107
108
value =value. view (value. shape [θ], value. shape [1], self. h, self. d_k). transpose(1, 2)
> OUT UI NE 注意力时我们不仅返回输出到下一层我们还给出这个
> WSL : Ubuntu > TIMELINE /23/2023
7:19 PM
108
107
value = value. view (value. shape [0], value. shape[1], self. h, self. d_k). transpose (1, 2) 注意力时, 我们不仅返回输出到下一层, 我们还给出这个
> OUTUINE
> WSL: Ubuntu > TIMELINE When we calculate the attention, we not only return the output to the next layer, we.
7:19 PM
108
107
value =value. view (value. shape [e], value. shape[1], self. h, self. d_k). transpose (1, 2)
109
110
self. attention _scores 注意力分数它是·soft max 的输出.
= Multi Head Attention Block. attention (query, key, value, mask, self. dropout )
> OUTUINE
111
> TIMELINE also give this attention scores, which is the output of the soft max. pton siosrtans
112
WSL: Ubuntu ENG
5/23/2023
7:19 PM
108
107
value =value. view (value. shape [θ], value. shape[1], self. h, self. d_k). transpose (1, 2)
109
110
x, self. attention _scores = Multi Head Attention Block. attention (query, key, value, mask, self. dropout )
> OUT UNE
111
112
#( Batch, h, Seq_ Len, d_k)-->( Batch, Seq_ Len, h, d_ K)-->( Batch, Seq_ Len, d_model )
> TIMELINE 113
x=x. transpose(1, 2). contiguous (). view (x. shape[0],-1, self. h*self. d_k)
> WSL: Ubuntu Ln 96, Col 60 (16 selected ) Spaces :4 UTF-8 LF{ Python 3. 10. 6 (trans
ENG
5/23/2023
7:19 PM
116
> OUTUNE 我们将其保存在这个变量中, self. attention Scores.
> WSL : Ubuntu > TIMELINE And we save it here in this variable, self. attention Scores. urs u 4 ython
4. 10. 6 (trar
7:19 PM
5/23/2023
116
118
119
class Residual Connection (n. Module):
> OUTUIN E
120
def_init_(self, dropout :float )-> No ne:
WSL : Ubuntu > TIMELINE 121
super(). init_()
Ln 109, Col33(21 selected ) Spaces 4 UTF-8 LF{↓ Python 3. 10. 6(trans
ENG
5/23/2023
7:19 PM
get _attn _map (attn _type, layer, head ),
> OUT UI NE 现在我们可以直接获取并可视化它.
WSL : Ubuntu > TIMELINE Now we can just retrieve it and visualize it.
merc on ida) Cell 5of 5
ENG
5/23/2023
7:19 PM
y=alt. Y("row_token", axis=alt. Axis(title =")),
color a "value ",
tooltip =["row ","column ","value ","row _token ","col _token "],
> OUT UI NE properties (height =4e0, width=400, title=f Layer {layer } Head [head ))
> TIME UNE. interactive ()
WSL: Ubuntu
Ln 20, Col15 Spaces :4 LFCell5of5
ENG
5/23/2023
7:19 PM
color ="value > OUT U NE 所以这个函数将根据我们想要从哪一层和哪个头获取注意力
WSL : Ubuntu > TIME UNE
ENG
5/23/2023
7:19 PM
color m "value ",
tooltip =["row ","column ",
> OUT UNE properties (height =4eo, width 选择正确的矩阵.
> TIMELINE interactive ()
> WSL : Ubuntu from which head will select the the matrix the correct matrix this function build sacalsofs
5/23/2023
7:20 PM
mark _rect ()
> OUT U NE 这个函数构建一个数据框来可视化信息, 即从该矩阵中提取的
> WSL : Ubuntu > TIMELINE /23/2023
7:20 PM
mark _rect ()
> OUT U NE 这个函数构建一个数据框来可视化信息, 即从该矩阵中提取的
WSL : Ubuntu > TIMELINE tooltip =["row ","colum data frame Cell 5 of 5
0
/23/2023
7:20 PM
mark _rect () 这个函数构建一个数据框来可视化信息, 即从该矩阵中提取的
> OUT UI NE
> WSL : Ubuntu > TIMELINE ENG
/23/2023
7:20 PM
. encode (
mark _rect ()
x=alt. x("col_token", axis=alt. A
y=alt. Y("row_token", axis=alt. 标记和分数.
> OUTUINE
color ="value "
> WSL : Ubuntu > TIME UNE
5/23/2023
7:20 PM
. mark _rect ()
> OUT UI NE 在这里, 我们将从这个矩阵中提取行和列, 然后我们还会
> WSL : Ubuntu > TIME UNE here so it will this matrix we extract'the row and the column oockd
Cell 5of 5
/23/2023
7:20 PM
mark _rect ()
> OUT U NE 在这里, 我们将从这个矩阵中提取行和列, 然后我们还会
WSL : Ubuntu > TIMELINE Cell 5of 5
n. 7, Col30
7:20 PM
/23/2023
properties (height =4e0, width=4eo, title=f" Layer [layer } Head {head }")
interactive ()
> OUT U NE 在这里我们将从这个矩阵中提取行和列, 然后我们还会
WSL : Ubuntu > TIMELINE and then we also build the chart the chart is um built with altair and what we will Cell 5 of 5
/23/2023
7:20 PM
for head in heads :
charts. append (alt. hconcat (*row charts )
row Charts. append (attn _map (attn _type, layer,
ow _tokens, col _tokens, max _sentence _len ))
return al t. vconcat (*charts ) 构建图表
> OUT UNE Pytho r
WSL : Ubuntu > TIMELINE and then we also build the chart the chart is um built with altair and what we will Cell 5of 5
5/23/2023
7:20 PM
for head in heads :
nc e_len))
> OUT UNE 图表是用 Altair 构建的, 实际上我们要做的是获取所有头的
Python WSL : Ubuntu > TIMELINE and then we also build the chart the chart is um built with altair and what we will Cell 5of 5
7:20 PM
Python > OUT UI NE 图表是用 Altair 构建的, 实际上我们要做的是获取所有头的
> WSL : Ubuntu > TIME LNE build actually is we will get the attention for all the we i built this method to get r
Cell5of 5
7:20 PM
Python > OUT UI NE 注意力, 我构建了这个方法来获取我们作为输入传递给这个
> WSL : Ubuntu > TIME UNE the Cell 5of 5
n55, Col32
7:20 PM
Python > OUT UI NE 注意力, 我构建了这个方法来获取我们作为输入传递给这个
> WSL : Ubuntu > TIME UNE attention for all the heads and all the layers that we pass to this function as input Cell 5of 5
7:20 PM
/23/2023
properties (height =4e0, width=4eo, title=f" Layer [layer } Head {head }")
interactive ()
> OUT UI NE 函数的所有层和所有头的注意力. 所以现在让我运行这个
> WSL : Ubuntu > TIME LNE
rlayer in layers :
so let me run this cell now Cell 5of 5
n 55, Col32
7:20 PM
/23/2023
get _attn _map (attn _type, layer, head ),
row _tokens,
max _sentence _len,
nax _sentence _len, 单元格.
> OUTUINE
col_tokens,
WSL: Ubuntu > TIME LNE so let me run this cellnow
Ln 55, Col32 Spaces 4 LFCell5of5
5/23/2023
7:20 PM
好的, 让我们创建一个新单元格, 然后运行它.
> OUT UI NE
WSL : Ubuntu > TIME LNE okay let's create a new cell and then let'sjust run it okay first we want to.
es 4 LFCell5of5
ENG
5/23/2023
7:20 PM
> OUT U NE 好的, 首先我们想要可视化我们正在处理的句子, 目 即批次
> WSL : Ubuntu > TIME LNE okay let'screate a new cell and then let'sjust run it okay first we want to Cell6of6
7:20 PM
5/23/2023
> OUT UI NE 好的, 首先我们想要可视化我们正在处理的句子, 即批次
> WSL: Ubuntu
> TIMELNE
5/23/2023
7:20 PM
> OUTUNE 好的, 首先我们想要可视化我们正在处理的句子, 民 即批次
WSL: Ubuntu0
> TIMELINE load a batch and then we visualize what is the source and the target s
Cell6of6
/23/2023
7:21 PM
> OUT U NE 好的, 首先我们想要可视化我们 正在处理的句子, 即批次
WSL : Ubuntu > TIME LNE
Cell 6of 6
n 50, Col 25
7:21 PM
> OUT U NE 其他输入标记, 所以我们加载一个批次, 然后我们可视化
WSL : Ubuntu > TIME LNE Cell 6of 6
n 50, Col 25
7:21 PM
5/23/2023
for head in heads :
row Charts. append (attn _map (attn _type, 1
charts. append(alt. hcon 源和目标, 还有目标.
> OUTUNE return al t. vconcat (*chart Pyt ho
WSL: Ubuntu
> TIMELNE
Ln50, Col25 Spaces 4 LF Cell6of6
5/23/2023
7:21 PM
> OUT U NE
WSL : Ubuntu > TIME UNE
Ln 50, Col25 Spaces:4 LFCell 6of 6
ENGe
5/23/2023
7:21 PM
> OUT U NE 最后我们还计算了长度.
> WSL : Ubuntu > TIME LNE And finally we calculate also the length.
Ln 50, Col25 Spaces 4 LFCell6of6
5/23/2023
7:21 PM
> OUT U NE
> WSL: Ubuntu
> TIMELNE
Ln50, Col25 Spaces 4 LFCell6. of6
5/23/2023
7:21 PM
长度是什么?
> OUT U NE
> WSL : Ubuntu > TIMELINE 10 O7fles to analyze What is the length?
Ln 50, Col25 Spaces :4 LF Cell6 of6
5/23/2023
7:22 PM
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 50, Col25 Spaces :4 LFCell6of6
ENG
5/23/2023
7:22 PM
> OUT UNE 好的, 基本上是所有在填充字符之前的字符, 即第一个填充
WSL: Ubuntu0
> TIMEUNE
Okay, it's basically all the characters that come before the padding character, so: celor6
ENG
/23/2023
7:22 PM
> OUT U NE 好的, 基本上是所有在填充字符之前的字符, 即第一个填充
> WSL : Ubuntu > TIME UNE the first occurrence of the padding character.
Cell 6of6
n50, Col25
7:22 PM
/23/2023
字符的出现.
> OUT U NE
> WSL Ubuntu > TIME UNE the first occurrence of the padding character.
Ln 50, Col25 Spaces 4 LFCell6 of6
5/23/2023
7:22 PM
> OUT U NE
> WSLUbuntu
> TIMEUNE
Ln 50, Col25 Spaces :4 LFCell6 of6
ENG
5/23/2023
7:22 PM
> OUT U NE 因为这是从数据集中提取的批次, 已经是为训练构建的张量
> WSL Ubuntu > TIME UNE Because this is the batch taken from the dataset, which is already the tensor built Cell 6of6
/23/2023
7:22 PM
> OUT U NE 因为这是从数据集中提取的批次, 已经是为训练构建的张量
WSL : Ubuntu > TIME UNE for training, so they already include the padding.
Cell 6of6
Ln 50, Col 25
7:22 PM
/23/2023
> OUT U NE 所以它们已经包含了填充.
> WSL : Ubuntu > TIME UNE for training, so they already include the padding.
Ln 50, Col25 Spaces:4 LFCell6of 6
ENGe
5/23/2023
7:22 PM
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln50, Col25 Spaces :4 LFCell6of6
ENG
5/23/2023
7:22 PM
在我们的例子中, 我们只想获取句子中实际字符的数量.
> OUT U NE
> WSL : Ubuntu > TIME UNE In our case, we just want to retrieve the number of actual characters. in. our es
Cell6of6
5/23/2023
7:22 PM
> OUT U NE 在我们的例子中, 我们只想获取句子中实际字符的数量.
WSL : Ubuntu > TIME LNE sentence.
Cell 6of6
n 50, Col25
7:22 PM
5/23/2023
> OUT U NE 所以这个, 我们可以, 句子中实际单词的数量, 所以我们
> WSL : Ubuntu > TIME LNE So this one, we can, the number of actual words in our sentence, so we can check. the 6f6
/23/2023
7:22 PM
> OUT U NE
> WSL Ubuntu > TIMELINE Ln50, Col25 Spaces:4 LFCell6of 6
ENG
5/23/2023
7:22 PM
> OUT U NE 所以让我们运行这个, 这里有一些问题, 我忘了这个函数是
> WSL : Ubuntu > TIMELINE so let'srun this one and there is some problem here l forgot to this function was Cell 7of7
ENG
5/23/2023
7:22 PM
错误的, 所以现在应该可以了. 这个句子太短了, 让我们
> OUT U NE
> WSL : Ubuntu > TIME UNE so let'srun this one and there is some problem here I forgot to this function was u
Cel6of7 口
ENG
5/23/2023
7:22 PM
> OUT UI NE 找一个更长的. 好的, 让我检查一下质量. 你不能保持
> WSL : Ubuntu > TIME LNE wrong so now it should work okay this sentence is too small let's get a longer one Cell 5of 7
7:22 PM
/23/2023
> OUT UI NE 找一个更长的. 好的, 让我检查一下质量. 你不能保持
WSL : Ubuntu > TIMELINE okay let me check the quality n 50, Col 25
Cell 7of 7
7:23 PM
/23/2023
> OUT U NE 现状, 尤其是你.
WSL : Ubuntu > TIMELINE okay let me check the quality Ln 50, Col25 Spaces :4 LFCell7of 7
5/23/2023
7:23 PM
现状, 尤其是你.
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 50, Col25 Spaces:4 LFCell7of 7
ENG
5/23/2023
7:23 PM
> OUT U NE 现状, 尤其是你.
WSL : Ubuntu > TIMELINE You can not remain as you are, especially you.
Ln 50, Col25 Spaces :4 LFCell7of 7
5/23/2023
7:23 PM
> OUT U NE
WSL : Ubuntu > TIMELINE Ln 50, Col25 Spaces:4 LFCell7of 7
ENG
5/23/2023
7:23 PM
> OUT U NE 我们不能, 你们不能继续这样, 尤其是现在.
WSL : Ubuntu > TIME LIN E
Noinon possiamo, voi non potete continuar e a stare cosi, special men te ora. ucel7o7
5/23/2023
7:23 PM
好的, 看起来不错.
> OUT U NE
WSL : Ubuntu > TIMELINE Ok, looks not bad.
Ln 50, Col25 Spaces 4 LFCell7of7
5/23/2023
7:23 PM
> OUTUNE 好的, 让我们打印第0、1、2层的注意力, 因为
> WSL: Ubuntu > TIMELINE okay let'sprint the attention for the layers let'ssay zero one and two because we Cell 7of 7
5/23/2023
7:23 PM
> OUT U NE 所以我们只可视化三层, 并且我们会可视化所有头. 每层有
> WSL : Ubuntu > TIMELINE have six of them if you remember the parameter is nis equal to six so we wil I just Cell 7of 7
7:23 PM
> OUT U NE 所以我们只可视化三层, 并且我们会可视化所有头. 每层有
WSL : Ubuntu > TIMELINE Visualize Cell 7of 7
7:23 PM
/23/2023
> OUT U NE 所以我们只可视化三层, 并且我们会可视化所有头. 每层有
> WSL : Ubuntu > TIMELINE o △&three layers and we will visualize all the heads we have eight of them for each layer celi7o7
7:23 PM
> OUTUNE 八个头. 所以头编号:0、1、2、3
3、4、5
> WSL: Ubuntu > TIME UNE three layers and we will visualize all the heads we have eight of them for each layer cel7o7
/23/2023
7:23 PM
> OUTUNE 八个头. 所以头编号:0、1、2、3
3、4、5
> WSL: Ubuntu > TIME UNE so the head number zero one two three four five six seven and seven s
4 LFCel7of7
5/23/2023
7:23 PM
> OUT U NE
WSL Ubuntu > TIMELINE Ln 50, Col25 Spaces4 LFCell7of 7
ENG
7:23 PM
5/23/2023
> OUT U NE 好的, 首先让我们可视化编码器自注意力, 我们确实得到了
> WSL : Ubuntu > TIMELINE okay let'sfirst visualize the encoder self attention and we do get all attention Cell 7of7
7:23 PM
/23/2023
get _ipython > OUT U NE 所有注意力图, 我们想要哪一个. 所以编码器的一个
WSL : Ubuntu > TIME LNE okay let'sfirst visualize the encoder self attention and we do get all attention LFCell7of7
/23/2023
7:23 PM
> OUT UNE 所有注意力图, 我们想要哪一个. 所以编码器的一个
WSL: Ubuntu
> TIMELNE
7:23 PM
我们想要这些层和这些头.
> OUT UNE
> WSL: Ubuntu
> TIMELNE
Z
7:23 PM
> OUT U NE 原始标记是什么, 编码器输入标记, 我们在列中想要什么?
WSL : Ubuntu > TIME LNE and what are the raw tokens the encoder input tokens Cell 7of 7
Ln 50, Col25
7:23 PM
/23/2023
> OUT U NE 原始标记是什么, 编码器输入标记, 我们在列中想要什么?
WSL : Ubuntu > TIME UNE And what do we want in the column?
n 50, Col 25
Cell 7 of 7
7:23 PM
/23/2023
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln 50, Col25 Spaces :4 LFCell7of 7
5/23/2023
7:24 PM
因为我们打算构建一个网格.
> OUT U NE
WSL : Ubuntu > TIME UNE Because we are going to build a grid.
Ln 50, Col25 Spaces :4 LFCell7of 7
5/23/2023
7:24 PM
> OUT U NE
WSL: Ubuntu
> TIMEUNE
Ln 50, Col25 Spaces :4 LFCell7of 7
5/23/2023
7:24 PM
> OUT U NE 所以如你所知, 注意力是一个将行与列相关联的网格.
WSL : Ubuntu > TIME UNE So as you know, the attention is a grid that correlates rows with columns.
Cell 7of 7
7:24 PM
/23/2023
> OUT U NE
WSL: Ubuntu
> TIMELNE
Lni50, Col25 Spaces 4 LFCell7of7
5/23/2023
7:24 PM
> OUT U NE 在我们的例子中, 我们谈论的是编码器的自注意力.
WSL : Ubuntu > TIME LNE In our case, we are talking about the self-attention of the encoder. o2s
Cell7of 7
7:24 PM
5/23/2023
> OUT U NE 所以这是同一个句子在自我关注,
WSL : Ubuntu > TIME LNE So it's the same sentence that is attending itself.
Ln 50. Col25 Spaces4 LFCell7 of 7
ENG
5/23/2023
7:24 PM
> OUT U NE 所以我们需要在行和列上都提供编码器的输入句子.
> WSL : Ubuntu > TIME UNE So we need to provide the input sentence of the encoder on both the rows and the celio7
/23/2023
7:24 PM
> OUT U NE 所以我们需要在行和列上都提供编码器的输入句子
> WSL : Ubuntu > TIME UNE columns.
Cell 7of 7
7:24 PM 口
ENG
/23/2023
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 50. Col25 Spaces :4 LFCell7of7
5/23/2023
7:24 PM
> OUT U NE 我们想要可视化的最大长度是多少?
> WSL : Ubuntu > TIME UNE And what is the maximum number of length that we want to visualize?
4 LFCel7of7
5/23/2023
7:24 PM
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
5/23/2023
7:24 PM
> OUT UNE 好的, 假设我们想要可视化不超过20个, 所以是20
> WSL: Ubuntu
> TIMEUNE
okay let's say we want to visualize no more than 20 so the minimum of 20and sentence o7
7:24 PM
/23/2023
> OUT U NE 和句子长度的最小值.
> WSL: Ubuntu
> TIMEUNE
5/23/2023
7:24 PM
> OUT U NE 和句子长度的最小值.
> WSL : Ubuntu > TIME UNE length Ln 50, Col 25 Space
7:24 PM
5/23/2023
> OUT U NE
> WSL: Ubuntu
> TIMEUNE
Ln 50, Col25 Spaces :4 LFCell 7 of7
5/23/2023
7:24 PM
007 as
006 r
007 a
007 ar
008 you 好的, 这是我们的可视化结果.
008 yo
> OUTUNE
009 an
009 an
WSL: Ubuntu > TIME UNE len we
001 W
001 W
002 canne 我们可以看到,
002
> OUTUNE 正如我们所预期的, 实际上, 当我们
> WSL : Ubuntu > TIME UNE visualize the attention we expect the values along the diagonals to be hig
hbecause ell8of8
5/23/2023
7:24 PM
002 car
001 we
001 W
001 We
> OUTUNE 可视化注意力时, 我们预期对角线上的值会很高, 国 因为这是
WSL : Ubuntu 00
> TIMEUNE
5/23/2023
7:24 PM
002 can
001 We
001 W
001 We 每个标记与其自身的点积. 我们还可以看到其他有趣的关系
> OUTUNE
> WSL Ubuntu > TIMELINE product of each token with itself and we can see also that there are other 5/23/2023
7:24 PM
002 can e
001 We
001 W
002 cann 例如, 假设句子开始标记和句子结束标记,
002 ci 至少对于头0
> OUTUINE
> WSL: Ubuntu > TIMELINE interesting relationship for example 5/23/2023
7:24 PM
002 ca
001 we
002 cann
001 We
002
> OUTUNE 和层0, 它们与其他词没有关联, 就像我实际预期的
> WSL: Ubuntu
> TIMELNE
say that the start of sentence token and the end of sentence token at least for the 5/23/2023
7:25 PM
002 can 001 we
001 W
002 canr 那样, 但其他头确实学习了一些非常小的映射, 我们可以
002
> OUTUNE
> WSL: Ubuntu > TIMELINE head zero and the layer zero they are not related to other words like i would expect ell8of 8 口
5/23/2023
7:25 PM
002 cane
001 We
002 ca
001 We
002 canne
001 We
003
003
> OUTUNE
004 Yo
004 Yoo
> WSL: Ubuntu > TIME UNE we hover over r
ell8of8 口
5/23/2023
7:25 PM
002 car
001 We
001 W
001 We 如果我们悬停在每个网格单元格上
002 ci 我们可以看到自注意力的
> OUT U NE
> WSL : Ubuntu > TIME UNE and but other heads they do learn some very small 口
5/23/2023
7:25 PM
002 can 001 We
001 W
001 We 实际值, 例如自注意力的分数. 我们可以看到这里的注意力
002 can
> OUT UI NE
> WSL : Ubuntu > TIME LNE of the self-attention 5/23/2023
7:25 PM
002 care
001 We
002
001 W
002 can n
001 We
003 非常强.
003
> OUTUNE
004 Yo
004 Yo
> WSL: Ubuntu
> TIMEUNE
for example we can see the attention is very strong here so the word l8of8
5/23/2023
7:25 PM
001 W
001 We
002 ca
002
002 ca
> OUTUNE 所以"especially " 和"specially 这两个词是相关的, 所以这是同一个词与
> WSL : Ubuntu > TIME LNE for example we can see the attention i is very strong here so the word especially and ell8of8
5/23/2023
7:25 PM
002 ca
001 we
001 W
001 We
002 我们可以为所有层
> OUTUNE 自身的关联, 但还有"especially 和
"now > WSL : Ubuntu > TIME L NE
thitself but also especi
randi
now
5/23/2023
7:25 PM
002 ca
001 we
001 W
001 We 自身的关联,
002
> OUTUNE now 我们可以为所有层
> WSL : Ubuntu > TIME LNE
5/23/2023
7:25 PM
002 car
001 we
001 W
001 We
002 我们可以为所有层
> OUT UI NE 自身的关联, 但还有"especially 和
now > WSL : Ubuntu > TIME LNE lavers so because each head Cell 8of8
5/23/2023
7:25 PM
> OUTUINE
ayer2 He 视化这种注意力
ayer2 Hea
Layer 2 Head 2
> WSL: Ubuntu
> TIMELNE
and we can visualize this kind of attention for all re rs so because each head 5/23/2023
000 [ SC
ayer2 Head0
Layer 2 Head 2 所以因为每个头会观察每个词的不同方面, 因为我们均匀地
> OUTUINE
WSL: Ubuntu00
> TIMELINE and we can visualize this kind of attention layers so because each. head 口
5/23/2023
7:25 PM
分配了词嵌入到各个头, 所
> OUT UI NE 所以每个头 头会看到词嵌入的不同
Layer 1 Head 2
> WSL: Ubuntu00
> TIME LNE
000 [ SOS 口
5/23/2023
7:25 PM
分配了词嵌入到各个头, 所
> OUT UINE 所以每个头会看到词嵌入的不同
La yer 1 Head 2
xwsuuntu o Aamong the heads equally so each head > TIME LNE
100 [ SOS
bedding. oi
8of8
5/23/2023
7:25 PM
> OUT UI NE Layer 1 Head 0 阝分
Layer 1 Head 2
> WSL: Ubuntu > TIMELINE 8of8
7:25 PM
> OUT UI NE Layer 1 Head 0
Layer 1 Head 2
> WSL: Ubuntu > TIMELINE between t
hewords and 5/23/2023
7:25 PM
我们也希望它们学习不同类型的词间映射一一
> OUTUINE 这实际上是
Layer 1 Head 2
> WSL: Ubuntu > TIMELINE the word we also hope that they learn di he words and 8of8 口
5/23/2023
7:25 PM
我们也希望它们学习不同类型的词间映射一一 这实际上是
> OUTUINE
Layer 1 Head 2
> WSL: Ubuntu > TIMELINE this is actual l
vthecase 口
5/23/2023
7:25 PM
间的不同映射. 我们还有不同的
> OUTUINE
Layer 1 Head 2
> WSL: Ubuntu > TIMELINE hecase
5/23/2023
7:25 PM
事实一一以及层与层之间的不同映射. 我们还有不同的
> OUT UI NE
> WSL : Ubuntu > TIMELINE and between one layer and the next we also have different Q, W, K and W, Vmetricsso7
/23/2023
7:26 PM
Q、 W、 K和 WV 矩阵, 所以它们也应该学习
> OUTUNE
TIMELINI
and betweenone layer and thenextwe also have different Q, W, K and W, Vmetrics so
> WSL: Ubuntu
LO
7:26 PM
Q、 W、 K和 WV矩阵, 所以它们也应该学习
> OUTUNE
TIMELINI
they should also learn different relationships now we can also want we may also want to > WSL: Ubuntu
LO
7:26 PM
008 yoi
009 ae 不同的关系.
> OUTUNE
011 espe
> WSL : Ubuntu > TIMELINE. they should also learn different relationshi ips now we can also want we may also. want to 5/23/2023
7:26 PM
Python > OUT U NE 现在我们可能还想可视化解码器的注意力. 所以让我们来做
> WSL : Ubuntu > TIMELINE they should also learn different relationships now we can also want we may also want to 7:26 PM
> OUT U NE 现在我们可能还想可视化解码器的注意力. 所以让我们来做
> WSL : Ubuntu > TIME LNE visualize the attention of the decoder so let's do it let me just copy the code and Cell7of8
ENG
/23/2023
7:26 PM
002 can ng
002 cat
002 can no
004 Yo
003
003
005 canno 让我复制代码并更改参数.
005 cannot 004 Yo
> OUTUNE
> WSL: Ubuntu > TIME UNE
ell 7of 8 口
7:26 PM
5/23/2023
012 noe
013
013
013
> OUTUNE
014 [ EOS WSL : Ubuntu > TIME LNE
5/23/2023
7:26 PM
> OUTUNE 好的, 这里我们想要解码器1, 我们想要相同的层等
> WSL: Ubuntu > TIMELINE okay here we want the decoder one we want the same layers etc but the tokens that we s
/23/2023
7:26 PM
但行和列上的标记将是解码器标记.
> OUT UI NE
> WSL : Ubuntu > TIMELINE will be on the rows and the columns are the decoder tokens so decoder input to kensaaofs
5/23/2023
7:26 PM
所以解码器输入标记和解码器输入标记.
> OUTUNE
WSL: Ubuntu00
> TIMELINE and decoder Ln 50. Col25 S
Cell8of 8
5/23/2023
7:26 PM
> OUT U NE 所以解码器输入标记和解码器输入标记.
> WSL : Ubuntu > TIMELINE input tokens let'svisualize and also we should see italian language now because we c&ors
5/23/2023
7:26 PM
> OUT UNE 让我们可视化一一而且我们现在应该看到意大利语, 因为
> TIME LNE input tokens let's visualize and also we should see italian language now because we alod&
/23/2023
7:26 PM
005 voi
005 v 让我们可视化 而且我们现在应该看到意大利语,
> OUTUNE
> TIMEUNE
are using the decoder self-attention 5/23/2023
我们使用的是解码器自注意力,
Layer 1 Head 0 确实是这样.
Layer1 H
> OUTUNE
WSL: Ubuntu00
> TIMELNE
7:26 PM
我们使用的是解码器自注意力,
er1 Head0
Layer1 H
> OUTUNE 确实是这样.
> WSL: Ubuntu > TIME LNE and it is so here we see a different kind of attention ENG
5/23/2023
7:27 PM
所以在这里我们看到解码器端不同的注意力类型,
Layer 1 Head 0
er1 Head 1
Layer 1 He
> OUTUNE 而且我们
WSL: Ubuntu00
> TIMELINE here we have multiple heads that should l ferent mapping and also different ll9of9 口
ENG
5/23/2023
:27 PM
还有多个头 它们应该学习不同的映射, 以及不同的层应该
> OUT UI NE
WSL : Ubuntu 00
> TIMELINE
layers
5/23/2023
> OUTUNE 还有多个头, 它们应该学习不同的映射, 以及不同的层应该
WSL: Ubuntu O0
> TIMELINE should learn different mappings between words the one i find most interesting is. the
学习词之间的不同映射. 我发现最有意思的是交叉注意力.
> OUT U NE
> WSL : Ubuntu > TIMELINE cross attention so let'shave a look at that okay let me just copy the code and run
Cell9of9
5/23/2023
7:27 PM
009 a
009 a
010 st 所以让我们来看一下. 好的,
> OUTUNE
> WSL: Ubuntu 00
> TIMELNE
itagain
5/23/2023
get _all _attention _maps ("decode
# De
Self-Attentio
1. 2s
Python Layer 0 He ad0
Layer 0 Head 1
Layer O He
> OUT UNE
000 [ SQ
000 [ SOS
> TIMELNE
n 50, Col2
Cell9of9
7:27 PM
5/23/2023
好的, 所以, 如果你记得这个方法, 它是编码器
> OUTUNE
WSL: Ubuntu00
> TIMELINE Okay so if you remember the method it's encoder decoder same layer so here on the asot9
5/23/2023
7:27 PM
> OUT UNE 所以在这里, 行上我们将显示编码器输入, 列上我们将显示
> TIME LNE Okay so if you remember the method it's encoder decoder same layer so here on the /23/2023
7:27 PM
> OUT U NE 所以在这里, 行上我们将显示编码器输入, 列上我们将显示
> WSL : Ubuntu > TIME LNE rows we will show the encoder input and on the columns we will show the decoder input 7:27 PM
解码器输入标记, 因为这是编码器和解码器之间的交叉
> OUTUNE
WSL: Ubuntu00
> TIMELNE
tokens In50, Col25
Cell9of9
7:27 PM
> OUTUNE 解码器输入标记, 因为这是编码器和解码器之间的交叉
WSL: Ubuntu0 O
> TIMELNE
because it's a cross attention between the encoder and the decoder. s
Cell 9of9
7:27 PM
> OUT U NE
WSL: Ubuntu00
> TIMELINE
Cell 9of 9
Ln50, Col25
7:27 PM
5/23/2023
007 a
007 as
008 yo
008 yo 好的, 这就是编码器和解码器之间交互的大致方式以及它是
> OUTUNE
WSL : Ubuntu > TIMELINE 5/23/2023
7:27 PM
005 can ng
004 Yot
005 cannot 005 cannc
004 Yo
006 rer 如何发生的.
> OUTUNE
> WSL : Ubuntu > TIMELINE 5/23/2023
7:28 PM
005 can s
004 Yot
005 cann
004 Yo
005 cannc
004 Yox 所以这就是我们找到交叉注意力的地方, 它是使用来自编码器
> OUTUNE
WSL: Ubuntu00
> TIMELNE
calculated using the 008 you ell10 of 10
7:28 PM
5/23/2023
005 can no
004 Yot
005 can nc
004 Yoi
005 cannc
004 Yo 的键和值计算的, 而查询来自解码器.
> OUTUNE
007 as
> WSL: Ubuntu > TIMELINE Cell 10 of 10
7:28 PM
ENG
5/23/2023
005 can n
004 Yot
005 cann
005 cannc
004 Yox 所以这实际上是翻译任务发生的地方, 这就是模型如何学习将
> OUTUINE
> WSL: Ubuntu > TIMELINE Cell 10 of 10
7:28 PM
Z
ENG
5/23/2023
014 [ EOS)
014 [ EOS
011especially 001 Noi
014 [ EOS 所以这实际上是翻译任务发生的地方, 这就是模型如何学习将
> OUTUINE
> WSL: Ubuntu > TIMELINE model learns to 5/23/2023
014 [ EOS}
014 [ EOS]
011especially 001 Noi 这两句话相互关联, 以实际计算翻译
> OUTUNE
> WSL: Ubuntu > TIME UNE model learns to Z
5/23/2023
014 [ EOS}
014 [ EOS]
001 Nol > OUTUNE 这两句话相互关联,! 以实际计
> WSL: Ubuntu > TIMELINE late the translation 5/23/2023
014 [ EOS
> OUTUINE
> WSL: Ubuntu00
> TIME LNE
5/23/2023
003
003
003.
004 Y
004 Yox 所以我邀请你们自己运行代码. 我给你们的第一个建议是
> OUTUNE
WSL : Ubuntu OO > TIMELINE so i invite you guys to run the code by yourself so the first suggestion i give you.
5/23/2023
012 now
eciall
013.
012 nov
013.
> OUT U IN E
014[ EOS) 和我一起看视频并编写代码
014 [ EOS
WSL: Ubuntu00
> TIMELINE is to write the code along with me. wit
5/23/2023
7:28 PM
012 nov
013 你可以暂停视频, 自己编写、运行代码. 好的, 让我给
> OUTUNE
> WSL: Ubuntu > TIMELINE 5/23/2023
11 especial l 012 noi
012 nov
013 你可以暂停视频, 自己编写、"运行代码. 好的 让我给
> OUTUINE
WSL: Ubuntu 00
> TIMEUNE
writerun
5/23/2023
012 now
012 nov
013
013.
> OUTUINE
014[ EOS] 你们一 些实际的例
014 [ EOS
> WSL: Ubuntu > TIME UNE write the code for by yourself okay let me vou some practical examples for 5/23/2023
7:28 PM
97
9
return (attention _scores @value ), attention _scores > OUT UI NE 例如, 当我编写模型代码时, 我建议你先看我编写一个特定
> WSL : Ubuntu > TIMELINE write the code for by yourself okay let me give you some practical examples for cm
ENG
/23/2023
7:28 PM
42
41
return self. dropout (x )
class Layer Normalization (nn. Module ): 层的代码, 然后暂停视频. 自己编写, 花点时间. 不要
> OUT UI NE
WSL : Ubuntu > TIMELINE self. alpha a meter (torch. ones (
particular layer T4. Col59
Pytho
7:28 PM
5/23/2023
42
41
return self. dropout (x )
class Layer Normalization (nn. Module ): 层的代码, 然后暂停视频. 自己编写, 花点时间. 不要
> OUT UI NE
> WSL : Ubuntu > TIMELINE and then stop the video write it by yourself take some time don't watch the solution.
5/23/2023
7:28 PM
42
41
return self. dropout(x)
44
class Layer Normalization (nn. Module ):
def _in it_(self, eps:float =1
super ()._in it _() 马上看解决方案.
> OUT UINE
47
46
self. eps=eps
> WSL: Ubuntu > TIME LNE and then stop the video write it by yourself take some time don't watch the solution 5/23/2023
7:29 PM
42
41
44
class Layer Normalization (nn. Module):
45
def_init_(self, eps:float = 马上看解决方案.
> OUTUINE
46
47
self. eps=eps
super ()._in it _()
> WSL Ubuntu > TIMELINE right away try to figure out what is going wrong and if you really can not after. one ENG
5/23/2023
7:29 PM
41
42
return self. dropout (x )
class Layer Normalization (nn. Module ): 尝试找出哪里出错了, 如果你真的在一两分钟后还是无法找出
> OUT U NE
> WSL : Ubuntu > TIMELINE right away try to figure out what is going wrong and if you really can not after one 7:29 PM
ENG
5/23/2023
42
41
return self. dropout (x )
class Layer Normalization (nn. Module ): 尝试找出哪里出错了, 如果你真的在一两分钟后还是无法找出
> OUTLINE WSL : Ubuntu > TIMELINE self. alpha = nn. Parameter (to rch. ones(1)) # Multi
two
n 10. Col37
Pytho
7:29 PM
5/23/2023
41 return self. dropout (x )
class Layer Normalization (nn. Module ): 尝试找出哪里出错了, 如果你真的在一两分钟后还是无法找出
> OUTLINE > WSL : Ubuntu > TIMELINE ENG
/23/2023
7:29 PM
42
41
return self. dropout(x)
44
class Layer Normalization (nn. Module):
45 当然, 有些事情你无法自己想出来,
> OUTUINE
46
> WSL: Ubuntu > TIME LNE the video but try to do it by yourself some things of course you. can not come up by 口
ENG
5/23/2023
7:29 PM
42
41
return self. dropout(x)
44
class Layer Normalization (nn. Module ):
> OUT UINE
47
> WSL: Ubuntu
> TIMELNE
self. alpha =nn. Parameter (to rch. ones(1))
yourself so Ln. 10, Col 37
ENG
7:29 PM
5/23/2023
42
41
r eturn self. dropout(x)
44
rmalization (nn. Module ):
> OUT UI NE 所以, 例如, 对于位置编码和所有这些计算.
> WSL : Ubuntu > TIMELINE for example for'the positional encoding and all this calculation it's basically just ENG
5/23/2023
7:29 PM
41
42
return self. dropout (x )
class Layer Normalization (nn. Module ): 这基本上只是公式的应用, 但重点是, 你至少应该能够自己
> OUT UI NE
WSL : Ubuntu > TIMELINE self. alpha = nn. Parameter (
an application of formulas Ln 29, Col. 89
Python 7:29 PM
5/23/2023
66
return self. linear_2(self. dropout (torch. relu(self. linear_1(x))))
#( Batch, Seq_ Len,
d_model )-->( Batch, Seq _ Len, d_ff)-->( Batch, Seq_ Len, d_model)
67
class Multi Head Attention Block (nn. Modul
69
68 构思出一个结构.
> OUTUNE
def
_in it_(self, d _model :int,
> WSL : Ubuntu > TIMELINE But the point is you'should at least be able to come with a structure by yourself.
5/23/2023
7:29 PM
66
67
return self. linear_2(self. dr c
ut(to rch. relu(self. linear_1(x))))
68
69
class Multi Head Atte 所以所有层是如何相互作用的
> OUTUINE
def
WSL: Ubuntu > TIMELINE So how all'the layers are interacting with each other. ufs u
ython 3. ios trans iomer ondl)
ENG
5/23/2023
7:29 PM
65
66
return self. linear_2(self. dropout (torch. relu(self. linear_1(x))))
#( Batch, Seq_ Len,
d_model ) -->( Batch, Seq _ Len, d_ff) -->( Batch, Seq_ Len, d_model)
6>
class Multi Head Attention Block (nn. Module ):
68
69
> OUTUIN E
70
71
d ef_init_(self, d_model :int, h:int, dropout :float )-> None :
super ()._in it _()
> WSL : Ubuntu > TIMELINE Ln 29, Col89 Spaces4 UTF-8 LF ( Python 3. 10. 6(transformer :conda)
ENG
5/23/2023
7:29 PM
65
66
#( Batch, Seq_ Len,
return self. linear _2(self. drop
d_model) -->( Batch, Seq_ Len, d_ff ) --> ( Batch, Seq _ Len, d_model)
sut (to rch. relu(self. linear_1(x)))
67
class Multi Head Attention Block (n
68
69 这是我的第一个建议.
> OUTUIN E
70
71
d ef
_init_(self, d_model :
super ()._in it _()
WSL : Ubuntu > TIME LNE This is my first recommendation.
Ln 29. Col89 Spaces:4 UTF-8 LF↓ Python 3. 10. 6(trans
ENG
7:29 PM 口
5/23/2023
65
66
#( Batch, Seq_ Len,
return self. linear _2(self. dropout(torch. relu(self. linear _1(x))))
d_model)-->( Batch, Seq_ Len, d_ff)-->( Batch, Seq_ Len, d_model)
68
class Multi Head Attention Block (nn. Module):
> OUTUIN E
69
70
d ef_init_(self, d_model :int, h:int, dropout :float )-> None :
> TIME LNE
71
super()._init_()
WSL: Ubuntu Ln 29. Col89 Spaces:4 UTF-8 LF ↓ Python 3. 10. 6 (transfc
ENG
5/23/2023
7:29 PM
#( Batch, Seq _ Len,
self. line a
ar_2(self. dr
ut(torch. relu(self. linear _1(x))))
> OUTUINE 而在训练循环方面, 训练部分实际上是非常标准的
WSL: Ubuntu
> TIMELNE
And while about the training loop, the training part actually is quite standard.
7:29 PM
5/23/2023
186
writer. flush()
188
189
> OUTUINE
190 所以它与其他你可能见过的训练循环非常相似
WSL: Ubuntu
> TIMELNE
So it's very similar to other training loops that you may have seen.. ython 3io6trsl
192
7:29 PM
ENG
5/23/2023
186 writer. flush ()
# Back propagate the loss 1oss. backward()
> OUTUINE
191
# Update the weights WSL: Ubuntu
> TIMELINE
192
optimizer. step()
Ln 165. Col37 Spaces 4 UTF-8 LF Python3. 10. 6(transformer :co
ENG
7:29 PM
5/23/2023
186 writer. flush()
loss 有趣的部分是我们如何计算损失.
> OUTUINE
191
WSL: Ubuntu > TIMELINE 192
ENG
7:29 PM
5/23/2023
writer. flush()
1oss. backward ()
# Back propagate the loss > OUTUINE
191
# Update the weights WSL: Ubuntu
> TIMEUNE
192
optimizer. step()
Ln 18t. Col 42(29 selected ) Spaces : 4 UTF-8 LF(↓ Python3. 10. 6(transformer :conda)
7:29 PM 口
5/23/2023
186
writer. flush()
187
gate the loss > OUTLINE 以及我们如何使用 Transformer 模型, 最后真正重要的是我们如何
> WSL : Ubuntu > TIMELINE and how We use the transformer model and the last thing that is really important is
/23/2023
7:29 PM
52
50
def
run_validation (model, validation _ds,
model. avai()
55
count = 进行模型推理, 这在贪心代码中
> OUTUINE
> WSL: Ubuntu > TIMELINE how we inference the model which is in this greedy code so thank you everyone for.
ENG
7:29 PM
decoder _input. squeeze (e ) 所以感谢大家观看视频并陪伴我这么长时间. 我可以保证这是
> OUT UI NE
> WSL : Ubuntu > TIME LNE how we inference the model which is in this greedy code so thank you everyone for.
ENG
/23/2023
7:29 PM
decoder _input. squeeze (e ) 所以感谢大家观看视频并陪伴我这么长时间. 我可以保证这是
> OUT UI NE
WSL : Ubuntu > TIME UNE odel. eval ()
watching the Ln46, Col 33
UTF-8 LFPytho
7:30 PM
/23/2023
decoder _input. squ
e eze(e) 值得的, 我希望在接下来的视频中, 能展示更多我熟悉的
> OUT UI NE
> WSL : Ubuntu > TIMELINE /23/2023
7:30 PM
48
49
eeze(e)
> OUTUINE
Transformer 和其他模型的例子, 并且我也想和你们一起探索,
> WSL : Ubuntu > TIME LNE hope in the next videos to make more examples of transformers and other models that i
/23/2023
7:30 PM
49
50
return decoder _input. squeeze (e)
51
> OUTUINE
52
53
def run_validation (model, validation _ds, tokenizer _src, tokenizer _tgt, max_len, device, print _msg, global _state, writer, num _examples =2):
WSL: Ubuntu > TIMELINE model. eval ()
Ln46. Col33 Spaces:4 UTF-8 LF Python 3. 10. 6(trans
ENG
5/23/2023
7:30 PM
所以如果有什么你不明白的或者你想让我解释得更清楚的, 请
> OUT UI NE
> WSL : Ubuntu > TIME LNE So let me know if'there is something that you don't understand or you want me. to ENG
/23/2023
7:30 PM
48
49
50
return decoder _input. squeeze (e)
51
> OUTUINE
52
53
def run_validation (model, validation _ds, tokenizer _src, tokenizer _tgt, max_len, device, print _msg, global _state, writer, num _examples =2):
> WSL: Ubuntu
> TIMEUNE
model. eval()
Ln 46, Col33 Spaces:4 UTF-8 LF Python 3. 10. 6(trar
7:30 PM
5/23/2023
49
50
return decoder _input. squeeze (e)
51 我也一定会关注评论区的
> OUTUINE
52
53
def run_validation (
bal _state, writer, num _examples =2):
WSL: Ubuntu
> TIMEUNE
will also for sure follow the comment section.
Spaces 4 UTF-8 LFPython 3. 10. 6(trans
7:30 PM
ENG
5/23/2023
48
49
50
return decoder _input. squeeze (e)
51
> OUTUINE
52
53
def run_validation (model, validation _ds, tokenizer _src, tokenizer _tgt, max_len, device, print _msg, global _state, writer, num _examples =2):
> WSL: Ubuntu > TIMELINE model. eval ()
Ln 46. Col33 Spaces4 UTF-8 LFPython 3. 10. 6 (trans
ENG
5/23/2023
7:30 PM
49
50
return decoder _input. squeeze (e)
51
52 谢谢, 祝你有个美好的一天
> OUTUINE
53
def run_validation (
WSL : Ubuntu > TIMELINE model. eval ()
Thank you and have a nice day.
Ln46. Col33 Spaces4 UTF-8 LFPython ENG
5/23/2023
7:30 PM
