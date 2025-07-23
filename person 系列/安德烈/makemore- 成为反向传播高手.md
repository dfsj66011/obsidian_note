
在前面的代码中，关于反向传播，我们使用的是 `loss.backward()`，即使用 PyTorch 的自动微分功能来计算整个过程中的所有梯度。而我希望去掉 `loss.backward()` 的使用，希望我们能在 Tensor 层面上手动编写反向传播过程。

我认为这是一个非常有用的练习，原因如下。实际上，我有一篇关于这个主题的完整[博文](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)，但我喜欢将反向传播称为一个“有漏洞的抽象”。我的意思是，反向传播并不能让你的神经网络神奇地工作。并不是说你随便堆砌一些可微分函数的乐高积木，然后双手合十祈祷反向传播就能万事大吉。事情不会自动变好。

这是一种"漏洞百出的抽象"——如果你不了解其内部机制，很容易就会搬起石头砸自己的脚。它会莫名其妙地失效或无法达到最佳效果，而如果你希望调试它或在神经网络中解决这个问题，你就需要了解它的底层工作原理，不能因为 PyTorch 或其他框架提供了自动求导功能，我们就可以忽视它的工作原理。

实际上，我们已经介绍过自动求导，还编写了 micrograd。但 micrograd 只是一个针对单个标量的自动求导引擎。但我认为这还不够。我希望我们也能在张量层面上思考反向传播。总而言之，我认为这是一个很好的练习。我认为它非常非常有价值。你会变得更擅长调试神经网络，并确保你理解自己在做什么。这将使一切变得完全透明。这样你就不会因为有什么被隐瞒而感到紧张。

这里有一个有趣的历史小插曲：如今，手动编写反向传播代码并不被推荐，除了练习目的之外，没有人会这么做。但大约 10 年前，在深度学习领域，这种做法相当普遍，甚至无处不在。那时候，包括我在内，每个人都习惯于手动编写反向传播代码。这就是当时的常规做法。所以我们过去都是手动编写反向传播。现在大家只是简单地调用 `loss.backward()`。我们失去了一些东西。



我想举几个例子来说明这一点。这是2006年杰弗里·辛顿和鲁斯兰·斯拉夫迪诺夫在《科学》杂志上发表的一篇颇具影响力的论文。当时他们训练了一些被称为受限玻尔兹曼机的架构。简单来说，这里训练的是一个自动编码器。这大概是在2010年左右。我当时有一个用于训练受限玻尔兹曼机的库。

那时候都是用MATLAB写的。所以Python在深度学习领域还没有普及开来。当时全都是用MATLAB。MATLAB是一个人人都在用的科学计算软件包。我们会用MATLAB编程，它也是一种编程语言。但它有一个非常方便的tensor类。

这就是这个计算环境。你会在这里运行。当然，这一切都在CPU上运行。但你会有非常漂亮的图表和一个内置的调试器。而且它相当不错。现在，我在2010年为这个包编写的用于拟合受限玻尔兹曼机的代码在很大程度上是可识别的。

但我想向你展示你会如何，嗯，我正在创建数据和xy批次。我正在初始化神经网络。所以它和我们习惯的一样，有权重和偏置。然后这是训练循环，我们在这里实际进行前向传递。而在当时，甚至不一定使用反向传播来训练神经网络。所以这里特别实现了对比散度，用于估计梯度。

然后在这里，我们获取该梯度，并按照我们熟悉的方式用它进行参数更新。对，就是这里。但你可以看到，基本上人们是在直接、即时地自行调整这些梯度。那时候使用自动求导引擎并不常见。再举一个例子，这是我2014年一篇名为《片段嵌入》的论文。在这里，我所做的工作是对齐图像和文本。

所以这有点像是一个剪辑工具，如果你熟悉的话。但它不是在整张图片和完整句子的层面上工作，而是在单个物体和句子的小片段层面上工作。我将它们嵌入，然后计算一个非常类似于剪辑工具的损失函数。

我翻出了2014年实现这个功能的代码。它已经是用NumPy和Python写的了。这里，我正在实现成本函数。而标准做法不仅是手动实现成本，还要手动实现反向传播。在这里，我计算图像嵌入、句子嵌入和损失函数，并计算得分。

这是损失函数。一旦我有了损失函数，我就在这里进行反向传播。因此，我通过损失函数和神经网络进行反向传播。这是损失函数。一旦我有了损失函数，我就会在这里进行反向传播。因此，我通过损失函数和神经网络进行反向传播。

而我加入了正则化。所以一切都是手工完成的。你得自己写出反向传播的过程。然后，你会使用梯度检查器来确保你对梯度的数值估计与反向传播过程中计算的结果一致。这在很长一段时间里都是非常标准的做法。但如今，当然，使用自动求导引擎已成为标准。

但这绝对是有用的。我认为人们从非常直观的层面理解了这些神经网络的工作原理。所以我认为这又是一次很好的练习。这就是我们想要达到的目标。提醒一下，这是我们之前讲座中实现的Jupyter Notebook。我们将保持一切不变。

因此，我们仍将使用一个带有批量归一化层的双层多层感知机。前向传播过程基本上与本讲座中的内容相同。但在这里，我们将去掉loss.backward，而是手动编写反向传播过程。以下是本讲座的起始代码。在这个笔记本中，我们将成为反向传播的高手。前几个单元格的内容与我们之前所熟悉的完全一致。

所以我们正在进行一些导入操作，加载数据集并处理数据集。这些都没有变化。现在在这里，我将介绍一个实用函数，稍后我们将用它来比较梯度。因此，我们将特别关注自己手动估算的梯度，以及PyTorch计算出的梯度。我们将进行正确性验证，当然前提是假设PyTorch的计算是正确的。

那么在这里，我们有一个非常熟悉的初始化过程。我们有字符的嵌入表，第一层，第二层，以及中间的批量归一化层。这里就是我们创建所有参数的地方。现在，你会注意到我把初始化参数稍微改小了一些。通常来说，你会把偏置项全部设为零。而在这里，我把它们设为了小的随机数。我这样做是因为，如果你的变量初始化为零，有时可能会掩盖梯度实现的错误。

因为当所有值都为零时，梯度表达式会大大简化，比原本的形式要简洁得多。通过使用较小的数值，我试图揭示这些计算中可能存在的错误。你还会注意到，我在第一层中使用了b1。我在这里使用了偏置项，尽管紧接着会进行批量归一化。通常你不会这么做，因为我们讨论过你其实不需要偏置项。但我这么做只是为了好玩，因为我们将要计算关于它的梯度。

我们可以验证即使存在这种虚假偏差，我们仍然在正确地进行计算。在这里，我计算了一个批次的数据。然后在这里，我进行了一次前向传播。现在，你会发现前向传播比我们习惯的要复杂得多。以前，前向传播就在这里。现在，前向传播变长的原因有两个。

首先，我们刚刚使用了f.cross_entropy函数。但在这里，我将重新实现一个显式的损失函数。其次，我已经将这个实现分解成了易于管理的部分。因此在前向传播过程中，我们会得到很多中间张量。这是因为我们即将进行反向传播，从底部到顶部计算梯度。所以我们将向上回溯。

就像我们在前向传播中有 logprops 张量一样，在反向传播中，我们会有一个 dlogprops，它将存储损失相对于 logprops 张量的导数。因此，我们将在这些张量的名称前加上 d，并在反向传播的过程中计算它们。举个例子，我们这里有一个原始的 b。

我们将以原始方式计算一个db。在这里，我告诉PyTorch我们希望保留所有这些中间值的梯度，因为在练习一中，我们要计算反向传播。因此，我们将计算所有这些d变量，并使用我之前介绍的CMP函数来检查我们相对于PyTorch给出的结果的正确性。

这是练习一，我们将通过整个计算图进行反向传播。现在，快速预览一下练习二及之后的内容：在这里，我们已经完全分解了损失函数，并通过其所有微小组成部分手动进行了反向传播。但在这里，我们将把损失函数简化为一个单一的交叉熵计算。

相反，我们将通过数学推导，用纸和笔来解析损失函数相对于逻辑值的梯度。我们不会逐个反向传播所有小块，而是直接解析出这个梯度，并实现它，这样效率会高得多，稍后我们会看到这一点。接下来，我们将对批量归一化做完全相同的事情。

因此，与其将批量归一化分解成众多小部件，我们将使用纸笔、数学和微积分来推导通过批量归一化层的梯度。也就是说，我们将以一种更高效的表达式来计算通过批量归一化层的反向传播过程，而不是独立地反向传播所有小部件。这就是练习三的内容。

然后在练习四中，我们将把所有内容整合起来。这是训练这个两层MLP的完整代码。我们将手动插入反向传播，并移除loss.backward。你会看到，完全使用自己的代码也能得到完全相同的结果。我们唯一从PyTorch中使用的就是torch.tensor，以提高计算效率。除此之外，你将完全理解神经网络的前向传播、反向传播以及训练过程的含义。我认为这将非常棒。

那么，我们开始吧。好的，我已经运行了这个笔记本的所有单元格，一直到这一步。现在我要把这些内容删掉。我将从DLogProps开始实现反向传播。我们需要理解这里应该放什么，以计算损失相对于LogProps张量所有元素的梯度。现在，我将在这里给出答案。

但我想在这里快速说明一下，我认为对你最有教学意义的是，实际上你可以进入这个视频的描述部分，找到这个Jupyter笔记本的链接。你既可以在GitHub上找到它，也可以在Google Colab上找到。所以你不必安装任何东西，只需访问Google Colab上的一个网站即可。

你可以尝试自己实现这些导数或梯度。如果你实在想不出来，再来看我的视频，看我如何解答。这样我们可以同步进行，先自己尝试，然后再看我揭晓答案。我认为这对您来说最有价值。这也是我建议您学习本讲座的方式。那么，我们就从DLogProps开始讲起。

现在，DLogProps将保存损失相对于LogProps所有元素的导数。LogProps里面有什么？它的形状是32乘27。因此，DLogProps也应该是一个大小为32乘27的数组，这一点不会让你感到意外，因为我们想要损失相对于其所有元素的导数。

所以这些的大小总是相等的。那么，LogProps 如何影响损失呢？损失是负的 LogProps，索引范围是 n 和 yb，然后取其平均值。提醒一下，yb 基本上就是一个包含所有正确索引的数组。所以我们在这里所做的是，我们正在处理一个大小为32乘27的LogProps数组。没错。然后我们要逐行处理每一行。

在每一行中，我们依次提取索引8、14、15等数据。这样我们逐行向下遍历，这就是n的迭代范围。然后我们总是提取由张量yb指定的列索引。所以在第零行，我们取第八列。在第一行，我们取第十四列，以此类推。这样LogProps就能提取出序列中下一个正确字符的所有LogProps概率。

原来这就是它的作用。而这个形状或者说大小当然是32，因为我们的批次大小是32。所以这些元素被挑选出来，然后它们的平均值取负值就变成了损失。所以我总是喜欢用更简单的例子来理解导数的数值形式。这里的情况是，一旦我们挑选出这些例子，我们就会取平均值然后取负数。所以损失函数基本上可以这样写，就是a加b加c的负数。这三个数的平均值就是负的，除以三。

这就是我们计算三个数a、b、c平均值的方式，尽管实际上这里有32个数。那么，比如dA造成的损失基本上是什么呢？如果我们从数学上简化这个表达式，就是负的a的三分之一加上负的b的三分之一加上负的c的三分之一。那么d loss by dA是什么呢？就是负的三分之一。所以你可以看到，如果我们不仅仅有a、b和c，而是有32个数，那么d loss by d，每一个数一般来说都会是1/n，因为n是批次的大小，在这里是32。因此，d loss by d logprobs在所有位置上都是负的1/n。

那么，logprobs中的其他元素呢？因为logprobs是一个很大的数组。你可以看到logprobs.shank的维度是32×27。但其中只有32个参与了损失计算。那么，那些没有被提取出来的大部分元素的导数是什么呢？直观地说，它们的损失是0。抱歉，更准确地说，它们的梯度直观上是0。这是因为它们并没有参与损失的计算。这个张量中的大部分数字实际上并没有影响到损失值。因此，如果我们改变这些数字，损失值并不会发生变化，这就相当于说损失相对于它们的导数为0。它们对损失没有影响。

那么，这里有一种实现这个导数的方法。我们首先创建一个形状为32×27的全零张量。或者这样说吧，为了避免硬编码数字，我们不如像logprobs那样使用tors.zeros。简单来说，这将创建一个与 logprobs 形状完全相同的零数组。然后我们需要在这些特定位置设置负1除以n的导数。具体可以这样做。

以相同方式索引的dlogprobs将被直接设为负1除以0除以n，对吧？就像我们在这里推导的那样。现在让我擦掉所有这些推理过程。那么这就是dlogprobs的候选导数。让我们取消第一行的注释，检查一下是否正确。好的，CMP运行了。

让我们回到CMP。你可以看到它正在计算我们计算的值（即dt）是否完全等于PyTorch计算的t.grad。然后它会确保所有元素都完全相等，并将结果转换为一个布尔值。因为我们不需要布尔张量，只需要一个布尔值。然后在这里，我们要确保，如果它们不完全相等，可能由于一些浮点数问题，它们近似相等，但非常非常接近。所以我们在这里使用torch.allclose，它允许有一点点的浮动空间，因为有时候结果会非常非常接近。

但如果使用稍微不同的计算方法，由于浮点运算的特性，可能会得到略微不同的结果。因此，这里检查的是你是否得到了一个近似接近的值。结果接近。然后我们在这里检查最大值，基本上是差异最大的那个值，以及两者之间的绝对差值是多少。因此，我们打印出是否完全相等、近似相等，以及最大的差异是多少。

因此，在这里，我们实际上看到了完全相等的情况。所以，当然，我们也有近似相等的情况。而最大差异正好为零。基本上，rdlogprops 完全等同于 PyTorch 在反向传播中计算出的 logprops.grad。到目前为止，我们进展得很顺利。好的，现在让我们继续反向传播过程。

我们有logprops通过log依赖于props。因此，props的所有元素都逐个应用了log2。现在，如果我们想要深层次的props，那么请记住你的micrograd训练，我们有一个类似log节点的东西。它接收props并生成logprops。深层props将是该独立操作（即对数运算）的局部导数，乘以损失相对于其输出（在此情况下为dlogprops）的导数。那么这个操作的局部导数是什么呢？实际上，我们正在逐元素进行对数运算。

我们可以来到这里，看到从alpha是你的朋友，log(x)对x的导数d/dx就是简单的1/x。因此，在这种情况下，x就是props。所以我们有d/dx等于1/x，也就是1/props。然后这就是局部导数。有时我们想把它串联起来。这就是链式法则，乘以dlogprops。然后让我取消注释并就地运行单元格。

而我们看到，正如我们在这里计算的那样，props 的导数完全正确。因此请注意这里的工作原理。Props 将被反转，然后在这里进行逐元素相乘。因此，如果你的概率值非常接近1，这意味着你的网络当前正确预测了字符，那么这将变为1除以1。而dlogprops则直接传递。但如果你的概率分配不正确，也就是说如果正确字符在这里得到的概率非常低，那么1.0除以它将会放大这个值，然后乘以dlogprops。所以直观上，这一行代码的作用是：它选取那些当前分配概率非常低的样本，并放大它们的梯度。

你可以这么看。接下来是countSumInv。所以我们需要求它的导数。现在让我在这里稍作停顿，简单介绍一下这里发生的情况，因为我知道这有点让人困惑。我们从神经网络中得到的是logits。这里我所做的是在每一行中找到最大值，并减去它，目的是为了数值稳定性。

我们讨论过，如果不这样做，当某些逻辑值过大时，就会出现数值问题，因为最终我们要对它们进行指数运算。所以这样做只是为了数值上的安全。然后这里对所有逻辑值进行指数运算以生成我们的计数。然后我们要对这些计数求和并进行归一化，使所有概率之和为1。这里我没有使用1除以countSum，而是使用了负1次方。在数学上它们是等价的。我只是发现PyTorch在实现反向被动除法时存在问题，会给出奇怪的结果，但对于星号星号负1这种情况不会发生，所以我改用这个公式。但基本上这里所做的就是：我们得到了logits，要对它们全部取指数，然后对计数进行归一化以创建我们的概率。

问题在于这涉及多条计算路径。所以现在我们需要先求导，然后反向传播到 countSumIf 函数，接着再传递到 counts 变量。那么 countSumIf 应该是什么呢？这里我们确实需要格外小心，因为必须仔细检查并确保张量形状匹配。

所以counts.shape和countSumInf.shape是不同的。具体来说，counts是32×27的矩阵，而这个countSumIf是32×1的列向量。因此，在这个乘法运算中，PyTorch会进行隐式广播操作——它需要将这个包含32个数字的列张量水平复制27次，使两个张量对齐才能进行逐元素相乘。用一个小例子来说明，实际运算过程是这样的：

我们这里实际上只是将属性 counts 乘以 countsSumIf，所以就是 c 等于 a 乘以 b，但 a 是 3×3 矩阵，而 b 只是 3×1 的列张量。PyTorch 在内部复制了 b 的这些元素，并在所有列上进行了这种复制。例如，b1 作为 b 的第一个元素，在这个乘法中会被复制到这里的所有列上。

现在我们正试图通过反向传播来计算这个操作对countsSumInf的影响。在计算这个导数时，重要的是要意识到这看似是一个单一操作，但实际上是由两个操作顺序执行的。PyTorch执行的第一个操作是获取这个列张量，并基本上将其复制到所有列上，共复制了27次。

这就是第一个操作，即复制。然后第二个操作是乘法。所以，我们先反向传播通过乘法运算。如果这两个数组大小相同，我们只有a和b，都是3×3的矩阵，那么如何通过乘法进行反向传播？如果我们只有标量而不是张量，那么如果有c等于a乘以b，那么c对b的导数是什么？其实就是a。这就是局部导数。

因此，在我们的例子中，撤销乘法并通过乘法本身进行反向传播（即逐元素操作），将得到局部导数。在本例中，局部导数就是计数（counts），因为计数就是a。所以这就是局部导数，然后根据链式法则乘以d props。这里的这个就是导数，或者说梯度，但它是相对于复制后的b的。但我们并没有一个复制后的b，我们只有一个单独的b列。

那么，我们现在如何通过复制进行反向传播呢？直观地说，这个b1是同一个变量，只是被多次重复使用。因此，你可以将其视为我们在micrograd中遇到的一个案例。这里，我只是随便拿出一个我们在micrograd中使用过的图。

我们有一个例子，其中一个节点的输出被输入到图中的两个分支，直到损失函数。我们讨论的是在反向传播过程中正确的做法是，我们需要将所有到达任一节点的梯度相加。因此，在这些不同的分支中，梯度会被求和。

因此，如果一个节点被多次使用，在反向传播过程中，其所有用途的梯度会被求和。在这里，b1 在所有这些列中被多次使用，因此正确的做法是水平地对所有行进行求和。所以我们希望在维度1上进行求和，但同时希望保留这个维度，以便 countSumInv 和它的梯度保持完全相同的形状。

因此，我们要确保它们保持真实，这样我们就不会失去这个维度。这将使countSumInv的形状精确为32x1。同样，揭示这个比较并运行它，我们可以看到得到了完全匹配的结果。因此，这个导数是完全正确的。让我把这个擦掉。现在我们也反向传播到countS，这是这里创建props的另一个变量。所以从props到countSumInv，我们刚刚已经完成了。

让我们也来看看countS。所以dcounts将是，dcounts是我们的a，因此dc/da就是b，所以它就是countSumInv，然后乘以链式法则的dprops。现在countSumInv是32×1，dprops是32×27。所以这些可以正常广播，并为我们提供dcounts。这里不需要额外的求和操作。在本次乘法运算中会进行一次广播，因为countSumInv需要再次复制才能正确乘以dprops。

但这将给出正确的结果。就这个单一操作而言，我们已经从props反向传播到了countS，但实际上我们无法检查countS的导数。我稍后会详细讲解这一点。原因在于countSumInv依赖于countS。因此，这里还存在第二个分支需要完成，因为countSumInv会反向传播到countSum，而countSum又会反向传播到countS。

因此，countS 是一个被使用了两次的节点。它在这里被用于 props，同时还通过 countSumInv 分支传递。所以尽管我们已经计算了它的第一个贡献，但稍后仍需计算它的第二个贡献。好的，我们继续这个分支。我们已经得到了countSumInv的导数。现在我们需要countSum的导数。

所以dcountSum等于，这个操作的局部导数是什么？这基本上就是countSum的逐元素倒数。countSum的负一次方就等于1除以countSum。如果我们去Wolfram Alpha查一下，会发现x的负一次方对x求导，结果基本上是负x的负二次方，对吧？负1除以x平方就等于负x的负二次方。因此这里的dcountSum，其局部导数就是负countSum的负二次方，这就是局部导数，再乘以链式法则中的dcountSumInv。

这就是dcountSum。让我们取消注释并验证我的判断是否正确。好的，现在我们得到了完全相等的结果。这里没有任何形状上的模糊之处，因为它们都是相同的形状。好的，接下来，我们要沿着这条线进行反向传播。countSum 是 count 沿行的求和。

所以我在这里写了一些帮助内容。我们必须记住，counts 当然是 32 行 27 列的矩阵，而 countSum 是 32 行 1 列的矩阵。因此在这个反向传播过程中，我们需要将这列导数转换成一个二维数组的导数。那么这个操作在做什么呢？我们输入某种矩阵，比如一个 3 行 3 列的矩阵 A，然后将各行求和得到一个列张量 B，即 B1、B2、B3，基本上就是这样。

所以现在我们得到了损失函数关于B的导数，也就是B的所有元素的导数。现在我们要推导损失函数关于所有这些小A的导数。那么B是如何依赖于A的，基本上就是我们想要知道的。这个操作的局部导数是什么？我们可以看到，B1只依赖于这里的这些元素。

B1对下面所有这些元素的导数都是0。但对于这里的这些元素，比如A11、A12等，局部导数是1，对吧？例如，DB1/DA11就是1。所以是1、1和1。因此，当我们计算损失函数对B1的导数时，B1对这些输入的局部导数在这里是0，但对这些元素是1。所以在链式法则中，我们有局部导数乘以B1的导数。由于这三个元素的局部导数是1，局部导数乘以B1的导数就只是B1的导数本身。

所以你可以把它看作一个路由器。基本上，加法就是梯度的路由器。无论来自上层的梯度是什么，它都会被均等地路由到参与该加法的所有元素中。因此，在这种情况下，B1的导数将均匀流向A11、A12和A13的导数。所以，如果我们有这个列张量中B所有元素的导数，也就是我们刚才计算的D计数总和，我们基本上可以看到，所有这些现在都流向A的所有这些元素，并且是水平流动的。因此，基本上，我们需要的是一个大小为32乘1的D计数总和，我们只需要水平复制27次，以创建一个32乘27的数组。

因此，实现这一操作有多种方法。当然，你可以直接复制张量，但我认为一个简洁的方法是让 D_counts 简单地等于 torch.ones，也就是一个形状与 counts 相同的二维全1数组（比如32x27），再乘以 D_counts 的总和。这样一来，我们实际上是通过广播机制来实现复制操作。

你可以这么看。但我们也必须小心，因为D计数已经全部计算过了。我们之前在这里计算过，那只是第一个分支，现在我们正在完成第二个分支。所以我们需要确保这些梯度相加，也就是使用 +=。然后在这里，我们把比较的部分注释掉，希望一切顺利，确保我们得到正确的结果。看来 PyTorch 在这个梯度上也和我们达成了一致。

好的，希望我们现在已经掌握了要点。Counts是normlogits的逐元素指数运算结果。所以现在我们需要denormlogits，由于这是逐元素操作，一切都非常简单。e的x次方的局部导数是什么？众所周知，它就是e的x次方，所以这就是局部导数。这就是局部导数。我们已经计算过了，它在counts里面，所以我们不妨直接复用counts。

这就是局部导数乘以D计数。虽然看起来有点滑稽。计数乘以D计数就是normlogits上的导数。现在让我们擦掉这个，验证一下，看起来不错。这就是normlogits。好的，现在我们来到这一行，denormlogits。

我们拥有这些数据，并且正在尝试计算D logits和D logit最大值，因此需要通过这一行进行反向传播。现在，我们必须小心，因为形状再次不一致，所以这里发生了隐式广播。因此，normlogits的形状是32乘以27。

Logits 也是如此，但 logit 的最大值仅为 32 乘 1，因此这里在减法运算中存在广播机制。现在，我尝试再次写出一个简单的示例。我们基本上有 C 等于 A 减去 B，由于形状的原因，我们可以看到这些是 3 乘 3 的矩阵，而这个只是一个列向量。

例如，对于C中的每一个元素，我们需要考察其生成过程。实际上，C中的每个元素都等于对应的A元素减去关联的B元素。现在可以非常清晰地看出：每个C元素对其输入的导数，在对应A元素处为1，在对应B元素处为-1。因此，C上的导数将等量流向对应的A元素和B元素。但除此之外，由于B元素存在广播机制，我们还需要像之前那样进行额外的求和操作。当然，B元素的导数会带有负号，因为此处的局部导数为-1（比如dC₃₂/dB₃=-1）。现在让我们具体实现这个逻辑。

基本上，dlogits 将完全复制 normlogits 的导数。因此 dlogits 等于 dnormlogits，为了安全起见我会做一个点克隆（dot clone），这样我们只是做一个副本。然后由于负号的存在，dlogit maxis 将会是 dnormlogits 的负数。

然后我们必须小心，因为logit_maxis是一个列向量，就像我们之前看到的那样，由于我们在所有列上重复相同的元素，因此在反向传播过程中，由于我们不断复用这个变量，这些就像是该变量的不同使用分支。因此，我们需要沿某一维度求和，同时保持维度不变（keepdims=True），这样才不会破坏这个维度。最终dlogit_maxis的形状将保持不变。

现在我们必须小心，因为这里的 dlogits 并不是最终的 dlogits。这是因为梯度信号不仅通过这里传递到 logits，而且 logit_maxis 也是 logits 的函数，这是进入 logits 的第二个分支。所以这还不是我们对 logits 的最终导数，稍后我们会回来处理第二个分支。目前，dlogit_maxis 是最终的导数。

那么让我在这里取消注释这个CMP，然后直接运行它。如果PyTorch和我们一致的话，logit最大值就会显示出来。这就是通过这一行的导数。在我们继续之前，我想在这里稍作停顿，仔细看看这些logit最大值，尤其是它们的梯度。我们在之前的讲座中提到过，我们这样做的唯一原因是为了确保这里实现的softmax函数的数值稳定性。我们还讨论过，如果你对这些例子中的任何一个logits（即logits张量的某一行）的所有元素同时加减相同的值，概率值（probs）将保持不变。

你并没有改变softmax。这样做只是为了确保exp不会溢出。我们使用最大值的原因是因为这样可以保证每一行的logits中，最大的数字都是零。因此，这将是安全的。所以基本上这会产生影响。如果改变 logit maxis 不会改变概率，因此也不会改变损失，那么 logit maxis 上的梯度应该为零。

对吧？因为说这两件事是一样的。所以我们确实希望这个数字非常非常小。事实上，我们希望这个数字是零。现在，由于浮点数的某种不稳定性，结果并不完全为零。只有部分行会显示为零。但我们得到了极小的值，比如1e-9或10。

因此，这表明 logit maxis 的值不会影响损失函数，正如它们本不应该影响一样。说实话，通过这个分支进行反向传播感觉有点奇怪，因为如果你在 PyTorch 中实现了类似 f.crossentropy 的函数，并且你将所有这些元素组合在一起，而不是逐块进行反向传播，那么你可能会认为这里的导数正好为零。所以你会跳过这个分支，因为它只是为了数值稳定性而做的。

但有趣的是，即使你将所有内容分解为完整的原子，并且仍然按照你希望的方式在数值稳定性方面进行计算，正确的事情仍然会发生。而且你在这里得到的梯度非常非常小，基本上反映了这些值相对于最终损失并不重要的事实。好的，那么现在让我们继续沿着这条线进行反向传播。

我们刚刚计算了logit最大值，现在要通过第二个分支反向传播回logits。这里我们当然取了logits，并沿着所有行取了最大值，然后在这里查看了它的值。PyTorch中的工作机制是这样的：这里的max操作会同时返回最大值和对应的索引位置，通过这些索引来统计最大值。

现在，在前向传播中，我们只使用了数值，因为这就是我们所需要的。但在反向传播中，了解这些最大值出现的位置非常有用。而我们恰好掌握了它们出现的索引位置。当然，这将帮助我们进行反向传播。因为在这种情况下，反向传播应该是什么样子的呢？我们有一个32乘27的logit张量。在每一行中，我们找到最大值。

然后这个值被提取到logit最大值中。直观地说，基本上，流经此处的导数应该是1乘以局部导数，对于被提取出的适当条目来说局部导数为1，然后再乘以logit最大值的全局导数。所以实际上，我们在这里所做的，如果你仔细想想，就是我们需要获取delogit最大值，并将其分散到这些logits中最大值来源的正确位置。

于是我想出了一行代码来实现这个功能。让我先擦掉这里的一些内容。这行代码，你可以采用和我们之前类似的做法，先创建一个全零数组，然后填充正确的元素。所以我们在这里使用索引，并将它们设为1。但你也可以使用one-hot编码。即f.one-hot，然后我在第一个维度上取logit最大值，点索引，并告诉PyTorch每个张量的维度应该是27。这样做的效果是——好吧，抱歉，这有点疯狂。

PLT。我对此很确定。它实际上只是一个数组，表示每行中最大值的位置，该元素为1，其他所有元素为0。所以每行都是一个独热向量，这些索引现在在正确的位置填充了一个1。然后我在这里所做的就是将它与最大值的对数相乘。

请记住，这是一个32行1列的矩阵，因此当我将其与logit最大值相乘时，logit最大值会进行广播操作，该列会被复制，然后逐元素相乘将确保每个值都被路由到对应的激活位上。这是实现此类操作的另一种方法，这两种方法都可以使用。我只是想展示一种等效的实现方式。

我使用加等运算符是因为我们已经在这里计算了逻辑值，现在是第二个分支。让我们看看逻辑值以确保这是正确的，可以看到我们得到了完全正确的答案。接下来，我们要继续处理这里的逻辑值。这是该线性层中矩阵乘法和偏置偏移的结果。我已打印出所有这些中间张量的形状。正如我们刚才所见，logits的形状自然是32×27。

这里的h是32乘以64的矩阵，所以这些是64维的隐藏状态。然后这个w矩阵将这些64维向量投影到27维，还有一个27维的偏移量，这是一个一维向量。现在我们应该注意到这里的加法实际上是广播操作，因为h乘以w2会得到一个32乘以27的矩阵。

因此，这里的b2加上这个就变成了一个27维的向量。根据广播规则，这个偏置向量会发生什么呢？这个一维的27维向量会在左边对齐一个填充的维度1，基本上就变成了一个行向量，然后它会垂直复制32次，变成32×27的矩阵，然后进行逐元素相乘。

现在的问题是，我们如何从逻辑回归值反向传播到隐藏状态、权重矩阵w2和偏置项b2？你可能会认为我们需要用到一些矩阵微积分，然后查阅矩阵乘法的导数，但实际上你不需要做这些，你可以回到基本原理，自己在一张纸上推导出来。

具体来说，我喜欢且对我很有效的方法是：先找一个具体的小例子完整写出来，在分析这个小例子的运作过程中，你就能理解更广泛的模式，从而能够归纳并写出完整的通用公式，解释这些导数在类似表达式中的流动规律。现在让我们试试这个方法。请原谅这里的低成本制作，我是在一张纸上写出来的。

实际上我们感兴趣的是，我们有一个a乘以b再加上c，这会产生一个d，而我们有了损失函数对d的导数，我们想知道损失函数对a、b和c的导数是什么。现在这里是一些关于矩阵乘法的小型二维示例。一个2×2的矩阵乘以另一个2×2的矩阵，再加上一个只有两个元素c1和c2的向量，会得到一个2×2的矩阵。注意这里我有一个叫做c的偏置向量，这个偏置向量是c1和c2，但正如我之前提到的，在广播过程中这个偏置向量会变成一个行向量，并且会垂直复制。所以这里发生的情况也是一样的。

c1 c2 被垂直复制，结果我们看到两行 c1 c2。所以当我说写出来时，我的意思就是这样。基本上就是将这个矩阵乘法分解成底层实际发生的操作。因此，矩阵乘法的运算机制决定了d11是矩阵a第一行与矩阵b第一列点积的结果。具体表现为a11乘以b11加上a12乘以b21再加上c1，以此类推计算d矩阵的所有其他元素。当你实际展开运算时，会发现这不过是一系列乘法与加法的组合——而我们在micrograd中早已掌握了如何对乘法和加法进行微分。所以矩阵乘法不再令人畏惧，它本质上只是繁琐的重复运算罢了。虽然过程冗长，但完全可解。我们已掌握dl对d矩阵各元素的偏导，现在需要求解的是dl对所有其他微小变量的偏导。

那么我们该如何实现这一点，又该如何实际获取梯度呢？好吧，低成本制作继续在这里进行。举个例子，让我们推导一下损失相对于a11的导数。可以看到，在这个简单的表达式中，a11出现了两次，就在这里和这里，并且影响了d11和d12。

那么，dl对a11的导数是什么呢？其实就是dl对d11的导数乘以d11对a11的局部导数，在这里就是b11，因为b11是与a11相乘的系数。同理，d12对a11的局部导数就是b12，因此在链式法则中，b12会乘以dl对d12的导数。由于a11同时参与了d11和d12的计算，我们需要将这两条并行链路的贡献相加，这就是为什么我们用加号把这两部分求和，最终得到dl对a11的导数。

我们可以对另一个元素进行完全相同的分析，对于a的所有其他元素也是如此。当你简单地写出来时，对这样的表达式求梯度就变得非常简单。你会发现我们正在寻找的矩阵dl/da，如果我们只是将它们排列成与a相同的形状，那么a只是一个矩阵，所以这里的dl/da也将是一个具有相同形状的张量，但现在包含导数，比如dl/da11等等。我们实际上可以看到，我们可以将这里写出的内容表示为矩阵乘法，因此恰好dl/...，我们通过求梯度得出的所有这些公式实际上都可以表示为矩阵乘法。

特别是，我们可以看到这是这两个矩阵的矩阵乘法，即dl乘以d，然后矩阵乘以b，但实际上b是转置的。因此，你会发现b21和b12的位置互换了，而之前我们当然有b11、b12、b21、b22。所以你可以看到另一个矩阵b被转置了。

简单来说，通过这个非常简单的例子进行分解推理，我们目前得出的结论是：dl/da（也就是这个表达式）实际上就等于dl/dd矩阵乘以b的转置。这就是我们目前所得到的结论。现在，我们还想要关于b和c的导数。对于b，我就不详细推导了，因为老实说这并不复杂，只是很繁琐，让人精疲力尽。

其实你可以自己进行这个分析。你还会发现，如果你这样做，你会发现dL/dB也是一个矩阵乘法。在这种情况下，你需要对矩阵A进行转置，然后与dL/dD进行矩阵乘法，这样就能得到dL/dB。然后对于偏移量C1和C2，如果你再次对C1进行微分，你会发现像这样的表达式，对C2也是一样，基本上你会发现dL/dC就是简单的，

因为它们只是抵消了这些表达式，你只需要取D的导数构成的dL乘以dD矩阵，然后对列进行求和，就能得到C的导数。简而言之，矩阵乘法的反向传播也是一个矩阵乘法，就像我们之前有D等于A乘以B加上C一样，在标量情况下，我们得到的结果非常非常相似，但现在是用矩阵乘法代替了标量乘法。

因此，D对A的导数等于dL对dD的矩阵乘以B的转置，而这里则是A的转置乘以dL对dD。但在这两种情况下，都是导数与乘法中的另一项进行矩阵相乘。至于C，则是一个求和操作。现在，我要告诉你一个秘密。我从来记不住我们刚刚推导出的矩阵乘法反向传播公式，但我能轻松通过这些表达式进行反向传播。之所以能这样做，是因为维度必须匹配。

那么，我来举个例子。假设我想创建DH，那么DH应该是什么呢？首先，我必须知道DH的形状必须与H的形状相同，而H的形状是32×64。其次，我掌握的另一个信息是，DH必须是dLogits与W2的某种矩阵乘法。

dLogits是32×27的矩阵，W2是64×27的矩阵。在这种情况下，只有一种方式能让维度匹配，而这确实是正确的结果。具体来说，这里的H需要是32×64的矩阵。实现这一目标的唯一方法是取一个dLogits并与...进行矩阵乘法。你看，我必须取W2，但必须转置它才能使维度匹配。所以，W2转置。这是唯一能让这两部分矩阵乘法形状匹配的方法，结果证明这是正确的公式。

所以，如果我们来到这里，我们需要DH，也就是dA，我们可以看到dA是dL乘以dD矩阵乘以B的转置。所以，那就是dLogits乘以，而B是W2。所以，W2的转置，这正是我们这里有的。所以，没必要记住这些公式。同理，现在如果我想求dW2，我知道它一定是dLogits和H的矩阵乘法。可能还需要转置一下……比如，里面有一个转置，我也不知道具体是哪个方向。所以，我得回到W2，看到它的形状是64乘27。

这必须来自这两者的矩阵乘法。因此，要得到一个64乘27的矩阵，我需要取H，需要转置它，然后需要进行矩阵乘法。所以，这将变成64乘以32。然后我需要将32乘以27进行矩阵乘法。这样就会得到64乘以27。

所以，我需要将这个与dLogits的形状进行矩阵相乘，就像这样。这是唯一能让维度匹配的方法。而且只用矩阵乘法。如果我们来到这里，就会发现这正是这里所展示的。所以，A的转置。对我们来说，A就是H。然后与dLogits相乘。

所以，这就是W2。然后dB2只是垂直求和。实际上，同样的道理，只有一种方法能让形状匹配。我不需要记住这是沿着第0轴的垂直求和，因为这是唯一合理的解释。因为B2的形状是27。所以，为了在这里得到dLogits，它的形状是32乘以27。

因此，既然知道这只是某个方向上的dLogits之和，那个方向必须为零，因为我需要消除这个维度。所以，就是这样。这有点像是一种取巧的方法。让我复制、粘贴并删除那个。然后让我跳到这里。希望这就是我们线性层的反向传播。

那么，现在让我们取消这三行的注释。我们要检查一下是否正确地得到了这三个导数。然后运行。我们看到H、W2和B2都是完全正确的。因此，我们已经通过线性层进行了反向传播。接下来，我们已经得到了H的导数。

我们需要通过tanh函数反向传播到HPREACT。因此，我们需要推导出dHPREACT。在这里，我们必须通过tanh进行反向传播。我们已经在micrograd中实现了这一点。还记得tanh的反向传播公式非常简单。但遗憾的是，如果我只是简单地在alpha两端输入d/dx tanh(x)，结果会让我们失望。

它告诉我们这是一个x的双曲正割函数的平方。这其实没什么帮助。但幸运的是，谷歌图片搜索没有让我们失望，它给出了更简单的公式。特别是，如果A等于tanh(z)，那么通过tanh反向传播的dA/dz就是1减去A的平方。需要注意的是，这里的1减去A的平方中的A是tanh的输出，而不是tanh的输入z。因此，这里的dA/dz是用tanh的输出表示的。同样，在谷歌图片搜索中，我们有完整的推导过程。

如果你想真正理解tanh的定义并通过数学推导来计算出1减去tanh(z)的平方。那么，1减去A的平方就是局部导数。在我们的例子中，就是1减去tanh输出的平方，这里也就是h。所以，就是h的平方。这就是局部导数。

然后乘以链式法则，dh。所以，这将是我们的候选实现。所以，如果我们来到这里，然后取消注释这个。让我们期待最好的结果。我们已经有了正确的答案。好的，接下来我们有dhpreact。我们想要反向传播到增益（gain）、bnRaw和bnBias。这里，这些是批归一化（batch norm）的参数——bnGain和bnBias，它们位于批归一化内部，接收严格符合单位高斯分布的bnRaw，然后对其进行缩放和偏移。这些就是批归一化的参数。

现在，这里有一个乘法运算。但值得注意的是，这里的乘法与矩阵乘法截然不同。矩阵乘法涉及的是这些矩阵的行与列之间的点积运算。这是逐元素相乘。因此，情况要简单得多。不过，我们确实需要注意这行代码中的一些广播操作。

所以，你看bnGain和bnBias的维度是1×64。而hpreact和bnRaw的维度是32×64。因此，我们必须小心处理这一点，确保所有维度都能正确匹配，并且广播机制在反向传播中能正确运作。因此，特别是让我们从dbnGain开始。dbnGain应该是，这里再次是逐元素相乘。每当我们有a乘以b等于c时，我们看到这里的局部导数就是，如果这是a，局部导数就是另一个数b。

因此，局部导数就是bnRaw，然后乘以链式法则。所以，dhpreact。这就是候选梯度。现在，我们再次需要小心，因为bnGain的尺寸是1乘64。但这里的尺寸会是32乘64。因此，在这种情况下，正确的做法当然是bnGain，这里是一个由64个数字组成的规则向量，它在这个操作中会垂直复制。

因此，正确的做法是对其进行求和，因为它正在被复制。因此，现在反向传播的每一行中的所有梯度都需要求和到同一个张量dbnGain。所以，我们基本上必须对所有零、所有示例进行求和，也就是这个张量被复制的方向。

现在，我们也要小心，因为 bnGain 的形状是 1 乘 64。所以实际上，我需要保持它们为真。否则，我只会得到 64。现在，我其实不太记得为什么bnGain和bnBias被我设置成1×64的矩阵了。但偏置项b1和b2，我只把它们设成了一维向量。它们不是二维张量。

所以，我不太记得为什么我把增益和偏置保留为二维的了。但只要保持一致并保持不变，其实也没什么关系。因此，在这种情况下，我们需要保持维度以确保张量的形状能正常工作。接下来，我们来看bnRaw。因此，dbnRaw等于bnGain乘以dhPreact。这就是我们的链式法则。

那么，这个的尺寸是怎样的呢？我们必须小心处理，对吧？dhPreact 是 32 × 64，bnGain 是 1 × 64。因此，它会通过复制来完成这个乘法运算，这是正确的做法，因为在正向传播过程中，它也是以同样的方式被复制的。

所以，实际上我们在这里不需要括号。我们已经完成了。而且形状已经是正确的。最后，关于偏置项，情况非常相似。这里的偏置项与我们在线性层中看到的偏置项极其相似。我们可以看到，来自hPreact的梯度会直接流入这些偏置项并累加，因为这些偏置项只是偏移量。

因此，基本上我们希望这是dhPreact，但它需要沿着正确的维度求和。在这种情况下，类似于增益，由于偏置是垂直复制的，我们需要在第零维度（即样本维度）上进行求和。同时，我们希望将keepDim设置为true。

因此，这将基本实现这一功能并对其进行总结，最终给出一个1×64的结果。这就是候选实施方案。它能让所有形状都正常运作。让我在这里提一下。然后让我取消这三行的注释，以检查我们是否得到了所有三个张量的正确结果。确实，我们看到所有这些都正确地进行了反向传播。

现在，我们来到了批归一化层。可以看到，这里的 bnGain 和 bnBias 是参数，因此反向传播到此结束。但 bnRaw 现在是标准化的输出。所以，我现在所做的当然是将批归一化分解成可管理的部分，以便我们可以单独对每一行进行反向传播。但基本上，bnMeanI 就是总和。也就是说，这就是 bnMeanI。

对于变量命名我表示歉意。bnDiff是x减去mu。bnDiff2是x减去mu的平方，这里是方差内部的。bnVar是方差，也就是sigma的平方。这就是bnVar。它基本上是平方和。所以，这是x减去μ的平方，然后求和。现在，你会注意到这里有一个不同之处。在这里，它被归一化为1除以m，即样本的数量。

在这里，我选择将归一化因子设为1/(n-1)而非m。这是有意为之的，稍后当我们讨论到这一行时我会再作解释。这种做法被称为贝塞尔校正（Bessel's correction）。但在当前案例中，这正是我需要的处理方式。

bnVar inv 基本上就变成了 bnVar 加上 epsilon。Epsilon 是我的负5。然后 1 除以平方根就相当于提高到负0.5次方，对吧？因为0.5次方就是平方根。而负号则使其变为1除以平方根。所以，bnVar_inv 就是分母的倒数。然后我们可以看到 bnRaw，也就是这里的 x hat，等于 bnDiff（分子）乘以 bnVar_inv。而创建 hPreact 的这一行代码，是我们已经反向传播过的最后一部分。

所以，我们现在要做的是，我们在这里，我们有bnRaw。然后我们首先要反向传播到bnDiff和bnVar inv。那么，现在我们在这里，我们有dbnRaw。我们需要沿着这条线进行反向传播。这里我已经写出了各部分的形状。实际上，bnVar inv 的形状是1×64。

所以，这里有一个广播机制需要我们小心处理。但这只是一个逐元素的简单乘法运算。到现在为止，我们应该对此相当熟悉了。要得到dbnDiff，我们知道这只是bnVar inv与dbnRaw相乘的结果。反过来，要得到dbnVar inv，我们需要取bnDiff并将其与dbnRaw相乘。因此，这就是候选方案。

但是，当然，我们需要确保广播规则得到遵守。因此，具体来说，bnVar的逆矩阵与dbnRaw相乘是可以的，并且会如我们所预期的那样得到32乘64的结果。但是dbnVar的逆矩阵将会取一个32乘64的矩阵，并将其与另一个32乘64的矩阵相乘。

所以，这是一个32乘64的矩阵。但是，当然，这个bnVar inv只有1乘64。因此，这里的第二行需要对所有样本求和。因为这里存在这个维度，我们需要确保keepDim设置为true。所以，这就是候选方案。让我们把这个擦掉。

让我们在这里转向并实现它。然后让我们注释掉 dbnVar inv 和 dbnDiff。现在，我们实际上会注意到，顺便说一下，dbnDiff 将是错误的。所以，当我运行这个时，bnVar inv是正确的。bnDiff不正确。这实际上是预料之中的，因为我们还没有完成bnDiff。

因此，具体来说，当我们滑动到这里时，可以看到bnRaw是bnDiff的函数。但实际上，bnVar inv是bnVar的函数，而bnVar又是bnDiff2的函数，bnDiff2又是bnDiff的函数。所以，它最终归结到这里。所以，bdnDiff，这些变量名太疯狂了。抱歉。它分成了两个分支，而我们只完成了其中一个分支。

我们必须继续反向传播，最终回到bnDiff。然后我们就可以进行加等于操作，得到实际正确的梯度。目前，验证cmp是否也有效是很好的。它不仅不会欺骗我们，告诉我们一切总是正确的。实际上，它能够检测到梯度不正确的情况。所以，这也是一个值得欣慰的方面。

好的。那么，现在我们有了这里的导数。我们正试图通过这条线进行反向传播。由于我们将其提升到负0.5次方，我提到了幂法则。可以看到，基本上现在bnVar会是...我们把指数降下来。所以，负0.5乘以x，就是这个。

现在将其提升到负0.5次方再减1，即负1.5次方。此时，我们还需要在脑海中应用一个小链式法则，因为我们需要对括号内的这个表达式进一步求bnVar的导数。但由于这是一个逐元素操作且一切相当简单，结果就是1。因此，这里无需额外操作。所以，这就是局部导数。

然后乘以全局导数来创建链式法则。这只是乘以bnVar。所以，这就是我们的候选者。让我把这个调低一点，取消注释检查项。我们看到现在得到了正确的结果。接下来，在反向传播到下一行代码之前，我想简要说明一下这里的注释——当我在这里对平方和进行归一化时，使用的是贝塞尔校正法（Bessel's correction），即除以n-1而非n。

现在，你会发现这与论文有所不同，论文中使用的是1/n，而不是1/(n-1)。这里的m相当于我们的n。因此，计算数组方差实际上有两种方法。一种是偏估计，即1/n；另一种是无偏估计，即1/(n-1)。令人困惑的是，论文中对此描述得并不十分清楚。而且我认为，这个细节实际上相当重要。

他们在训练时使用的是有偏版本。但后来，在讨论推理时，他们提到在进行推理时使用的是无偏估计，基本上是n减1版本，用于推理并校准运行均值和运行方差。因此，他们实际上引入了训练-测试不匹配的问题，即在训练时使用有偏版本，而在测试时使用无偏版本。

我觉得这非常令人困惑。你可以进一步了解贝塞尔校正，以及为什么在总体规模或样本量非常小的情况下，除以n减1能对方差给出更好的估计。这对我们来说确实如此，因为我们处理的是小批量数据。而这些小批量样本是整个训练集这个大群体中的一小部分样本。因此，如果用1/n来计算方差，结果几乎总是会低估真实方差。这是一个有偏估计量，建议使用无偏版本，即除以n-1。你可以参考这篇我喜欢的文章，它详细解释了其中的原理。

我会在视频描述中附上链接。当你计算扭转方差时，你会发现它们采用了无偏标志，即选择除以n还是n减1。令人困惑的是，文档没有说明无偏的默认设置是什么，但我认为默认情况下无偏是开启的。我不明白为什么这里的文档没有提到这一点。

现在，关于一维批量归一化（batch norm 1D）的文档又出现了错误和令人困惑的地方。文档中说标准差是通过有偏估计量计算的，但这实际上并不完全正确。自那以后，人们已经在多个问题中指出这是错误的。实际上，兔子洞更深，他们完全按照论文操作。训练时使用的是有偏版本，但在估算运行标准差时却采用了无偏版本。

所以，又出现了训练测试不匹配的问题。长话短说，我不喜欢训练测试之间的差异。我基本上认为，我们使用的是有偏见的训练版本和无偏见的测试版本。我基本上认为这是一个错误。而且我不认为这有什么充分的理由。他们在论文中并没有详细解释背后的原因。

这就是为什么我基本上更倾向于在自己的工作中使用贝塞尔校正。遗憾的是，BatchNorm并没有提供一个关键字参数来让你选择在训练测试中使用无偏版本还是有偏版本。因此，在我看来，任何使用BatchNormalization的人，代码里基本上都存在一点小问题。

而如果小批量的大小稍大一些，这个问题就会小得多。但我仍然觉得这有点难以接受。所以也许有人能解释一下为什么这样做是可以的。但目前，我倾向于在训练和测试时都一致使用无偏版本。这就是为什么我在这里使用1除以n减1。好的，那么现在让我们实际反向传播这一行。

所以我总是喜欢做的第一件事就是先仔细检查形状。特别是这里，看看涉及到的形状，我发现bnvar的形状是1×64。所以它是一个行向量，而bndiff2的形状是32×64。很明显，这里我们正在对第0轴进行求和，以压缩形状的第一个维度。这立刻让我想到在反向传播过程中会有某种复制或广播操作。也许你已经注意到了这里的模式。

但基本上，在前向传播中只要有求和操作，在反向传播中就会沿着同一维度变为复制或广播。反之，若前向传播中存在复制或广播操作，则表明存在变量复用。因此在反向传播时，该操作就会转变为沿完全相同维度的求和运算。

因此，希望你能注意到这种二元性，即前向传播和反向传播中这两者某种程度上是相互对立的。现在，一旦我们理解了形状，接下来我总喜欢做的是在脑海中设想一个简单的例子，以便大致理解数学公式中变量的依赖关系是怎样的。在这里，我们最终得到一个二维数组，其大小为2，我们用一个常数对其进行缩放。

然后我们对列进行垂直求和。如果我们有一个2x2的矩阵A，然后我们对列进行求和并缩放，就会得到一个行向量b1, b2。b1以这种方式依赖于A，即对A的第一列求和并缩放。而b2则以这种方式依赖于A，即对第二列求和并缩放。

因此，从根本上来看，我们现在要做的是，我们已经得到了关于b1和b2的导数，现在需要将它们反向传播回A的导数。显然，通过简单的头脑微分，这里的局部导数对于每一个A来说都是1/(n-1)乘以1。基本上，b1的导数必须通过A的列进行流动，并按1/(n-1)进行缩放。这就是这里大致发生的情况。

直观地说，导数流告诉我们 dbn_diff2 将是这个操作的局部导数。顺便说一句，有很多方法可以做到这一点，但我喜欢这样做：torch.ones_like(bn_diff2)。所以我将创建一个由1组成的二维大数组。然后我会对其进行缩放。即1.0除以n减1。这样得到一个1/(n-1)的数组。这有点像局部导数。

现在轮到链式法则了，我只需将其乘以dbn var。注意接下来会发生什么。这个矩阵是32×64的，而这个只是1×64的。所以我让广播机制来完成复制操作，因为在PyTorch内部，这个1×64的行向量dbn var会在这个乘法运算中被垂直复制，直到两个张量形状相同。然后会进行逐元素相乘。因此广播机制实际上就是在执行复制功能。

最终我将在这里得到dbn diff2的导数。这就是候选解。我们把它移到这里来。让我们取消这一行的注释，检查一下。希望一切顺利。果然，我们看到这就是正确的公式。接下来，我们在这里对bn diff进行微分。这里我们将bn diff逐元素平方，得到bn diff2。这是一个相对简单的导数，因为它只是一个简单的逐元素运算。

所以这有点像标量的情况。我们有dbn_diff应该是，如果这是x的平方，那么它的导数就是2x，对吧？所以简单地就是2乘以bn_diff。这就是局部导数。然后应用链式法则。它们的形状相同。它们的形状是一样的。所以这次就这样。这就是这个变量的反向传播。让我把它记在这里。

现在我们必须小心，因为我们已经计算过dbn_diff了，对吧？这只是另一条分支的终点又回到了bn_diff。因为bn_diff已经通过反向传播从bn_raw一路传到这里了。这样我们就完成了第二条分支的计算。这就是为什么我必须使用加等于操作。如果你还记得的话，之前我们对bn diff的导数计算有误。我希望在补充上这最后缺失的部分后，我们就能得到完全正确的结果。

那就让我们跑起来吧。bn diff2，bn diff现在实际上显示了完全正确的导数。这让人感到欣慰。好的，现在让我们反向传播到这一行。首先，我们当然要检查形状。我已经在这里写出来了。基本上，这个形状是32乘64。H prebn的形状相同。但bn mean i是一个行向量，1乘64。

所以这里的减号实际上会进行广播操作。因此我们必须小心处理这一点。作为提示，由于对偶性，前向传播中的广播意味着变量的重复使用。因此，在反向传播过程中会有一个求和运算。现在让我们在这里写出反向传播的过程。反向传播到H_prebn。

因为它们的形状相同，所以这里每个元素的局部导数在对应元素处仅为1。基本上，这意味着梯度只是简单地复制。这只是一个变量赋值。这是平等。所以为了安全起见，我将克隆这个张量，以创建一个精确的副本。dbn diff。

然后在这里，为了反向传播到这个部分，我倾向于在这里做的是 dbn_mean，我基本上会是局部导数是什么？嗯，它就是负的 torch.ones，形状与 bn_diff 相同，对吧？然后乘以这里的导数 dbn_diff。而这里就是复制后的 bn_mean_i 的反向传播。所以我仍然需要通过广播中的复制进行反向传播。我通过求和来实现这一点。

所以我要把整个这个东西拿出来，然后对第零维度（也就是复制的那个维度）求和。顺便说一句，如果你仔细看的话，会发现这个形状和那个是一样的。因此我在这里做的操作其实没有太大意义，因为这只不过是一个由1组成的数组在乘dbn_diff。

所以实际上，我可以直接这样做，效果是等价的。这就是候选的反向传播算法。让我把它复制到这里。然后让我注释掉这个。还有这个。回车。这是错的。该死。其实，抱歉，这本来就是错的。这被认为是错误的，因为我们是从 bn diff 反向传播到 h prebn。但我们还没有完成，因为 bn 均值 i 依赖于 h prebn。这部分导数还会有第二部分来自第二个分支。

所以我们还没完成。而且我们预计会有错误。就是这样。现在让我们从bn_mean_i反向传播到h_prebn。这里我们仍需谨慎，因为存在沿第0维度的广播操作，或者说沿该维度求和。因此在反向传播过程中，这会转换为广播机制。

我会在这一行加快一点速度，因为这与我们之前处理过的线条非常相似，实际上就是之前的线条乘以某个因子。因此，dh_prebn 将会是梯度按 1/n 的比例缩放。然后，这里的 dbn_mean_i 梯度基本上也会按 1/n 的比例缩放。接着，它会流经所有列，最终汇集到 dh_prebn 中。所以我们需要的是这个量按 1/n 的比例缩放。让我先把常数项放在前面。

因此，我们需要缩小梯度。现在我们需要在所有行上复制它。我喜欢通过torch.oncelike函数来实现，基本上类似于h_prebn。我会让广播来完成复制工作。就像这样。所以这是dh prebn。希望我们可以加上这个。所以这里是广播，然后这是缩放。所以这应该是正确的。好的。

这样，Bastrom层的反向传播就完成了，我们现在到了这一步。接下来，我们来反向传播这里的第一个线性层。由于垂直方向的内容开始变得有点混乱，我在这里复制粘贴了这条线，我们就针对这一条线进行反向传播。

首先，我们当然要检查形状，可以看到这是32乘64。mcat是32乘30，w1是30乘64，而b1只有64。正如我提到的，通过线性层进行反向传播相当简单，只需匹配形状即可。


So let's do that. We have that dmcat should be some matrix multiplication of dhprebn with w1, and one transpose thrown in there. So to make mcat be 32 by 30, I need to take dhprebn, 32 by 64, and multiply it by w1 dot transpose.

To get dw1, I need to end up with 30 by 64. So to get that, I need to take mcat transpose and multiply that by dhprebn. Finally, to get db1, this is an addition.

We saw that basically I need to just sum the elements in dhprebn along some dimension. To make the dimensions work out, I need to sum along the 0th axis here to eliminate this dimension, and we do not keep dims, so that we want to just get a single one-dimensional vector of 64. So these are the claimed derivatives.

Let me put that here, and let me uncomment three lines and cross our fingers. Everything is great. Okay, so we now continue.

Almost there. We have the derivative of mcat, and we want to backpropagate into mb. I again copied this line over here.

This is the forward pass, and then this is the shapes. Remember that the shape here was 32 by 30, and the original shape of mb was 32 by 3 by 10. This layer in the forward pass, as you recall, did the concatenation of these three 10-dimensional character vectors.

Now we just want to undo that. This is actually a relatively straightforward operation, because the backward pass of the... What is a view? A view is just a representation of the array. It's just a logical form of how you interpret the array.

So let's just reinterpret it to be what it was before. So in other words, dmb is not 32 by 30. It is basically dmbcat, but if you view it as the original shape.

So just mdot shape. You can pass in tuples into view, and so this should just be... Okay, we just re-represent that view, and then we uncomment this line here, and hopefully... Yeah, so the derivative of mb is correct. So in this case, we just have to re-represent the shape of those derivatives into the original view.

So now we are at the final line, and the only thing that's left to backpropagate through is this indexing operation here. msc at xb. So as I did before, I copy-pasted this line here, and let's look at the shapes of everything that's involved and remind ourselves how this worked.

So mdot shape was 32 by 3 by 10. So it's 32 examples, and then we have three characters. Each one of them has a 10-dimensional embedding, and this was achieved by taking the lookup table C, which have 27 possible characters, each of them 10-dimensional, and we looked up at the rows that were specified inside this tensor xb.

So xb is 32 by 3, and it's basically giving us, for each example, the identity or the index of which character is part of that example. And so here I'm showing the first five rows of this tensor xb. And so we can see that, for example, here it was the first example in this batch is that the first character and the first character and the fourth character comes into the neural net, and then we want to predict the next character in a sequence after the character is 1, 1, 4. So basically what's happening here is there are integers inside xb, and each one of these integers is specifying which row of C we want to pluck out, right? And then we arrange those rows that we've plucked out into a 32 by 3 by 10 tensor, and we just package them into this tensor.

And now what's happening is that we have dimp. So for every one of these basically plucked out rows, we have their gradients now. But they're arranged inside this 32 by 3 by 10 tensor.

So all we have to do now is we just need to route this gradient backwards through this assignment. So we need to find which row of C did every one of these 10-dimensional embeddings come from. And then we need to deposit them into dc.

So we just need to undo the indexing. And of course, if any of these rows of C was used multiple times, which almost certainly is the case, like the row 1 and 1 was used multiple times, then we have to remember that the gradients that arrive there have to add. So for each occurrence, we have to have an addition.

So let's now write this out. And I don't actually know of a much better way to do this than a for loop, unfortunately, in Python. So maybe someone can come up with a vectorized efficient operation.

But for now, let's just use for loops. So let me create a torch.zeros like C to initialize just a 27 by 10 tensor of all zeros. And then honestly, for k in range, xb.shape at 0. Maybe someone has a better way to do this.

For j in range, xb.shape at 1. This is going to iterate over all the elements of xb, all these integers. And then let's get the index at this position. So the index is basically xb at kj.

So an example of that is 11 or 14 and so on. And now in the forward pass, we basically took the row of C at index, and we deposited it into m at kj. That's what happened.

That's where they are packaged. So now we need to go backwards. And we just need to route dm at the position kj.

We now have these derivatives for each position. And it's 10-dimensional. And you just need to go into the correct row of C. So dc, rather, at ix is this.

But plus equals, because there could be multiple occurrences, like the same row could have been used many, many times. And so all of those derivatives will just go backwards through the indexing, and they will add. So this is my candidate solution.

Let's copy it here. Let's uncomment this and cross our fingers. Yay.

So that's it. We've backpropagated through this entire beast. So there we go.

Totally makes sense. So now we come to exercise two. It basically turns out that in this first exercise, we were doing way too much work.

We were backpropagating way too much. And it was all good practice and so on. But it's not what you would do in practice.

And the reason for that is, for example, here I separated out this loss calculation over multiple lines. And I broke it up all to its smallest atomic pieces. And we backpropagated through all of those individually.

But it turns out that if you just look at the mathematical expression for the loss, then actually you can do the differentiation on pen and paper. And a lot of terms cancel and simplify. And the mathematical expression you end up with can be significantly shorter and easier to implement than backpropagating through all the little pieces of everything you've done.

So before we had this complicated forward pass going from logits to the loss. But in PyTorch, everything can just be glued together into a single call at that cross entropy. You just pass in logits and the labels, and you get the exact same loss as I verify here.

So our previous loss and the fast loss coming from the chunk of operations as a single mathematical expression is the same, but it's much, much faster in a forward pass. It's also much, much faster in backward pass. And the reason for that is if you just look at the mathematical form of this and differentiate again, you will end up with a very small and short expression.

So that's what we want to do here. We want to, in a single operation or in a single go, or like very quickly, go directly into dlogits. And we need to implement dlogits as a function of logits and ybs.

But it will be significantly shorter than whatever we did here, where to get to dlogits, we had to go all the way here. So all of this work can be skipped in a much, much simpler mathematical expression that you can implement here. So you can give it a shot yourself, basically look at what exactly is the mathematical expression of loss and differentiate with respect to the logits.

So let me show you a hint. You can, of course, try it fully yourself. But if not, I can give you some hint of how to get started mathematically.

So basically, what's happening here is we have logits. Then there's a softmax that takes the logits and gives you probabilities. Then we are using the identity of the correct next character to pluck out a row of probabilities, take the negative log of it to get our negative log probability.

And then we average up all the log probabilities or negative log probabilities to get our loss. So basically, what we have is for a single individual example, rather, we have that loss is equal to negative log probability, where p here is kind of like thought of as a vector of all the probabilities. So at the y-th position, where y is the label.

And we have that p here, of course, is the softmax. So the i-th component of p, of this probability vector, is just the softmax function. So raising all the logits basically to the power of e and normalizing.

So everything sums to one. Now, if you write out p of y here, you can just write out the softmax. And then basically, what we're interested in is we're interested in the derivative of the loss with respect to the i-th logit.

And so basically, it's a d by d li of this expression here, where we have l indexed with the specific label y. And on the bottom, we have a sum over j of e to the lj and the negative log of all that. So potentially, give it a shot, pen and paper, and see if you can actually derive the expression for the loss by d li. And then we're going to implement it here.

Okay, so I'm going to give away the result here. So this is some of the math I did to derive the gradients analytically. And so we see here that I'm just applying the rules of calculus from your first or second year of bachelor's degree, if you took it.

And we see that the expressions actually simplify quite a bit. You have to separate out the analysis in the case where the i-th index that you're interested in inside logits is either equal to the label or it's not equal to the label. And then the expressions simplify and cancel in a slightly different way.

And what we end up with is something very, very simple. We either end up with basically p at i, where p is, again, this vector of probabilities after a softmax, or p at i minus 1, where we just simply subtract to 1. But in any case, we just need to calculate the softmax p. And then in the correct dimension, we need to subtract to 1. And that's the gradient, the form that it takes analytically. So let's implement this basically.

And we have to keep in mind that this is only done for a single example. But here we are working with batches of examples. So we have to be careful of that.

And then the loss for a batch is the average loss over all the examples. So in other words, is the example for all the individual examples, is the loss for each individual example summed up and then divided by n. And we have to backpropagate through that as well and be careful with it. So d logits is going to be f dot softmax.

PyTorch has a softmax function that you can call. And we want to apply the softmax on the logits. And we want to go in the dimension that is 1. So basically, we want to do the softmax along the rows of these logits.

Then at the correct positions, we need to subtract a 1. So d logits at iterating over all the rows and indexing into the columns provided by the correct labels inside YB, we need to subtract 1. And then finally, it's the average loss that is the loss. And in the average, there's a 1 over n of all the losses added up. And so we need to also backpropagate through that division.

So the gradient has to be scaled down by n as well because of the mean. But this otherwise should be the result. So now if we verify this, we see that we don't get an exact match.

But at the same time, the maximum difference from logits from PyTorch and rd logits here is on the order of 5e negative 9. So it's a tiny, tiny number. So because of floating point wonkiness, we don't get the exact bitwise result, but we basically get the correct answer approximately. Now I'd like to pause here briefly before we move on to the next exercise because I'd like us to get an intuitive sense of what d logits is because it has a beautiful and very simple explanation, honestly.

So here I'm taking d logits and I'm visualizing it. And we can see that we have a batch of 32 examples of 27 characters. And what is d logits intuitively, right? d logits is the probabilities that the probabilities matrix in the forward pass.

But then here, these black squares are the positions of the correct indices where we subtracted a 1. And so what is this doing, right? These are the derivatives on d logits. And so let's look at just the first row here. So that's what I'm doing here.

I'm calculating the probabilities of these logits, and then I'm taking just the first row. And this is the probability row. And then d logits of the first row and multiplying by n, just for us so that we don't have the scaling by n in here and everything is more interpretable.

We see that it's exactly equal to the probability, of course, but then the position of the correct index has a minus equals 1. So minus 1 on that position. And so notice that if you take d logits at 0 and you sum it, it actually sums to 0. And so you should think of these gradients here at each cell as like a force. We are going to be basically pulling down on the probabilities of the incorrect characters, and we're going to be pulling up on the probability at the correct index.

And that's what's basically happening in each row. And the amount of push and pull is exactly equalized because the sum is 0. So the amount to which we pulled down on the probabilities and the amount that we push up on the probability of the correct character is equal. So it's sort of the repulsion and the attraction are equal.

And think of the neural net now as like a massive pulley system or something like that. We're up here on top of d logits, and we're pulling down the probabilities of incorrect and pulling up the probability of the correct. And in this complicated pulley system, because everything is mathematically just determined, just think of it as sort of like this tension translating to this complicating pulley mechanism.

And then eventually we get a tug on the weights and the biases. And basically in each update, we just kind of like tug in the direction that we like for each of these elements. And the parameters are slowly given in to the tug.

And that's what training a neural net kind of like looks like on a high level. And so I think the forces of push and pull in these gradients are actually very intuitive here. We're pushing and pulling on the correct answer and the incorrect answers.

And the amount of force that we're applying is actually proportional to the probabilities that came out in the forward pass. And so, for example, if our probabilities came out exactly correct, so they would have had zero everywhere except for one at the correct position, then the d logits would be all a row of zeros for that example. There would be no push and pull.

So the amount to which your prediction is incorrect is exactly the amount by which you're going to get a pull or a push in that dimension. So if you have, for example, a very confidently mispredicted element here, then what's going to happen is that element is going to be pulled down very heavily. And the correct answer is going to be pulled up to the same amount.

And the other characters are not going to be influenced too much. So the amount to which you mispredict is then proportional to the strength of the pull. And that's happening independently in all the dimensions of this tensor.

And it's very intuitive and very easy to think through. And that's basically the magic of the cross-entropy loss and what it's doing dynamically in the backward pass of the neural net. So now we get to exercise number three, which is a very fun exercise, depending on your definition of fun.

And we are going to do for batch normalization exactly what we did for cross-entropy loss in exercise number two. That is, we are going to consider it as a glued single mathematical expression and back-propagate through it in a very efficient manner because we are going to derive a much simpler formula for the backward pass of batch normalization. And we're going to do that using pen and paper.

So previously, we've broken up batch normalization into all of the little intermediate pieces and all the atomic operations inside it. And then we back-propagated through it one by one. Now we just have a single sort of forward pass of a batch norm.

And it's all glued together. And we see that we get the exact same result as before. Now for the backward pass, we'd like to also implement a single formula, basically, for back-propagating through this entire operation, that is the batch normalization.

So in the forward pass previously, we took H pre-bn, the hidden states of the pre-batch normalization, and created H pre-act, which is the hidden states just before the activation. In the batch normalization paper, H pre-bn is X and H pre-act is Y. So in the backward pass, what we'd like to do now is we have the H pre-act and we'd like to produce D H pre-bn. And we'd like to do that in a very efficient manner.

So that's the name of the game. Calculate D H pre-bn given D H pre-act. And for the purposes of this exercise, we're going to ignore gamma and beta and their derivatives because they take on a very simple form in a very similar way to what we did up above.

So let's calculate this given that right here. So to help you a little bit, like I did before, I started off the implementation here on pen and paper. And I took two sheets of paper to derive the mathematical formulas for the backward pass.

And basically to set up the problem, just write out the mu sigma square variance, X I hat and Y I, exactly as in the paper, except for the Bessel correction. And then in the backward pass, we have the derivative of the loss with respect to all the elements of Y. And remember that Y is a vector. There's multiple numbers here.

So we have all the derivatives with respect to all the Ys. And then there's a gamma and a beta. And this is kind of like the compute graph.

The gamma and the beta, there's the X hat, and then the mu and the sigma square and the X. So we have D L by D Y I, and we want D L by D X I for all the I's in these vectors. So this is the compute graph. And you have to be careful because I'm trying to note here that these are vectors.

There's many nodes here inside X, X hat and Y, but mu and sigma, sorry, sigma square are just individual scalars, single numbers. So you have to be careful with that. You have to imagine there's multiple nodes here or you're going to get your math wrong.

So as an example, I would suggest that you go in the following order, one, two, three, four, in terms of the backpropagation. So backpropagate into X hat, then into sigma square, then into mu and then into X. Just like in a topological sort in micro grad, we would go from right to left. You're doing the exact same thing, except you're doing it with symbols and on a piece of paper.

So for number one, I'm not giving away too much. If you want D L of D X I hat, then we just take D L by D Y I and multiply it by gamma because of this expression here, where any individual Y I is just gamma times X I hat plus beta. So didn't help you too much there, but this gives you basically the derivatives for all the X hats.

And so now try to go through this computational graph and derive what is D L by D sigma square? And then what is D L by D mu? And then what is D L by D X eventually? So give it a go and I'm going to be revealing the answer one piece at a time. Okay, so to get D L by D sigma square, we have to remember again, like I mentioned, that there are many X's, X hats here. And remember that sigma square is just a single individual number here.

So when we look at the expression for D L by D sigma square, we have that, we have to actually consider all the possible paths that we basically have that there's many X hats and they all feed off from, they all depend on sigma square. So sigma square has a large fan out. There's lots of arrows coming out from sigma square into all the X hats.

And then there's a back propagating signal from each X hat into sigma square. And that's why we actually need to sum over all those I's from I equal to one to M of the D L by D X I hat, which is the global gradient times the X I hat by D sigma square, which is the local gradient of this operation here. And then mathematically, I'm just working it out here and I'm simplifying and you get a certain expression for D L by D sigma square.

And we're going to be using this expression when we back propagate into mu and then eventually into X. So now let's continue our back propagation into mu. So what is D L by D mu? Now, again, be careful that mu influences X hat and X hat is actually lots of values. So for example, if our mini batch size is 32, as it is in our example that we were working on, then this is 32 numbers and 32 arrows going back to mu.

And then mu going to sigma square is just a single arrow because sigma square is a scalar. So in total, there are 33 arrows emanating from mu and then all of them have gradients coming into mu and they all need to be summed up. And so that's why when we look at the expression for D L by D mu, I am summing up over all the gradients of D L by D X I hat times the X I hat by D mu.

So that's this arrow and that's 32 arrows here and then plus the one arrow from here, which is D L by D sigma square times D sigma square by D mu. So now we have to work out that expression and let me just reveal the rest of it. Simplifying here is not complicated, the first term, and you just get an expression here.

For the second term, though, there's something really interesting that happens. When we look at D sigma square by D mu and we simplify, at one point, if we assume that in a special case where mu is actually the average of X I's, as it is in this case, then if we plug that in, then actually the gradient vanishes and becomes exactly zero. And that makes the entire second term cancel.

And so these, if you just have a mathematical expression like this and you look at D sigma square by D mu, you would get some mathematical formula for how mu impacts sigma square. But if it is the special case that mu is actually equal to the average, as it is in the case of batch normalization, that gradient will actually vanish and become zero. So the whole term cancels and we just get a fairly straightforward expression here for D L by D mu.

And now we get to the craziest part, which is deriving D L by D X I, which is ultimately what we're after. Now let's count, first of all, how many numbers are there inside X? As I mentioned, there are 32 numbers. There are 32 little X I's.

And let's count the number of arrows emanating from each X I. There's an arrow going to mu, an arrow going to sigma square, and then there's an arrow going to X hat. But this arrow here, let's scrutinize that a little bit. Each X I hat is just a function of X I and all the other scalars.

So X I hat only depends on X I and none of the other X's. And so therefore, there are actually in this single arrow, there are 32 arrows, but those 32 arrows are going exactly parallel. They don't interfere.

They're just going parallel between X and X hat. You can look at it that way. And so how many arrows are emanating from each X I? There are three arrows, mu, sigma square, and the associated X hat.

And so in backpropagation, we now need to apply the chain rule and we need to add up those three contributions. So here's what that looks like. If I just write that out, we have, we're going through, we're chaining through mu, sigma square, and through X hat.

And those three terms are just here. Now we already have three of these. We have D L by D X I hat.

We have D L by D mu, which we derived here. And we have D L by D sigma square, which we derived here. But we need three other terms here.

The, this one, this one, and this one. So I invite you to try to derive them. It's not that complicated.

You're just looking at these expressions here and differentiating with respect to X I. So give it a shot, but here's the result. Or at least what I got. Yeah, I'm just, I'm just differentiating with respect to X I for all of these expressions.

And honestly, I don't think there's anything too tricky here. It's basic calculus. Now what gets a little bit more tricky is we are now going to plug everything together.

So all of these terms multiplied with all of these terms and add it up according to this formula. And that gets a little bit hairy. So what ends up happening is you get a large expression and the thing to be very careful with here, of course, is we are working with a DL by D X I for a specific I here.

But when we are plugging in some of these terms, like say this term here, DL by D Sigma squared, you see how DL by D Sigma squared, I end up with an expression and I'm iterating over little I's here, but I can't use I as the variable when I plug in here, because this is a different I from this I. This I here is just a place like a local variable for a for loop in here. So here when I plug that in, you notice that I renamed the I to a J because I need to make sure that this J is not this I. This J is like a little local iterator over 32 terms. And so you have to be careful with that when you're plugging in the expressions from here to here, you may have to rename I's into J's.

And you have to be very careful what is actually an I with respect to DL by D X I. So some of these are J's, some of these are I's, and then we simplify this expression. And I guess like the big thing to notice here is a bunch of terms just going to come out to the front and you can refactor them. There's a sigma squared plus epsilon raised to the power of negative three over two.

This sigma squared plus epsilon can be actually separated out into three terms. Each of them are sigma squared plus epsilon to the negative one over two. So the three of them multiplied is equal to this.

And then those three terms can go different places. Because of the multiplication. So one of them actually comes out to the front and will end up here outside.

One of them joins up with this term and one of them joins up with this other term. And then when you simplify the expression, you'll notice that some of these terms that are coming out are just the X I hats. So you can simplify just by rewriting that.

And what we end up with at the end is a fairly simple mathematical expression over here that I cannot simplify further. But basically you'll notice that it only uses the stuff we have and it derives the thing we need. So we have D L by D Y for all the I's and those are used plenty of times here.

And also in addition, what we're using is these X I hats and X J hats, and they just come from the forward pass. And otherwise, this is a simple expression and it gives us D L by D.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)



(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

So that's the end of BatchNorm backward pass analytically. Let's now implement this final result. Okay, so I implemented the expression into a single line of code here, and you can see that the max diff is tiny, so this is the correct implementation of this formula.

Now, I'll just basically tell you that getting this formula here from this mathematical expression was not trivial, and there's a lot going on packed into this one formula, and this is a whole exercise by itself, because you have to consider the fact that this formula here is just for a single neuron and a batch of 32 examples, but what I'm doing here is I'm actually, we actually have 64 neurons, and so this expression has to in parallel evaluate the BatchNorm backward pass for all of those 64 neurons in parallel independently. So this has to happen basically in every single column of the inputs here, and in addition to that, you see how there are a bunch of sums here, and we need to make sure that when I do those sums, that they broadcast correctly onto everything else that's here, and so getting this expression is just like highly non-trivial, and I invite you to basically look through it and step through it, and it's a whole exercise to make sure that this checks out, but once all the shapes agree, and once you convince yourself that it's correct, you can also verify that PyTorch gets the exact same answer as well, and so that gives you a lot of peace of mind that this mathematical formula is correctly implemented here and broadcasted correctly and replicated in parallel for all of the 64 neurons inside this BatchNorm layer. Okay, and finally, exercise number four asks you to put it all together, and here we have a redefinition of the entire problem, so you see that we reinitialized the neural net from scratch and everything, and then here, instead of calling loss.backward, we want to have the manual backpropagation here as we derived it up above.

So go up, copy-paste all the chunks of code that we've already derived, put them here, and derive your own gradients, and then optimize this neural net basically using your own gradients all the way to the calibration of the BatchNorm and the evaluation of the loss, and I was able to achieve quite a good loss, basically the same loss you would achieve before, and that shouldn't be surprising because all we've done is we've really gone into loss.backward and we've pulled out all the code and inserted it here, but those gradients are identical and everything is identical and the results are identical. It's just that we have full visibility on exactly what goes on under the hood of loss.backward in this specific case. Okay, and this is all of our code.

This is the full backward pass using basically the simplified backward pass for the cross-entropy loss and the BatchNormalization. So backpropagating through cross-entropy, the second layer, the 10H nonlinearity, the BatchNormalization through the first layer and through the embedding, and so you see that this is only maybe, what is this, 20 lines of code or something like that, and that's what gives us gradients, and now we can potentially erase loss.backward. So the way I have the code set up is you should be able to run this entire cell once you fill this in, and this will run for only 100 iterations and then break, and it breaks because it gives you an opportunity to check your gradients against PyTorch. So here, our gradients we see are not exactly equal.

They are approximately equal, and the differences are tiny, one in negative nine or so, and I don't exactly know where they're coming from, to be honest. So once we have some confidence that the gradients are basically correct, we can take out the gradient checking, we can disable this breaking statement, and then we can basically disable loss.backward. We don't need it anymore. Feels amazing to say that.

And then here, when we are doing the update, we're not gonna use p.grad. This is the old way of PyTorch. We don't have that anymore because we're not doing backward. We are going to use this update where you see that I'm iterating over, I've arranged the grads to be in the same order as the parameters, and I'm zipping them up, the gradients and the parameters, into p and grad, and then here I'm going to step with just the grad that we derive manually.

So the last piece is that none of this now requires gradients from PyTorch. And so one thing you can do here is you can do withTorch.noGrad and offset this whole code block. And really what you're saying is you're telling PyTorch that, hey, I'm not gonna call backward on any of this.

And this allows PyTorch to be a bit more efficient with all of it. And then we should be able to just run this. And it's running.

And you see that loss of backward is commented out and we're optimizing. So we're going to leave this run and hopefully we'll get a good result. Okay, so I allowed the neural net to finish optimization.

Then here I calibrate the bastion parameters because I did not keep track of the running mean and variance in their training loop. Then here I ran the loss and you see that we actually obtained a pretty good loss, very similar to what we've achieved before. And then here I'm sampling from the model and we see some of the name-like gibberish that we're sort of used to.

So basically the model worked and samples pretty decent results compared to what we were used to. So everything is the same, but of course the big deal is that we did not use lots of backward. We did not use PyTorch autograd and we estimated our gradients ourselves by hand.

And so hopefully you're looking at this, the backward pass of this neural net and you're thinking to yourself, actually, that's not too complicated. Each one of these layers is like three lines of code or something like that. And most of it is fairly straightforward, potentially with the notable exception of the bastion normalization backward pass.

Otherwise it's pretty good. Okay, and that's everything I wanted to cover for this lecture. So hopefully you found this interesting and what I liked about it honestly is that it gave us a very nice diversity of layers to back propagate through.

And I think it gives a pretty nice and comprehensive sense of how these backward passes are implemented and how they work. And you'd be able to derive them yourself, but of course in practice, you probably don't want to and you want to use the PyTorch autograd. But hopefully you have some intuition about how gradients flow backwards through the neural net, starting at the loss and how they flow through all the variables and all the intermediate results.

And if you understood a good chunk of it, and if you have a sense of that, then you can count yourself as one of these buff dojis on the left instead of the dojis on the right here. Now, in the next lecture, we're actually going to go to recurrent neural nets, LSTMs and all the other variants of RNNs. And we're going to start to complexify the architecture and start to achieve better log likelihoods.

And so I'm really looking forward to that and I'll see you then. Hi everyone. Today we are continuing our implementation of MakeMore, our favorite character level language model.

Now, you'll notice that the background behind me is different. That's because I am in Kyoto and it is awesome. So I'm in a hotel room here.

Now, over the last few lectures, we've built up to this architecture that is a multi-layer perceptron character level language model. So we see that it receives three previous characters and tries to predict the fourth character in a sequence using a very simple multi-layer perceptron using one hidden layer of neurons with 10H0 neuralties. So what I'd like to do now in this lecture is I'd like to complexify this architecture.

In particular, we would like to take more characters in a sequence as an input, not just three. And in addition to that, we don't just want to feed them all into a single hidden layer because that squashes too much information too quickly. Instead, we would like to make a deeper model that progressively fuses this information to make its guess about the next character in a sequence.

And so we'll see that as we make this architecture more complex, we're actually going to arrive at something that looks very much like a WaveNet. So WaveNet is this paper published by DeepMind in 2016. And it is also a language model basically, but it tries to predict audio sequences instead of character level sequences or word level sequences.

But fundamentally, the modeling setup is identical. It is an autoregressive model and it tries to predict the next character in a sequence. And the architecture actually takes this interesting hierarchical sort of approach to predicting the next character in a sequence with this tree-like structure.

And this is the architecture and we're going to implement it in the course of this video. So let's get started. So the starter code for part five is very similar to where we ended up in part three.

Recall that part four was the manual backpropagation exercise that is kind of an aside. So we are coming back to part three, copy pasting chunks out of it. And that is our starter code for part five.

I've changed very few things otherwise. So a lot of this should look familiar to you if you've gone through part three. So in particular, very briefly, we are doing imports.

We are reading our dataset of words and we are processing the dataset of words into individual examples. And none of this data generation code has changed. And basically we have lots and lots of examples.

In particular, we have 182,000 examples of three characters trying to predict the fourth one. And we've broken up every one of these words into little problems of given three characters predict the fourth one. So this is our dataset and this is what we're trying to get the neural net to do.

Now, in part three, we started to develop our code around these layer modules that are, for example, a class linear. And we're doing this because we want to think of these modules as building blocks and like a Lego building block bricks that we can sort of like stack up into neural networks. And we can feed data between these layers and stack them up into sort of graphs.

Now, we also developed these layers to have APIs and signatures very similar to those that are found in PyTorch. So we have torch.nn and it's got all these layer building blocks that you would use in practice. And we were developing all of these to mimic the APIs of these.

So for example, we have linear. So there will also be a torch.nn.linear and its signature will be very similar to our signature and the functionality will be also quite identical as far as I'm aware. So we have the linear layer with the bastion 1D layer and the 10H layer that we developed previously.

And linear just does a matrix multiply in the forward pass of this module. BatchNorm, of course, is this crazy layer that we developed in the previous lecture. And what's crazy about it is, well, there's many things.

Number one, it has these running mean and variances that are trained outside of backpropagation. They are trained using exponential moving average inside this layer when we call the forward pass. In addition to that, there's this training flag because the behavior of BatchNorm is different during train time and evaluation time.

And so suddenly we have to be very careful that BatchNorm is in its correct state, that it's in the evaluation state or training state. So that's something to now keep track of, something that sometimes introduces bugs because you forget to put it into the right mode. And finally, we saw that BatchNorm couples the statistics or the activations across the examples in the batch.

So normally we thought of the batch as just an efficiency thing, but now we are coupling the computation across batch elements, and it's done for the purposes of controlling the activation statistics as we saw in the previous video. So it's a very weird layer, at least to a lot of bugs, partly, for example, because you have to modulate the training and eval phase and so on. In addition, for example, you have to wait for the mean and the variance to settle and to actually reach a steady state.

And so you have to make sure that you, basically there's state in this layer and state is harmful usually. Now I brought out the generator object. Previously we had a generator equals G and so on inside these layers.

I've discarded that in favor of just initializing the torch RNG outside here just once globally just for simplicity. And then here we are starting to build out some of the neural network elements. This should look very familiar.

We have our embedding table C and then we have a list of layers and it's a linear, feeds to batch or feeds to 10H and then a linear output layer and its weights are scaled down so we are not confidently wrong at initialization. We see that this is about 12,000 parameters. We're telling PyTorch that the parameters require gradients.

The optimization is as far as I'm aware, identical and should look very, very familiar. Nothing changed here. Loss function looks very crazy.

We should probably fix this and that's because 32 batch elements are too few and so you can get very lucky or unlucky in any one of these batches and it creates a very thick loss function. So we're gonna fix that soon. Now, once we want to evaluate the trained neural network we need to remember because of the batch norm layers to set all the layers to be training equals false.

So this only matters for the batch norm layer so far and then we evaluate. We see that currently we have a validation loss of 2.10 which is fairly good but there's still a ways to go but even at 2.10, we see that when we sample from the model we actually get relatively name-like results that do not exist in a training set. So for example, Yvonne, Kilo, Pras, Alaya, et cetera.

So certainly not unreasonable I would say but not amazing and we can still push this validation loss even lower and get much better samples that are even more name-like. So let's improve this model now. Okay, first let's fix this graph because it is daggers in my eyes and I just can't take it anymore.

So loss I, if you recall, is a Python list of floats. So for example, the first 10 elements look like this. Now, what we'd like to do basically is we need to average up some of these values to get a more sort of representative value along the way.

So one way to do this is the following. In PyTorch, if I create, for example, a tensor of the first 10 numbers, then this is currently a one-dimensional array but recall that I can view this array as two-dimensional. So for example, I can view it as a two-by-five array and this is a 2D tensor now, two-by-five.

And you see what PyTorch has done is that the first row of this tensor is the first five elements and the second row is the second five elements. I can also view it as a five-by-two as an example. And then recall that I can also use negative one in place of one of these numbers and PyTorch will calculate what that number must be in order to make the number of elements work out.

So this can be this or like that, both will work. Of course, this would not work. Okay, so this allows it to spread out some of the consecutive values into rows.

So that's very helpful because what we can do now is first of all, we're going to create a Torch.tensor out of the list of floats. And then we're going to view it as whatever it is but we're going to stretch it out into rows of 1,000 consecutive elements. So the shape of this now becomes 200 by 1,000 and each row is 1,000 consecutive elements in this list.

That's very helpful because now we can do a mean along the rows and the shape of this will just be 200. And so we've taken basically the mean on every row. So plt.plot of that should be something nicer.

That's better. So we see that we've basically made a lot of progress and then here, this is the learning rate decay. So here we see that the learning rate decay subtracted a ton of energy out of the system and allowed us to settle into sort of the local minimum in this optimization.

So this is a much nicer plot. Let me come up and delete the monster and we're going to be using this going forward. Now, next up, what I'm bothered by is that you see our forward pass is a little bit gnarly and takes way too many lines of code.

So in particular, we see that we've organized some of the layers inside the layers list but not all of them for no reason. So in particular, we see that we still have the embedding table special case outside of the layers. And in addition to that, the viewing operation here is also outside of our layers.

So let's create layers for these and then we can add those layers to just our list. So in particular, the two things that we need is here, we have this embedding table and we are indexing at the integers inside the batch XB, inside the tensor XB. So that's an embedding table lookup just done with indexing.

And then here we see that we have this view operation which if you recall from the previous video, simply rearranges the character embeddings and stretches them out into row. And effectively what that does is the concatenation operation basically, except it's free because viewing is very cheap in PyTorch. And no memory is being copied.

We're just re-representing how we view that tensor. So let's create modules for both of these operations, the embedding operation and the flattening operation. So I actually wrote the code just to save some time.

So we have a module embedding and a module flatten, and both of them simply do the indexing operation in a forward pass and the flattening operation here. And this C now will just become a self.weight inside an embedding module. And I'm calling these layers specifically embedding and flatten because it turns out that both of them actually exist in PyTorch.

So in PyTorch, we have n and dot embedding, and it also takes the number of embeddings and the dimensionality of the embedding, just like we have here. But in addition, PyTorch takes in a lot of other keyword arguments that we are not using for our purposes yet. And for flatten, that also exists in PyTorch.

And it also takes additional keyword arguments that we are not using. So we have a very simple flatten. But both of them exist in PyTorch, they're just a bit more simpler.

And now that we have these, we can simply take out some of these special cased things. So instead of C, we're just going to have an embedding and a vocab size and n embed. And then after the embedding, we are going to flatten.

So let's construct those modules. And now I can take out this C. And here, I don't have to special case it anymore because now C is the embedding's weight. And it's inside layers.

So this should just work. And then here, our forward pass simplifies substantially because we don't need to do these now outside of these layer, outside and explicitly. They're now inside layers.

So we can delete those. But now to kick things off, we want this little X, which in the beginning is just XB, the tensor of integers specifying the identities of these characters at the input. And so these characters can now directly feed into the first layer, and this should just work.

So let me come here and insert a break because I just want to make sure that the first iteration of this runs and that there's no mistake. So that ran properly. And basically we've substantially simplified the forward pass here.

Okay, I'm sorry, I changed my microphone. So hopefully the audio is a little bit better. Now, one more thing that I would like to do in order to PyTorchify our code in further is that right now we are maintaining all of our modules in a naked list of layers.

And we can also simplify this because we can introduce the concept of PyTorch containers. So in torch.nn, which we are basically rebuilding from scratch here, there's a concept of containers. And these containers are basically a way of organizing layers into lists or dicts and so on.

So in particular, there's a sequential, which maintains a list of layers and is a module class in PyTorch. And it basically just passes a given input through all the layers sequentially, exactly as we are doing here. So let's write our own sequential.

I've written a code here. And basically the code for sequential is quite straightforward. We pass in a list of layers, which we keep here, and then given any input in a forward pass, we just call all the layers sequentially and return the result.

And in terms of the parameters, it's just all the parameters of the child modules. So we can run this, and we can again simplify this substantially because we don't maintain this naked list of layers. We now have a notion of a model, which is a module, and in particular is a sequential of all the layers.

And now parameters are simply just model.parameters. And so that list comprehension now lives here. And then here we are doing all the things we used to do. Now here, the code again simplifies substantially because we don't have to do this forwarding here.

Instead, we just call the model on the input data. And the input data here are the integers inside XB. So we can simply do logits, which are the outputs of our model, are simply the model called on XB.

And then the cross entropy here takes the logits and the targets. So this simplifies substantially, and then this looks good. So let's just make sure this runs.

That looks good. Now here, we actually have some work to do still here, but I'm gonna come back later. For now, there's no more layers.

There's a model.layers, but it's not easy to access attributes of these classes directly. So we'll come back and fix this later. And then here, of course, this simplifies substantially as well because logits are the model called on X. And then these logits come here.

So we can evaluate the training validation loss, which currently is terrible because we just initialized the neural net. And then we can also sample from the model, and this simplifies dramatically as well because we just want to call the model onto the context and outcome logits. And then these logits go into Softmax and get the probabilities, et cetera.

So we can sample from this model. What did I screw up? Okay, so I fixed the issue, and we now get the result that we expect, which is gibberish because the model is not trained because we reinitialized it from scratch. The problem was that when I fixed this cell to be model.layers instead of just layers, I did not actually run the cell.

And so our neural net was in a training mode. And what caused the issue here is the BatchNorm layer, as BatchNorm layer often likes to do, because BatchNorm was in the training mode. And here we are passing in an input, which is a batch of just a single example made up of the context.

And so if you are trying to pass in a single example into a BatchNorm that is in the training mode, you're gonna end up estimating the variance using the input. And the variance of a single number is not a number because it is a measure of a spread. So for example, the variance of just a single number five, you can see is not a number.

And so that's what happened. And BatchNorm basically caused an issue, and then that polluted all of the further processing. So all that we had to do was make sure that this runs.

And we basically made the issue of, again, we didn't actually see the issue with the loss. We could have evaluated the loss, but we got the wrong result because BatchNorm was in the training mode. And so we still get a result, it's just the wrong result because it's using the sample statistics of the batch, whereas we want to use the running mean and running variance inside the BatchNorm.

And so, again, an example of introducing a bug inline because we did not properly maintain the state of what is training or not. Okay, so I re-run everything, and here's where we are. As a reminder, we have the training loss of 2.05 and validation of 2.10. Now, because these losses are very similar to each other, we have a sense that we are not overfitting too much on this task, and we can make additional progress in our performance by scaling up the size of the neural network and making everything bigger and deeper.

Now, currently we are using this architecture here, where we are taking in some number of characters, going into a single hidden layer, and then going to the prediction of the next character. The problem here is we don't have a naive way of making this bigger in a productive way. We could, of course, use our layers, sort of building blocks and materials, to introduce additional layers here and make the network deeper, but it is still the case that we are crushing all of the characters into a single layer all the way at the beginning.

And even if we make this a bigger layer and add neurons, it's still kind of like silly to squash all that information so fast in a single step. So what we'd like to do instead is we'd like our network to look a lot more like this in the WaveNet case. So you see in the WaveNet, when we are trying to make the prediction for the next character in the sequence, it is a function of the previous characters that feed in, but not, all of these different characters are not just crushed to a single layer, and then you have a sandwich.

They are crushed slowly. So in particular, we take two characters and we fuse them into sort of like a bigram representation. And we do that for all these characters consecutively.

And then we take the bigrams and we fuse those into four character level chunks. And then we fuse that again. And so we do that in this like tree-like hierarchical manner.

So we fuse the information from the previous context slowly into the network as it gets deeper. And so this is the kind of architecture that we want to implement. Now, in the WaveNet's case, this is a visualization of a stack of dilated causal convolution layers.

And this makes it sound very scary, but actually the idea is very simple. And the fact that it's a dilated causal convolution layer is really just an implementation detail to make everything fast. We're gonna see that later.

But for now, let's just keep the basic idea of it, which is this progressive fusion. So we want to make the network deeper. And at each level, we want to fuse only two consecutive elements, two characters, then two bigrams, then two fourgrams, and so on.

So let's implement this. Okay, so first up, let me scroll to where we built the dataset and let's change the block size from three to eight. So we're going to be taking eight characters of context to predict the ninth character.

So the dataset now looks like this. We have a lot more context feeding in to predict any next character in a sequence. And these eight characters are going to be processed in this tree-like structure.

Now, if we scroll here, everything here should just be able to work. So we should be able to redefine the network. You see that number of parameters has increased by 10,000, and that's because the block size has grown.

So this first linear layer is much, much bigger. Our linear layer now takes eight characters into this middle layer. So there's a lot more parameters there.

But this should just run. Let me just break right after the very first iteration. So you see that this runs just fine.

It's just that this network doesn't make too much sense. We're crushing way too much information way too fast. So let's now come in and see how we could try to implement the hierarchical scheme.

Now, before we dive into the detail of the reimplementation here, I was just curious to actually run it and see where we are in terms of the baseline performance of just lazily scaling up the context length. So I'll let it run. We get a nice loss curve.

And then evaluating the loss, we actually see quite a bit of improvement just from increasing the context length. So I started a little bit of a performance log here. And previously where we were is we were getting a performance of 2.10 on the validation loss.

And now simply scaling up the context length from three to eight gives us a performance of 2.02. So quite a bit of an improvement here. And also when you sample from the model, you see that the names are definitely improving qualitatively as well. So we could of course spend a lot of time here tuning things and making it even bigger and scaling up the network further, even with a simple sort of setup here.

But let's continue and let's implement hierarchical model and treat this as just a rough baseline performance. But there's a lot of optimization left on the table in terms of some of the hyperparameters that you're hopefully getting a sense of now. Okay, so let's scroll up now and come back up.

And what I've done here is I've created a bit of a scratch space for us to just look at the four paths of the neural net and inspect the shape of the tensors along the way as the neural net forwards. So here I'm just temporarily for debugging, creating a batch of just say four examples. So four random integers, then I'm plucking out those rows from our training set.

And then I'm passing into the model, the input XB. Now the shape of XB here, because we have only four examples is four by eight. And this eight is now the current block size.

So inspecting XB, we just see that we have four examples. Each one of them is a row of XB. And we have eight characters here.

And this integer tensor just contains the identities of those characters. So the first layer of our neural net is the embedding layer. So passing XB, this integer tensor through the embedding layer creates an output that is four by eight by 10.

So our embedding table has for each character a 10 dimensional vector that we are trying to learn. And so what the embedding layer does here is it plucks out the embedding vector for each one of these integers and organizes it all in a four by eight by 10 tensor now. So all of these integers are translated into 10 dimensional vectors.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

...inside this three-dimensional tensor now. Now passing that through the flattened layer, as you recall, what this does is it views this tensor as just a 4x80 tensor, and what that effectively does is that all these 10-dimensional embeddings for all these eight characters just end up being stretched out into a long row, and that looks kind of like a concatenation operation basically. So by viewing the tensor differently, we now have a 4x80, and inside this 80 it's all the 10-dimensional vectors just concatenate next to each other. 

And the linear layer, of course, takes 80 and creates 200 channels just via matrix multiplication. So, so far, so good. Now I'd like to show you something surprising. 

Let's look at the insides of the linear layer and remind ourselves how it works. The linear layer here in the forward pass takes the input x, multiplies it with a weight, and then optionally adds a bias. And the weight here is two-dimensional, as defined here, and the bias is one-dimensional here. 

So effectively, in terms of the shapes involved, what's happening inside this linear layer looks like this right now. And I'm using random numbers here, but I'm just illustrating the shapes and what happens. Basically, a 4x80 input comes into the linear layer, gets multiplied by this 80x200 weight matrix inside, and there's a plus 200 bias. 

And the shape of the whole thing that comes out of the linear layer is 4x200, as we see here. Now, notice here, by the way, that this here will create a 4x200 tensor, and then plus 200, there's a broadcasting happening here, but 4x200 broadcasts with 200, so everything works here. So now the surprising thing that I'd like to show you that you may not expect is that this input here that is being multiplied doesn't actually have to be two-dimensional.

This matrix multiply operator in PyTorch is quite powerful, and in fact, you can actually pass in higher dimensional arrays or tensors, and everything works fine. So for example, this could be 4x5x80, and the result in that case will become 4x5x200. You can add as many dimensions as you like on the left here. 

And so effectively, what's happening is that the matrix multiplication only works on the last dimension, and the dimensions before it in the input tensor are left unchanged. So basically, these dimensions on the left are all treated as just a batch dimension. So we can have multiple batch dimensions, and then in parallel over all those dimensions, we are doing the matrix multiplication on the last dimension. 

So this is quite convenient because we can use that in our network now. Because remember that we have these eight characters coming in, and we don't want to now flatten all of it out into a large eight-dimensional vector, because we don't want to matrix multiply 80 into a weight matrix multiply immediately. Instead, we want to group these like this. 

So every consecutive two elements, 1, 2, 3, 4, 5, 6, 7, 8, all of these should be now basically flattened out and multiplied by a weight matrix. But all of these four groups here, we'd like to process in parallel. So it's kind of like a batch dimension that we can introduce. 

And then we can in parallel basically process all of these bigram groups in the four batch dimensions of an individual example, and also over the actual batch dimension of the four examples in our example here. So let's see how that works. Effectively, what we want is right now we take a 4 by 80 and multiply it by 80 by 200 in the linear layer. 

This is what happens. But instead, what we want is we don't want 80 characters or 80 numbers to come in. We only want two characters to come in on the very first layer, and those two characters should be fused.

So in other words, we just want 20 to come in, right? 20 numbers would come in. And here, we don't want a 4 by 80 to feed into the linear layer. We actually want these groups of two to feed in. 

So instead of 4 by 80, we want this to be a 4 by 4 by 20. So these are the four groups of two, and each one of them is 10-dimensional vector. So what we want is now is we need to change the flattened layer so it doesn't output a 4 by 80, but it outputs a 4 by 4 by 20, where basically every two consecutive characters are packed in on the very last dimension.

And then these four is the first batch dimension, and this four is the second batch dimension, referring to the four groups inside every one of these examples. And then this will just multiply like this. So this is what we want to get to. 

So we're going to have to change the linear layer in terms of how many inputs it expects. It shouldn't expect 80, it should just expect 20 numbers. And we have to change our flattened layer so it doesn't just fully flatten out this entire example. 

It needs to create a 4 by 4 by 20 instead of 4 by 80. So let's see how this could be implemented. Basically, right now, we have an input that is a 4 by 8 by 10 that feeds into the flattened layer. 

And currently, the flattened layer just stretches it out. So if you remember the implementation of flatten, it takes our x and it just views it as whatever the batch dimension is, and then negative 1. So effectively, what it does right now is it does edut view of 4, negative 1, and the shape of this, of course, is 4 by 80. So that's what currently happens, and we instead want this to be a 4 by 4 by 20, where these consecutive 10-dimensional vectors get concatenated. 

So you know how in Python you can take a list of range of 10, so we have numbers from 0 to 9, and we can index like this to get all the even parts. And we can also index like starting at 1 and going in steps of 2 to get all the odd parts. So one way to implement this, it would be as follows. 

We can take e, and we can index into it for all the batch elements, and then just even elements in this dimension. So at indexes 0, 2, 4, and 8, and then all the parts here from this last dimension. And this gives us the even characters, and then here this gives us all the odd characters.

And basically what we want to do is we want to make sure that these get concatenated in PyTorch, and then we want to concatenate these two tensors along the second dimension. So this and the shape of it would be 4 by 4 by 20. This is definitely the result we want.

We are explicitly grabbing the even parts and the odd parts, and we're arranging those 4 by 4 by 10 right next to each other and concatenate. So this works, but it turns out that what also works is you can simply use view again and just request the right shape. And it just so happens that in this case, those vectors will again end up being arranged exactly the way we want. 

So in particular, if we take e and we just view it as a 4 by 4 by 20, which is what we want, we can check that this is exactly equal to what let me call this. This is the explicit concatenation, I suppose. So explicit dot shape is 4 by 4 by 20. 

If you just view it as 4 by 4 by 20, you can check that when you compare it to explicit, this is element-wise operation, so making sure that all of them are true, the values are true. So basically, long story short, we don't need to make an explicit call to concatenate, etc. We can simply take this input tensor to flatten, and we can just view it in whatever way we want. 

And in particular, we don't want to stretch things out with negative one. We want to actually create a three-dimensional array, and depending on how many vectors that are consecutive, we want to fuse, like for example, two, then we can just simply ask for this dimension to be 20, and use a negative one here, and PyTorch will figure out how many groups it needs to pack into this additional batch dimension. So let's now go into flatten and implement this. 

Okay, so I scrolled up here to flatten, and what we'd like to do is we'd like to change it now. So let me create a constructor and take the number of elements that are consecutive that we would like to concatenate now in the last dimension of the output. So here, we're just going to remember self.n equals n, and then I want to be careful here, because PyTorch actually has a Torch.flatten, and its keyword arguments are different, and they kind of like function differently, so our flatten is going to start to depart from PyTorch.flatten. So let me call it flatten consecutive, or something like that, just to make sure that our APIs are about equal. 

So this basically flattens only some n consecutive elements and puts them into the last dimension. Now here, the shape of x is b by t by c, so let me pop those out into variables, and recall that in our example down below, b was 4, t was 8, and c was 10. Now, instead of doing x.view of b by negative 1, right, this is what we had before. 

We want this to be b by negative 1 by, and basically here, we want c times n. That's how many consecutive elements we want. And here, instead of negative 1, I don't super love the use of negative 1, because I like to be very explicit so that you get error messages when things don't go according to your expectation. So what do we expect here? We expect this to become t divide n, using integer division here. 

So that's what I expect to happen. And then one more thing I want to do here is, remember previously, all the way in the beginning, n was 3, and basically we're concatenating all the three characters that existed there. So we basically concatenated everything. 

And so sometimes that can create a spurious dimension of 1 here. So if it is the case that x.shape at 1 is 1, then it's kind of like a spurious dimension. So we don't want to return a three-dimensional tensor with a 1 here. 

We just want to return a two-dimensional tensor exactly as we did before. So in this case, basically, we will just say x equals x.squeeze, that is a PyTorch function. And squeeze takes a dimension that it either squeezes out all the dimensions of a tensor that are 1, or you can specify the exact dimension that you want to be squeezed. 

And again, I like to be as explicit as possible always, so I expect to squeeze out the first dimension only of this tensor, this three-dimensional tensor. And if this dimension here is 1, then I just want to return b by c times n. And so self.out will be x, and then we return self.out. So that's the candidate implementation. And of course, this should be self.in instead of just n. So let's run. 

And let's come here now and take it for a spin. So flattened consecutive. And in the beginning, let's just use 8. So this should recover the previous behavior. 

So flattened consecutive of 8, which is the current block size, we can do this. That should recover the previous behavior. So we should be able to run the model.

And here we can inspect. I have a little code snippet here where I iterate over all the layers. I print the name of this class and the shape. 

And so we see the shapes as we expect them after every single layer in its output. So now let's try to restructure it using our flattened consecutive and do it hierarchically. So in particular, we want to flatten consecutive not block size, but just 2. And then we want to process this with linear. 

Now the number of inputs to this linear will not be n embed times block size. It will now only be n embed times 2, 20. This goes through the first layer. 

And now we can, in principle, just copy paste this. Now the next linear layer should expect n hidden times 2. And the last piece of it should expect n hidden times 2 again. So this is sort of like the naive version of it. 

So running this, we now have a much, much bigger model. And we should be able to basically just forward the model. And now we can inspect the numbers in between. 

So 4 byte by 20 was flattened consecutively into 4 by 4 by 20. This was projected into 4 by 4 by 200. And then Bashorm just worked out of the box. 

We have to verify that Bashorm does the correct thing, even though it takes a three-dimensional embed instead of two-dimensional embed. Then we have 10h, which is element-wise. Then we crushed it again. 

So we flattened consecutively and ended up with a 4 by 2 by 400 now. Then linear brought back down to 200, Bashorm 10h. And lastly, we get a 4 by 400. 

And we see that the flattened consecutive for the last flattened here, it squeezed out that dimension of 1. So we only ended up with 4 by 400. And then linear Bashorm 10h and the last linear layer to get our logits. And so the logits end up in the same shape as they were before. 

But now we actually have a nice three-layer neural net. And it basically corresponds to exactly to this network now, except only this piece here, because we only have three layers. Whereas here in this example, there's four layers with a total receptive field size of 16 characters instead of just eight characters. 

So the block size here is 16. So this piece of it is basically implemented here. And now we just have to figure out some good channel numbers to use here. 

Now, in particular, I changed the number of hidden units to be 68 in this architecture, because when I use 68, the number of parameters comes out to be 22,000. So that's exactly the same that we had before. And we have the same amount of capacity at this neural net in terms of the number of parameters.

But the question is whether we are utilizing those parameters in a more efficient architecture. So what I did then is I got rid of a lot of the debugging cells here. And I re-ran the optimization.

And scrolling down to the result, we see that we get the identical performance roughly. So our validation loss now is 2.029. And previously, it was 2.027. So controlling for the number of parameters, changing from the flat to hierarchical is not giving us anything yet. That said, there are two things to point out. 

Number one, we didn't really torture the architecture here very much. This is just my first guess. And there's a bunch of hyperparameter search that we could do in terms of how we allocate our budget of parameters to what layers.

Number two, we still may have a bug inside the BatchNorm1D layer. So let's take a look at that, because it runs, but does it do the right thing? So I pulled up the layer inspector that we have here and printed out the shapes along the way. And currently, it looks like the BatchNorm is receiving an input that is 32 by 4 by 68.

And here on the right, I have the current implementation of BatchNorm that we have right now. Now, this BatchNorm assumed, in the way we wrote it and at the time, that x is two-dimensional. So it was n by d, where n was the batch size. 

So that's why we only reduced the mean and the variance over the zeroth dimension. But now, x will basically become three-dimensional. So what's happening inside the BatchNorm layer right now and how it's working at all and not giving any errors? The reason for that is basically because everything broadcasts properly, but the BatchNorm is not doing what we want it to do. 

So in particular, let's basically think through what's happening inside the BatchNorm, looking at what's happening here. I have the code here. So we're receiving an input of 32 by 4 by 68. 

And then we are doing here, x.mean. Here, I have e instead of x. But we're doing the mean over zero. And that's actually giving us 1 by 4 by 68. So we're doing the mean only over the very first dimension. 

And that's giving us a mean and a variance that still maintain this dimension here. So these means are only taken over 32 numbers in the first dimension. And then when we perform this, everything broadcasts correctly still. 

But basically, what ends up happening is when we also look at the running mean, the shape of it. So I'm looking at the model that layers at 3, which is the first BatchNorm layer, and then looking at whatever the running mean became and its shape. The shape of this running mean now is 1 by 4 by 68. 

Instead of it being just size of dimension, because we have 68 channels, we expect to have 68 means and variances that we're maintaining. But actually, we have an array of 4 by 68. And so basically, what this is telling us is this BatchNorm is currently working in parallel over 4 times 68 instead of just 68 channels. 

So basically, we are maintaining statistics for every one of these 4 positions individually and independently. And instead, what we want to do is we want to treat this 4 as a BatchDimension, just like the 0th dimension. So as far as the BatchNorm is concerned, we don't want to average over 32 numbers.

We want to now average over 32 times 4 numbers for every single one of these 68 channels. And so let me now remove this. It turns out that when you look at the documentation of torch.mean, in one of its signatures, when we specify the dimension, we see that the dimension here can be int or it can also be a tuple of ints.

So we can reduce over multiple dimensions at the same time. So instead of just reducing over 0, we can pass in a tuple, 0, 1. And here, 0, 1 as well. And then what's going to happen is the output, of course, is going to be the same. 

But now what's going to happen is because we reduce over 0 and 1, if we look at inmean.shape, we see that now we've reduced. We took the mean over both the 0th and the 1st dimension. So we're just getting 68 numbers in a bunch of spurious dimensions here.

So now this becomes 1 by 1 by 68. And the running mean and the running variance analogously will become 1 by 1 by 68. So even though there are the spurious dimensions, the correct thing will happen in that we are only maintaining means and variances for 68 channels. 

And we're not calculating the mean and variance across 32 times 4 dimensions. So that's exactly what we want. And let's change the implementation of BatchNorm1D that we have so that it can take in 2-dimensional or 3-dimensional inputs and perform accordingly. 

So at the end of the day, the fix is relatively straightforward. Basically, the dimension we want to reduce over is either 0 or the tuple 0 and 1, depending on the dimensionality of x. So if x.ndim is 2, so it's a 2-dimensional tensor, then the dimension we want to reduce over is just the integer 0. If x.ndim is 3, so it's a 3-dimensional tensor, then the dims we're going to assume are 0 and 1 that we want to reduce over. And then here, we just pass in dim. 

And if the dimensionality of x is anything else, we'll now get an error, which is good. So that should be the fix. Now, I want to point out one more thing.

We're actually departing from the API of PyTorch here a little bit, because when you come to BatchNorm1D in PyTorch, you can scroll down and you can see that the input to this layer can either be n by c, where n is the batch size and c is the number of features or channels, or it actually does accept 3-dimensional inputs, but it expects it to be n by c by l, where l is, say, the sequence length or something like that. So this is a problem, because you see how c is nested here in the middle. And so when it gets 3-dimensional inputs, this BatchNorm layer will reduce over 0 and 2 instead of 0 and 1. So basically, PyTorch BatchNorm1D layer assumes that c will always be the first dimension, whereas we assume here that c is the last dimension and there are some number of batch dimensions beforehand. 

And so it expects n by c or n by c by l. We expect n by c or n by l by c. And so it's a deviation. I think it's okay. I prefer it this way, honestly, so this is the way that we will keep it for our purposes. 

So I redefined the layers, reinitialized the neural net, and did a single forward pass with a break just for one step. Looking at the shapes along the way, they're of course identical, all the shapes are the same, but the way we see that things are actually working as we want them to now is that when we look at the BatchNorm layer, the running mean shape is now 1 by 1 by 68. So we're only maintaining 68 means for every one of our channels, and we're treating both the zeroth and the first dimension as a batch dimension, which is exactly what we want. 

So let me retrain the neural net now. Okay, so I've retrained the neural net with the bug fix, we get a nice curve, and when we look at the validation performance, we do actually see a slight improvement. So it went from 2.029 to 2.022. So basically, the bug inside the BatchNorm was holding us back a little bit, it looks like, and we are getting a tiny improvement now, but it's not clear if this is statistically significant.

And the reason we slightly expect an improvement is because we're not maintaining so many different means and variances that are only estimated using 32 numbers effectively. Now we are estimating them using 32 times 4 numbers, so you just have a lot more numbers that go into any one estimate of the mean and variance, and it allows things to be a bit more stable and less wiggly inside those estimates of those statistics. So pretty nice. 

With this more general architecture in place, we are now set up to push the performance further by increasing the size of the network. So for example, I've bumped up the number of embeddings to 24 instead of 10, and also increased the number of hidden units. But using the exact same architecture, we now have 76,000 parameters, and the training takes a lot longer, but we do get a nice curve. 

And then when you evaluate the performance, we are now getting validation performance of 1.993. So we've crossed over the 2.0 territory, and we're at about 1.99. But we are starting to have to wait quite a bit longer, and we're a little bit in the dark with respect to the correct setting of the hyper parameters here and the learning rates and so on, because the experiments are starting to take longer to train. And so we are missing an experimental harness on which we could run a number of experiments and really tune this architecture very well. So I'd like to conclude now with a few notes. 

We basically improved our performance from a starting of 2.1 down to 1.9. But I don't want that to be the focus, because honestly, we're kind of in the dark. We have no experimental harness. We're just guessing and checking. 

And this whole thing is terrible. We're just looking at the training loss. Normally, you want to look at both the training and the validation loss together.

The whole thing looks different if you're actually trying to squeeze out numbers. That said, we did implement this architecture from the WaveNet paper, but we did not implement this specific forward pass of it, where you have a more complicated linear layer, sort of, that is this gated linear layer, kind of. And there's residual connections and skip connections and so on. 

So we did not implement that. We just implemented this structure. I would like to briefly hint or preview how what we've done here relates to convolutional neural networks as used in the WaveNet paper. 

And basically, the use of convolutions is strictly for efficiency. It doesn't actually change the model we've implemented. So here, for example, let me look at a specific name to work with an example. 

So there's a name in our training set, and it's D'Andrea. And it has seven letters, so that is eight independent examples in our model. So all these rows here are independent examples of D'Andrea. 

Now, you can forward, of course, any one of these rows independently. So I can take my model and call it on any individual index. Notice, by the way, here, I'm being a little bit tricky. 

The reason for this is that extra at seven dot shape is just one dimensional array of eight. So you can't actually call the model on it, you're going to get an error, because there's no batch dimension. So when you do extra at list of seven, then the shape of this becomes one by eight. 

So I get an extra batch dimension of one, and then we can forward the model. So that forwards a single example. And you might imagine that you actually may want to forward all of these eight at the same time. 

So pre-allocating some memory and then doing a for loop eight times and forwarding all of those eight here will give us all the logits in all these different cases. Now, for us with the model, as we've implemented it right now, this is eight independent calls to our model. But what convolutions allow you to do is it allow you to basically slide this model efficiently over the input sequence. 

And so this for loop can be done not outside in Python, but inside of kernels in CUDA. And so this for loop gets hidden into the convolution. So the convolution basically, you can think of it as it's a for loop applying a little linear filter over space of some input sequence. 

And in our case, the space we're interested in is one dimensional, and we're interested in sliding these filters over the input data. So this diagram actually is fairly good as well. Basically, what we've done is here they are highlighting in black one single sort of like tree of this calculation. 

So just calculating the single output example here. And so this is basically what we've implemented here. We've implemented a single, this black structure, we've implemented that and calculated a single output, like a single example. 

But what convolutions allow you to do is it allows you to take this black structure and kind of like slide it over the input sequence here and calculate all of these orange outputs at the same time. Or here that corresponds to calculating all of these outputs of at all the positions of DeAndre at the same time. And the reason that this is much more efficient is because number one, as I mentioned, the for loop is inside the CUDA kernels in the sliding. 

So that makes it efficient. But number two, notice the variable reuse here. For example, if we look at this circle, this node here, this node here is the right child of this node, but it's also the left child of the node here. 

And so basically this node and its value is used twice. And so right now, in this naive way, we'd have to recalculate it. But here we are allowed to reuse it. 

So in the convolutional neural network, you think of these linear layers that we have up above as filters. And we take these filters, and they're linear filters, and you slide them over input sequence. And we calculate the first layer, and then the second layer, and then the third layer, and then the output layer of the sandwich. 

And it's all done very efficiently using these convolutions. So we're going to cover that in a future video. The second thing I hope you took away from this video is you've seen me basically implement all of these layer Lego building blocks or module building blocks. 

And I'm implementing them over here. And we've implemented a number of layers together. And we're also implementing these containers. 

And we've overall PyTorchified our code quite a bit more. Now, basically what we're doing here is we're re-implementing Torch.nn, which is the neural network's library on top of Torch.tensor. And it looks very much like this, except it is much better because it's in PyTorch instead of a janky-like Jupyter notebook. So I think going forward, I will probably have considered us having unlocked Torch.nn. We understand roughly what's in there, how these modules work, how they're nested, and what they're doing on top of Torch.tensor. So hopefully we'll just switch over and continue and start using Torch.nn directly.

The next thing I hope you got a bit of a sense of is what the development process of building deep neural networks looks like, which I think was relatively representative to some extent. So number one, we are spending a lot of time in the documentation page of PyTorch. And we're reading through all the layers, looking at documentations, what are the shapes of the inputs, what can they be, what does the layer do, and so on. 

Unfortunately, I have to say the PyTorch documentation is not very good. They spend a ton of time on hardcore engineering of all kinds of distributed primitives, etc.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

documentation, it will lie to you, it will be wrong, it will be incomplete, it will be unclear, so unfortunately it is what it is and you just kind of do your best with what they've given us. Number two, the other thing that I hope you got a sense of is there's a ton of trying to make the shapes work and there's a lot of gymnastics around these multi-dimensional arrays and are two-dimensional, three-dimensional, four-dimensional, what layers take what shapes, is it NCL or NLC and you're permuting and viewing and it just can get pretty messy and so that brings me to number three. I very often prototype these layers and implementations in Jupyter notebooks and make sure that all the shapes work out and I'm spending a lot of time basically babysitting the shapes and making sure everything is correct and then once I'm satisfied with the functionality in a Jupyter notebook, I will take that code and copy-paste it into my repository of actual code that I'm training with and so then I'm working with VS code on the side, so I usually have Jupyter notebook and VS code, I develop a Jupyter notebook, I paste into VS code and then I kick off experiments from from the repo of course, from the code repository. 

So that's roughly some notes on the development process of working with neural nets. Lastly, I think this lecture unlocks a lot of potential further lectures because number one, we have to convert our neural network to actually use these dilated causal convolutional layers, so implementing the ConNet. Number two, I'm potentially starting to get into what this means, where are residual connections and skip connections and why are they useful. 

Number three, as I mentioned, we don't have any experimental harness so right now I'm just guessing, checking everything. This is not representative of typical deep learning workflows. You have to set up your evaluation harness, you can kick off experiments, you have lots of arguments that your script can take, you're kicking off a lot of experimentation, you're looking at a lot of plots of training and validation losses and you're looking at what is working and what is not working and you're working on this like population level and you're doing all these hyperparameter searches and so we've done none of that so far, so how to set that up and how to make it good I think is a whole another topic. 

Number three, we should probably cover recurrent neural networks, RNNs, LSTMs, GRUs and of course transformers, so many places to go and we'll cover that in the future. For now, bye. Sorry, I forgot to say that if you are interested, I think it is kind of interesting to try to beat this number 1.993 because I really haven't tried a lot of experimentation here and there's quite a bit of longing fruit potentially to still push this further, so I haven't tried any other ways of allocating these channels in this neural net, maybe the number of dimensions for the embedding is all wrong, maybe it's possible to actually take the original network with just one hidden layer and make it big enough and actually beat my fancy hierarchical network, it's not obvious, that would be kind of embarrassing if this did not do better even once you torture it a little bit.

Maybe you can read the WaveNet paper and try to figure out how some of these layers work and implement them yourselves using what we have and of course you can always tune some of the initialization or some of the optimization and see if you can improve it that way. So I'd be curious if people can come up with some ways to beat this and yeah, that's it for now, bye. Hi everyone, so by now you have probably heard of ChatGPT, it has taken the world and the AI community by storm and it is a system that allows you to interact with an AI and give it text-based tasks. 

So for example, we can ask ChatGPT to write us a small haiku about how important it is that people understand AI and then they can use it to improve the world and make it more prosperous. So when we run this, AI knowledge brings prosperity for all to see, embrace its power. Okay, not bad. 

And so you could see that ChatGPT went from left to right and generated all these words sequentially. Now I asked it already the exact same prompt a little bit earlier and it generated a slightly different outcome. AI's power to grow, ignorance holds us back, learn, prosperity waits. 

So pretty good in both cases and slightly different. So you can see that ChatGPT is a probabilistic system and for any one prompt it can give us multiple answers sort of replying to it. Now this is just one example of a prompt, people have come up with many many examples and there are entire websites that index interactions with ChatGPT and so many of them are quite humorous, explain HTML to me like I'm a dog, write release notes for chess 2, write a note about Elon Musk buying a Twitter and so on. 

So as an example, please write a breaking news article about a leaf falling from a tree. In a shocking turn of events, a leaf has fallen from a tree in the local park. Witnesses report that the leaf, which was previously attached to a branch of a tree, detached itself and fell to the ground. 

Very dramatic. So you can see that this is a pretty remarkable system and it is what we call a language model because it models the sequence of words or characters or tokens more generally and it knows how sort of words follow each other in English language and so from its perspective what it is doing is it is completing the sequence. So I give it the start of a sequence and it completes the sequence with the outcome and so it's a language model in that sense. 

Now I would like to focus on the under the hood components of what makes ChessGPT work. So what is the neural network under the hood that models the sequence of these words? And that comes from this paper called Attention is All You Need. In 2017, a landmark paper in AI that produced and proposed the transformer architecture. 

So GPT is short for generatively pre-trained transformer. So transformer is the neural net that actually does all the heavy lifting under the hood. It comes from this paper in 2017. 

Now if you read this paper, this reads like a pretty random machine translation paper and that's because I think the authors didn't fully anticipate the impact that the transformer would have on the field and this architecture that they produced in the context of machine translation, in their case, actually ended up taking over the rest of AI in the next five years after. And so this architecture with minor changes was copy-pasted into a huge amount of applications in AI in more recent years and that includes at the core of ChessGPT. Now we are not going to, what I'd like to do now is I'd like to build out something like ChessGPT, but we're not going to be able to of course reproduce ChessGPT. 

This is a very serious production grade system. It is trained on a good chunk of internet and then there's a lot of pre-training and fine tuning stages to it and so it's very complicated. What I'd like to focus on is just to train a transformer-based language model and in our case it's going to be a character level language model. 

I still think that is a very educational with respect to how these systems work. So I don't want to train on the chunk of internet, we need a smaller data set. In this case I propose that we work with my favorite toy data set. 

It's called Tiny Shakespeare and what it is is basically it's a concatenation of all of the works of Shakespeare in my understanding. And so this is all of Shakespeare in a single file. This file is about one megabyte and it's just all of Shakespeare. 

And what we are going to do now is we're going to basically model how these characters follow each other. So for example given a chunk of these characters like this, given some context of characters in the past, the transformer neural network will look at the characters that I've highlighted and is going to predict that G is likely to come next in the sequence. And it's going to do that because we're going to train that transformer on Shakespeare and it's just going to try to produce character sequences that look like this. 

And in that process it's going to model all the patterns inside this data. So once we've trained the system, I'd just like to give you a and of course it's a fake thing that looks kind of like Shakespeare. Apologies for there's some jank that I'm not able to resolve in here but you can see how this is going character by character and it's kind of like predicting Shakespeare-like language. 

So verily my lord the sites have left thee again the king coming with my with precious pale and then troniosa something else etc. And this is just coming out of the transformer in a very similar manner as it would come out in chatgpt. In our case character by character in chatgpt it's coming out on the token by token level and tokens are these sort of like little sub word pieces. 

So they're not word level they're kind of like word chunk level. And now I've already written this entire code to train these transformers and it is in a github repository that you can find and it's called nano-gpt. So nano-gpt is a repository that you can find on my github and it's a repository for training transformers on any given text.

And what I think is interesting about it because there's many ways to train transformers but this is a very simple implementation. So it's just two files of 300 lines of code each one file defines the gpt model the transformer and one file trains it on some given text data set. And here I'm showing that if you train it on a open web text data set which is a fairly large data set of web pages then I reproduce the the performance of gpt2. 

So gpt2 is an early version of openai's gpt from 2017 if I recall correctly and I've only so far reproduced the smallest 124 million parameter model but basically this is just proving that the code base is correctly arranged and I'm able to load the neural network weights that openai has released later. So you can take a look at the finished code here in nano-gpt but what I would like to do in this lecture is I would like to basically write this repository from scratch. So we're going to begin with an empty file and we're going to define a transformer piece by piece. 

We're going to train it on the tiny shakespeare data set and we'll see how we can then generate infinite shakespeare. And of course this can copy paste to any arbitrary text data set that you like but my goal really here is to just make you understand and appreciate how under the hood chat gpt works and really all that's required is a proficiency in python and some basic understanding of calculus and statistics and it would help if you also see my previous videos on the same youtube channel in particular my make more series where I define smaller and simpler neural network language models so multilayered perceptrons and so on. It really introduces the language modeling framework and then here in this video we're going to focus on the transformer neural network itself.

Okay so I created a new google colab jupyter notebook here and this will allow me to later easily share this code that we're going to develop together with you so you can follow along. So this will be in the video description later. Now here I've just done some preliminaries. 

I downloaded the data set the tiny shakespeare data set at this url and you can see that it's about a one megabyte file. Then here I open the input.txt file and just read in all the text as a string and we see that we are working with 1 million characters roughly and the first 1000 characters if we just print them out are basically what you would expect. This is the first 1000 characters of the tiny shakespeare data set roughly up to here so so far so good. 

Next we're going to take this text and the text is a sequence of characters in python so when I call the set constructor on it I'm just going to get the set of all the characters that occur in this text and then I call list on that to create a list of those characters instead of just a set so that I have an ordering an arbitrary ordering and then I sort that. So basically we get just all the characters that occur in the entire data set and they're sorted. Now the number of them is going to be our vocabulary size. 

These are the possible elements of our sequences and we see that when I print here the characters there's 65 of them in total. There's a space character and then all kinds of special characters and then capitals and lowercase letters. So that's our vocabulary and that's the sort of like possible characters that the model can see or emit. 

Okay so next we would like to develop some strategy to tokenize the input text. Now when people say tokenize they mean convert the raw text as a string to some sequence of integers according to some notebook according to some vocabulary of possible elements. So as an example here we are going to be building a character level language model so we're simply going to be translating individual characters into integers. 

So let me show you a chunk of code that sort of does that for us. So we're building both the encoder and the decoder and let me just talk through what's happening here. When we encode an arbitrary text like hi there we're going to receive a list of integers that represents that string. 

So for example 46, 47, etc. And then we also have the reverse mapping so we can take this list and decode it to get back the exact same string. So it's really just like a translation to integers and back for arbitrary string and for us it is done on a character level.

Now the way this was achieved is we just iterate over all the characters here and create a lookup table from the character to the integer and vice versa and then to encode some string we simply translate all the characters individually and to decode it back we use the reverse mapping and concatenate all of it. Now this is only one of many possible encodings or many possible sort of tokenizers and it's a very simple one but there's many other schemas that people have come up with in practice. So for example Google uses a sentence piece so sentence piece will also encode text into integers but in a different schema and using a different vocabulary and sentence piece is a sub-word sort of tokenizer and what that means is that you're not encoding entire words but you're not also encoding individual characters. 

It's a sub-word unit level and that's usually what's adopted in practice. For example also OpenAI has this library called TicToken that uses a byte pair encoding tokenizer and that's what GPT uses and you can also just encode words into like hello world into lists of integers. So as an example I'm using the TicToken library here I'm getting the encoding from GPT2 or that was used for GPT2. 

Instead of just having 65 possible characters or tokens they have 50,000 tokens and so when they encode the exact same string hi there we only get a list of three integers but those integers are not between 0 and 64 they are between 0 and 50,256. So basically you can trade off the codebook size and the sequence lengths so you can have very long sequences of integers with very small vocabularies or you can have short sequences of integers with very large vocabularies and so typically people use in practice these sub-word encodings but I'd like to keep our tokenizer very simple so we're using character level tokenizer and that means that we have very small codebooks we have very simple encode and decode functions but we do get very long sequences as a result but that's the level at which we're going to stick with this lecture because it's the simplest thing. Okay so now that we have an encoder and a decoder effectively a tokenizer we can tokenize the entire training set of Shakespeare so here's a chunk of code that does that and I'm going to start to use the PyTorch library and specifically the Torch.Tensor from the PyTorch library so we're going to take all of the text in Tiny Shakespeare, encode it and then wrap it into a Torch.Tensor to get the DataTensor. 

So here's what the DataTensor looks like when I look at just the first 1,000 characters or the 1,000 elements of it. So we see that we have a massive sequence of integers and this sequence of integers here is basically an identical translation of the first 1,000 characters here. So I believe for example that 0 is a newline character and maybe 1 is a space. 

I'm not 100% sure but from now on the entire data set of text is re-represented as just it's just stretched out as a single very large sequence of integers. Let me do one more thing before we move on here. I'd like to separate out our data set into a train and a validation split. 

So in particular we're going to take the first 90% of the data set and consider that to be the training data for the transformer and we're going to withhold the last 10% at the end of it to be the validation data and this will help us understand to what extent our model is overfitting. So we're going to basically hide and keep the validation data on the side because we don't want just a perfect memorization of this exact Shakespeare. We want a neural network that sort of creates Shakespeare-like text and so it should be fairly likely for it to produce the actual like stowed away true Shakespeare text and so we're going to use this to get a sense of the overfitting. 

Okay so now we would like to start plugging these text sequences or integer sequences into the transformer so that it can train and learn those patterns. Now the important thing to realize is we're never going to actually feed the entire text into a transformer all at once. That would be computationally very expensive and prohibitive.

So when we actually train a transformer on a lot of these data sets we only work with chunks of the data set and when we train the transformer we basically sample random little chunks out of the training set and train them just chunks at a time and these chunks have basically some kind of a length and it's a maximum length. Now the maximum length typically at least in the code I usually write is called block size. You can you can find it under different names like context length or something like that. 

Let's start with the block size of just eight and let me look at the first trained data characters the first block size plus one characters. I'll explain why plus one in a second. So this is the first nine characters in the sequence in the training set. 

Now what I'd like to point out is that when you sample a chunk of data like this so say these nine characters out of the training set this actually has multiple examples packed into it and that's because all of these characters follow each other and so what this thing is going to say when we plug it into a transformer is we're going to actually simultaneously train it to make prediction at every one of these positions. Now in a chunk of nine characters there's actually eight individual examples packed in there so there's the example that when 18 when in the context of 18 47 likely comes next in a context of 18 and 47 56 comes next in the context of 18 47 56 57 can come next and so on so that's the eight individual examples. Let me actually spell it out with code so here's a chunk of code to illustrate x are the inputs to the transformer it will just be the first block size characters y will be the next block size characters so it's offset by one and that's because y are the targets for each position in the input and then here I'm iterating over all the block size of eight and the context is always all the characters in x up to t and including t and the target is always the t-th character but in the targets array y so let me just run this and basically it spells out what I said in words these are the eight examples hidden in a chunk of nine characters that we sampled from the training set. 

I want to mention one more thing we train on all the eight examples here with context between one all the way up to context of block size and we train on that not just for computational reasons because we happen to have the sequence already or something like that it's not just done for efficiency it's also done to make the transformer network be used to seeing contexts all the way from as little as one all the way to block size and we'd like the transformer to be used to seeing everything in between and that's going to be useful later during inference because while we're sampling we can start sampling generation with as little as one character of context and the transformer knows how to predict next character with all the way up to just context of one and so then it can predict everything up to block size and after block size we have to start truncating because the transformer will never receive more than block size inputs when it's predicting the next character. Okay so we've looked at the time dimension of the tensors that are going to be feeding into the transformer there's one more dimension to care about and that is the batch dimension and so as we're sampling these chunks of text we're going to be actually every time we're going to feed them into a transformer we're going to have many batches of multiple chunks of text that are all like stacked up in a single tensor and that's just done for efficiency just so that we can keep the GPUs busy because they are very good at parallel processing of data and so we just want to process multiple chunks all at the same time but those chunks are processed completely independently they don't talk to each other and so on. So let me basically just generalize this and introduce a batch dimension here's a chunk of code let me just run it and then I'm going to explain what it does.

So here because we're going to start sampling random locations in the data sets to pull chunks from I am setting the seed so that in the random number generator so that the numbers I see here are going to be the same numbers you see later if you try to reproduce this. Now the batch size here is how many independent sequences we are processing every forward backward pass of the transformer. The block size as I explained is the maximum context length to make those predictions so let's say batch size 4 block size 8 and then here's how we get batch for any arbitrary split if the split is a training split then we're going to look at train data otherwise at val data that gives us the data array and then when I generate random positions to grab a chunk out of I actually grab I actually generate batch size number of random offsets so because this is 4 we are ix is going to be a four numbers that are randomly generated between 0 and len of data minus block size so it's just random offsets into the training set and then x's as I explained are the first block size characters starting at i the y's are the offset by one of that so just add plus one and then we're going to get those chunks for every one of integers i in ix and use a torch dot stack to take all those one-dimensional tensors as we saw here and we're going to stack them up as rows and so they all become a row in a four by eight tensor so here's where I'm printing them when I sample a batch xb and yb the inputs to the transformer now are the input x is the four by eight tensor four rows of eight columns and each one of these is a chunk of the training set and then the targets here are in the associated array y and they will come in to the transformer all the way at the end to create the loss function so they will give us the correct answer for every single position inside x and then these are the four independent rows so spelled out as we did before this four by eight array contains a total of 32 examples and they're completely independent as far as the transformer is concerned so when the input is 24 the target is 43 or rather 43 here in the y array when the input is 24 43 target is 58 when the input is 24 43 58 the target is 5 etc or like when it is a 52 58 1 the target is 58 right so you can sort of see this spelled out these are the 32 independent examples packed in to a single batch of the input x and then the desired targets are in y and so now this integer tensor of x is going to feed into the transformer and that transformer is going to simultaneously process all these examples and then look up the correct integers to predict in every one of these positions in the tensor y okay so now that we have our batch of input that we'd like to feed into a transformer let's start basically feeding this into neural networks now we're going to start off with the simplest possible neural network which in the case of language modeling in my opinion is the bigram language model and we've covered the bigram language model in my make more series in a lot of depth and so here i'm going to sort of go faster and let's just implement the pytorch module directly that implements the bigram language model so i'm importing the pytorch nn module for reproducibility and then here i'm constructing a bigram language model which is a subclass of nn module and then i'm calling it and i'm passing in the inputs and the targets and i'm just printing now when the inputs and targets come here you see that i'm just taking the index the inputs x here which i rename to idx and i'm just passing them into this token embedding table so what's going on here is that here in the constructor we are creating a token embedding table and it is of size vocab size by vocab size and we're using nn.embedding which is a very thin wrapper around basically a tensor of shape vocab size by vocab size and what's happening here is that when we pass idx here every single integer in our input is going to refer to this embedding table and is going to pluck out a row of that embedding table corresponding to its index so 24 here will go to the embedding table and will pluck out the 24th row and then 43 will go here and pluck out the 43rd row etc and then pytorch is going to arrange all of this into a batch by time by channel tensor in this case batch is 4 time is 8 and c which is the channels is vocab size or 65 and so we're just going to pluck out all those rows arrange them in a b by t by c and now we're going to interpret this as the logits which are basically the scores for the next character in a sequence and so what's happening here is we are predicting what comes next based on just the individual identity of a single token and you can do that because um i mean currently the tokens are not talking to each other and they're not seeing any context except for they're just seeing themselves so i'm a i'm a token number five and then i can actually make pretty decent predictions about what comes next just by knowing that i'm token five because some characters know follow other characters in typical scenarios so we saw a lot of this in a lot more depth in the make more series and here if i just run this then we currently get the predictions the scores the logits for every one of the four by eight positions now that we've made predictions about what comes next we'd like to evaluate the loss function and so in make more series we saw that a good way to measure a loss or like a quality of the predictions is to use the negative log likelihood loss which is also implemented in pytorch under the name cross entropy so what we'd like to do here is loss is the cross entropy on the predictions and the targets and so this measures the quality of the logits with respect to the targets in other words we have the identity of the next character so how well are we predicting the next character based on the logits and intuitively the correct um the correct dimension of logits depending on whatever the target is should have a very high number and all the other dimensions should be very low number right now the issue is that this won't actually this is what we want we want to basically output the logits and the loss this is what we want but unfortunately this won't actually run we get an error message but intuitively we want to measure this now when we go to the pytorch cross entropy documentation here we're trying to call the cross entropy in its functional form so that means we don't have to create like a module for it but here when we go to the documentation you have to look into the details of how pytorch expects these inputs and basically the issue here is pytorch expects if you have multi-dimensional input which we do because we have a b by t by c tensor then it actually really wants the channels to be the second dimension here so if you so basically it wants a b by c by t instead of a b by p

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

And so it's just the details of how PyTorch treats these kinds of inputs. And so we don't actually want to deal with that. So what we're going to do instead is we need to basically reshape our logits.

So here's what I like to do. I like to take, basically give names to the dimensions. So logits.shape is b by t by c and unpack those numbers.

And then let's say that logits equals logits.view. And we want it to be a b times t by c. So just a two-dimensional array, right? So we're going to take all of these positions here, and we're going to stretch them out in a one-dimensional sequence and preserve the channel dimension as the second dimension. So we're just kind of like stretching out the array so it's two-dimensional. And in that case, it's going to better conform to what PyTorch sort of expects in its dimensions.

Now, we have to do the same to targets, because currently targets are of shape b by t, and we want it to be just b times t. So one-dimensional. Now, alternatively, you could also still just do minus one, because PyTorch will guess what this should be if you want to lay it out. But let me just be explicit and say b times t. Once we reshape this, it will match the cross-entropy case, and then we should be able to evaluate our loss.

Okay, so with that right now, and we can do loss. And so currently, we see that the loss is 4.87. Now, because we have 65 possible vocabulary elements, we can actually guess at what the loss should be. And in particular, we covered negative log-likelihood in a lot of detail.

We are expecting log or ln of 1 over 65 and negative of that. So we're expecting the loss to be about 4.17, but we're getting 4.87. And so that's telling us that the initial predictions are not super diffuse. They've got a little bit of entropy, and so we're guessing wrong.

So yes, but actually, we are able to evaluate the loss. Okay, so now that we can evaluate the quality of the model on some data, we'd like to also be able to generate from the model. So let's do the generation.

Now, I'm going to go again a little bit faster here, because I covered all this already in previous videos. So here's a generate function for the model. So we take the same kind of input, idx here.

And basically, this is the current context of some characters in some batch. So it's also b by t. And the job of generate is to basically take this b by t and extend it to be b by t plus 1, plus 2, plus 3. And so it's just basically it continues the generation in all the batch dimensions in the time dimension. So that's its job.

And it will do that for max new tokens. So you can see here on the bottom, there's going to be some stuff here. But on the bottom, whatever is predicted is concatenated on top of the previous idx along the first dimension, which is the time dimension to create a b by t plus 1. So that becomes a new idx.

So the job of generate is to take a b by t and make it a b by t plus 1, plus 2, plus 3. As many as we want max new tokens. So this is the generation from the model. Now, inside the generation, what are we doing? We're taking the current indices.

We're getting the predictions. So we get those are in the logits. And then the loss here is going to be ignored, because we're not using that.

And we have no targets that are sort of ground truth targets that we're going to be comparing with. Then once we get the logits, we are only focusing on the last step. So instead of a b by t by c, we're going to pluck out the negative one, the last element in the time dimension, because those are the predictions for what comes next.

So that gives us the logits, which we then convert to probabilities via softmax. And then we use Torch.multinomial to sample from those probabilities. And we ask PyTorch to give us one sample.

And so idx next will become a b by 1. Because in each one of the batch dimensions, we're going to have a single prediction for what comes next. So this numSamples equals 1 will make this b a 1. And then we're going to take those integers that come from the sampling process, according to the probability distribution given here. And those integers got just concatenated on top of the current, sort of like running stream of integers.

And this gives us a b by t plus 1. And then we can return that. Now, one thing here is, you see how I'm calling self of idx, which will end up going to the forward function. I'm not providing any targets.

So currently, this would give an error because targets is sort of like not given. So targets has to be optional. So targets is none by default.

And then if targets is none, then there's no loss to create. So it's just loss is none. But else, all of this happens and we can create a loss.

So this will make it so if we have the targets, we provide them and get a loss. If we have no targets, we'll just get the logits. So this here will generate from the model.

And let's take that for a ride now. Oops. So I have another code chunk here, which will generate from the model.

And OK, this is kind of crazy. So maybe let me break this down. So these are the idx, right? So I'm creating a batch will be just one.

Time will be just one. So I'm creating a little one by one tensor and it's holding a zero. And the dtype, the data type is integer.

So zero is going to be how we kick off the generation. And remember that zero is the element standing for a newline character. So it's kind of like a reasonable thing to feed in as the very first character in a sequence to be the newline.

So it's going to be idx, which we're going to feed in here. Then we're going to ask for 100 tokens. And then end.generate will continue that.

Now, because generate works on the level of batches, we then have to index into the zeroth row to basically unplug the single batch dimension that exists. And then that gives us a timesteps, just a one-dimensional array of all the indices, which we will convert to simple Python list from PyTorch tensor, so that that can feed into our decode function and convert those integers into text. So let me bring this back.

And we're generating 100 tokens. Let's run. And here's the generation that we achieved.

So obviously it's garbage. And the reason it's garbage is because this is a totally random model. So next up, we're going to want to train this model.

Now, one more thing I wanted to point out here is this function is written to be general, but it's kind of like ridiculous right now because we're feeding in all this. We're building out this context. And we're concatenating it all.

And we're always feeding it all into the model. But that's kind of ridiculous because this is just a simple bigram model. So to make, for example, this prediction about k, we only needed this w. But actually what we fed into the model is we fed the entire sequence.

And then we only looked at the very last piece and predicted k. So the only reason I'm writing it in this way is because right now this is a bigram model. But I'd like to keep this function fixed. And I'd like it to work later when our characters actually basically look further in the history.

And so right now the history is not used. So this looks silly. But eventually the history will be used.

And so that's why we want to do it this way. So just a quick comment on that. So now we see that this is random.

So let's train the model so it becomes a bit less random. OK, let's now train the model. So first what I'm going to do is I'm going to create a PyTorch optimization object.

So here we are using the optimizer AdamW. Now in the Makemore series, we've only ever used stochastic gradient descent, the simplest possible optimizer, which you can get using the SGD instead. But I want to use Adam, which is a much more advanced and popular optimizer.

And it works extremely well. For a typical good setting for the learning rate is roughly 3e-4. But for very, very small networks, like is the case here, you can get away with much, much higher learning rates, learning negative 3 or even higher probably.

But let me create the optimizer object, which will basically take the gradients and update the parameters using the gradients. And then here, our batch size up above was only 4. So let me actually use something bigger, let's say 32. And then for some number of steps, we are sampling a new batch of data.

We're evaluating the loss. We're zeroing out all the gradients from the previous step, getting the gradients for all the parameters, and then using those gradients to update our parameters. So typical training loop, as we saw in the Makemore series.

So let me now run this for, say, 100 iterations, and let's see what kind of losses we're going to get. So we started around 4.7, and now we're getting down to like 4.6, 4.5, etc. So the optimization is definitely happening, but let's sort of try to increase the number of iterations and only print at the end, because we probably want to train for longer.

Okay, so we're down to 3.6, roughly. Roughly down to 3. This is the most janky optimization. Okay, it's working.

Let's just do 10,000. And then from here, we want to copy this. And hopefully, we're going to get something reasonable.

And of course, it's not going to be Shakespeare from a bigram model, but at least we see that the loss is improving. And hopefully, we're expecting something a bit more reasonable. Okay, so we're down at about 2.5-ish. Let's see what we get.

Okay, a dramatic improvement, certainly, on what we had here. So let me just increase the number of tokens. Okay, so we see that we're starting to get something at least like reasonable-ish.

Certainly not Shakespeare, but the model is making progress. So that is the simplest possible model. So now, what I'd like to do is... Obviously, this is a very simple model because the tokens are not talking to each other.

So given the previous context of whatever was generated, we're only looking at the very last character to make the predictions about what comes next. So now, these tokens have to start talking to each other and figuring out what is in the context so that they can make better predictions for what comes next. And this is how we're going to kick off the transformer.

Okay, so next, I took the code that we developed in this Jupyter Notebook and I converted it to be a script. And I'm doing this because I just want to simplify our intermediate work into just the final product that we have at this point. So in the top here, I put all the hyperparameters that we've defined.

I introduced a few, and I'm going to speak to that in a little bit. Otherwise, a lot of this should be recognizable. Reproducibility, read data, get the encoder and the decoder, create the train and test splits.

Use the data loader that gets a batch of the inputs and targets. This is new, and I'll talk about it in a second. Now, this is the bigram language model that we developed.

And it can forward and give us a logits and loss, and it can generate. And then here, we are creating the optimizer, and this is the training loop. So everything here should look pretty familiar.

Now, some of the small things that I added. Number one, I added the ability to run on a GPU if you have it. So if you have a GPU, then this will use CUDA instead of just CPU, and everything will be a lot more faster.

Now, when device becomes CUDA, then we need to make sure that when we load the data, we move it to device. When we create the model, we want to move the model parameters to device. So as an example, here we have the NN embedding table, and it's got a dot weight inside it, which stores the lookup table.

So that would be moved to the GPU, so that all the calculations here happen on the GPU, and they can be a lot faster. And then finally here, when I'm creating the context that feeds it to generate, I have to make sure that I create on the device. Number two, what I introduced is the fact that here in the training loop, here I was just printing the loss.item inside the training loop.

But this is a very noisy measurement of the current loss, because every batch will be more or less lucky. And so what I want to do usually is I have an estimate loss function. And the estimate loss basically then goes up here, and it averages up the loss over multiple batches.

So in particular, we're going to iterate eval iter times, and we're going to basically get our loss, and then we're going to get the average loss for both splits. And so this will be a lot less noisy. So here, when we call the estimate loss, we're going to report the pretty accurate train and validation loss.

Now, when we come back up, you'll notice a few things here. I'm setting the model to evaluation phase, and down here, I'm resetting it back to training phase. Now, right now for our model as is, this doesn't actually do anything, because the only thing inside this model is this nn.embedding. And this network would behave the same in both evaluation mode and training mode.

We have no dropout layers, we have no batch norm layers, etc. But it is a good practice to think through what mode your neural network is in, because some layers will have different behavior at inference time or training time. And there's also this context manager, torch.nograd. And this is just telling PyTorch that everything that happens inside this function, we will not call .backward on.

And so PyTorch can be a lot more efficient with its memory use, because it doesn't have to store all the intermediate variables, because we're never going to call backward. And so it can be a lot more efficient in that way. So also a good practice to tell PyTorch when we don't intend to do backpropagation.

So right now, this script is about 120 lines of code, and that's kind of our starter code. I'm calling it bigram.py, and I'm going to release it later. Now running this script gives us output in the terminal, and it looks something like this.

It basically, as I ran this code, it was giving me the train loss and the val loss, and we see that we convert to somewhere around 2.5 with the bigram model. And then here's the sample that we produced at the end. And so we have everything packaged up in the script, and we're in a good position now to iterate on this.

Okay, so we are almost ready to start writing our very first self-attention block for processing these tokens. Now, before we actually get there, I want to get you used to a mathematical trick that is used in the self-attention inside a transformer, and is really just at the heart of an efficient implementation of self-attention. And so I want to work with this toy example to just get you used to this operation, and then it's going to make it much more clear once we actually get to it.

In the script again. So let's create a B by T by C, where B, T, and C are just 4, 8, and 2 in the toy example. And these are basically channels, and we have batches, and we have the time component, and we have some information at each point in the sequence, so C. Now, what we would like to do is we would like these tokens, so we have up to eight tokens here in a batch, and these eight tokens are currently not talking to each other, and we would like them to talk to each other.

We'd like to couple them. And in particular, we want to couple them in a very specific way. So the token, for example, at the fifth location, it should not communicate with tokens in the sixth, seventh, and eighth location, because those are future tokens in the sequence.

The token on the fifth location should only talk to the one in the fourth, third, second, and first. So it's only... So information only flows from previous context to the current time step, and we cannot get any information from the future, because we are about to try to predict the future. So what is the easiest way for tokens to communicate? The easiest way I would say is, okay, if we're a fifth token and I'd like to communicate with my past, the simplest way we can do that is to just do an average of all the preceding elements.

So for example, if I'm the fifth token, I would like to take the channels that make up... that are information at my step, but then also the channels from the fourth step, third step, second step, and the first step. I'd like to average those up, and then that would become sort of like a feature vector that summarizes me in the context of my history. Now, of course, just doing a sum or like an average is an extremely weak form of interaction.

Like this communication is extremely lossy. We've lost a ton of information about spatial arrangements of all those tokens. But that's okay for now.

We'll see how we can bring that information back later. For now, what we would like to do is for every single batch element independently, for every teeth token in that sequence, we'd like to now calculate the average of all the vectors in all the previous tokens and also at this token. So let's write that out.

I have a small snippet here, and instead of just fumbling around, let me just copy paste it and talk to it. So in other words, we're going to create x, and B-O-W is short for bag of words, because bag of words is kind of like a term that people use when you are just averaging up things. So this is just a bag of words.

Basically, there's a word stored on every one of these eight locations, and we're doing a bag of words. We're just averaging. So in the beginning, we're going to say that it's just initialized at zero, and then I'm doing a for loop here.

So we're not being efficient yet. That's coming. But for now, we're just iterating over all the batch dimensions independently, iterating over time, and then the previous tokens are at this batch dimension, and then everything up to and including the Tth token.

So when we slice out x in this way, xprev becomes of shape how many T elements there were in the past, and then, of course, C. So all the two-dimensional information from these little tokens. So that's the previous sort of chunk of tokens from my current sequence, and then I'm just doing the average or the mean over the zeroth dimension. So I'm averaging out the time here, and I'm just going to get a little C one-dimensional vector, which I'm going to store in x bag of words.

So I can run this, and this is not going to be very informative, because let's see. So this is x of zero. So this is the zeroth batch element, and then xbow at zero.

Now, you see how at the first location here, you see that the two are equal, and that's because we're just doing an average of this one token. But here, this one is now an average of these two, and now this one is an average of these three, and so on. And this last one is the average of all of these elements.

So vertical average, just averaging up all the tokens, now gives this outcome here. So this is all well and good, but this is very inefficient. Now, the trick is that we can be very, very efficient about doing this using matrix multiplication.

So that's the mathematical trick, and let me show you what I mean. Let's work with the toy example here. Let me run it, and I'll explain.

So I have a simple matrix here that is a 3x3 of all ones, a matrix B of just random numbers, and it's a 3x2, and a matrix C, which will be 3x3 multiply 3x2, which will give out a 3x2. So here, we're just using matrix multiplication. So A multiply B gives us C. Okay, so how are these numbers in C achieved, right? So this number in the top left is the first row of A dot product with the first column of B. And since all the row of A right now is all just ones, then the dot product here with this column of B is just going to do a sum of this column.

So 2 plus 6 plus 6 is 14. The element here in the output of C is also the first column here, the first row of A multiplied now with the second column of B. So 7 plus 4 plus 5 is 16. Now, you see that there's repeating elements here.

So this 14 again is because this row is again all ones, and it's multiplying the first column of B. So we get 14. And this one is, and so on. So this last number here is the last row dot product last column.

Now, the trick here is the following. This is just a boring number of, it's just a boring array of all ones, but Torch has this function called trill, which is short for a triangular, something like that. And you can wrap it in Torch.once, and it will just return the lower triangular portion of this.

Okay, so now it will basically zero out these guys here. So we just get the lower triangular part. Well, what happens if we do that? So now we'll have A like this and B like this.

Now, what are we getting here in C? Well, what is this number? Well, this is the first row times the first column. And because this is zeros, these elements here are now ignored. So we just get a 2. And then this number here is the first row times the second column.

And because these are zeros, they get ignored. And it's just 7. The 7 multiplies this 1. But look what happened here. Because this is 1 and then zeros, what ended up happening is we're just plucking out this row of B, and that's what we got.

Now, here we have 1, 1, 0. So here, 1, 1, 0 dot product with these two columns will now give us 2 plus 6, which is 8, and 7 plus 4, which is 11. And because this is 1, 1, 1, we ended up with the addition of all of them. And so basically, depending on how many 1s and 0s we have here, we are basically doing a sum, currently, of the variable number of these rows.

And that gets deposited into C. So currently, we're doing sums because these are 1s. But we can also do average, right? And you can start to see how we could do average of the rows of B sort of in an incremental fashion. Because we don't have to... We can basically normalize these rows so that they sum to 1, and then we're going to get an average.

So if we took A, and then we did A equals A divide, a torch dot sum of A in the 1th dimension, and then let's keep them as true. So therefore, the broadcasting will work out. So if I rerun this, you see now that these rows now sum to 1. So this row is 1, this row is 0.5, 0.5 is 0, and here we get 1 thirds.

And now when we do A multiply B, what are we getting? Here we are just getting the first row, first row. Here now we are getting the average of the first two rows. Okay, so 2 and 6 average is 4, and 4 and 7 average is 5.5. And on the bottom here, we are now getting the average of these three rows.

So the average of all of elements of B are now deposited here. And so you can see that by manipulating these elements of this multiplying matrix, and then multiplying it with any given matrix, we can do these averages in this incremental fashion, because we just get, and we can manipulate that based on the elements of A. Okay, so that's very convenient. So let's swing back up here and see how we can vectorize this and make it much more efficient using what we've learned.

So in particular, we are going to produce an array A, but here I'm going to call it Y, short for weights. But this is our A, and this is how much of every row we want to average up. And it's going to be an average because you can see that these rows sum to 1. So this is our A, and then our B in this example, of course, is X. So what's going to happen here now is that we are going to have an expo 2. And this expo 2 is going to be Y multiplying Rx.

So let's think this through. Y is T by T, and this is matrix multiplying in PyTorch a B by T by C. And it's giving us what shape. So PyTorch will come here and it will see that these shapes are not the same.

So it will create a batch dimension here, and this is a batch matrix multiply. And so it will apply this matrix multiplication in all the batch elements in parallel and individually. And then for each batch element, there will be a T by T multiplying T by C exactly as we had below.

So this will now create B by T by C, and expo 2 will now become identical to expo. So we can see that Torch.allclose of expo and expo 2 should be true. So this kind of convinces us that these are, in fact, the same.

So expo and expo 2, if I just print them, okay, we're not going to be able to just stare it down, but well, let me try expo basically just at the zeroth element and expo 2 at the zeroth element. So just the first batch, and we should see that this and that should be identical, which they are. Right? So what happened here? The trick is we were able to use batched matrix multiply to do this aggregation, really.

And it's a weighted aggregation. And the weights are specified in this T by T array. And we're basically doing weighted sums.

And these weighted sums are, according to the weights inside here, they take on sort of this triangular form. And so that means that a token at the Tth dimension will only get sort of information from the tokens perceiving it. So that's exactly what we want.

And finally, I would like to rewrite it in one more way. And we're going to see why that's useful. So this is the third version.

And it's also identical to the first and second. But let me talk through it. It uses softmax.

So trill here is this matrix, lower triangular ones. Weigh begins as all zero. Okay, so if I just print weigh in the beginning, it's all zero.

Then I use masked fill. So what this is doing is weight.masked fill, it's all zeros. And I'm saying, for all the elements where trill is equal to equal zero, make them be negative infinity.

So all the elements where trill is zero will become negative infinity now. So this is what we get. And then the final one here is softmax.

So if I take a softmax along every single, so dim is negative one. So along every single row, if I do a softmax, what is that going to do? Well, softmax is also like a normalization operation, right? And so spoiler alert, you get the exact same matrix. Let me bring back the softmax.

And recall that in softmax, we're going to exponentiate every single one of these and then we're going to divide by the sum. And so if we exponentiate every single element here, we're going to get a one. And here we're going to get basically zero, zero, zero, zero, everywhere else.

And then when we normalize, we just get one. Here we're going to get one, one, and then zeros. And then softmax will again divide and this will give us 0.5, 0.5, and so on.

And so this is also the same way to produce this mask. Now, the reason that this is a bit more interesting, and the reason we're going to end up using it in self-attention, is that these weights here...

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Begin with zero and you can think of this as like an interaction strength or like an affinity. So basically it's telling us how much of each token from the past do we want to aggregate and average up. And then this line is saying tokens from the past cannot communicate. 

By setting them to negative infinity we're saying that we will not aggregate anything from those tokens. And so basically this then goes through softmax and through the weighted and this is the aggregation through matrix multiplication. And so what this is now is you can think of these as these zeros are currently just set by us to be zero but a quick preview is that these affinities between the tokens are not going to be just constant at zero. 

They're going to be data dependent. These tokens are going to start looking at each other and some tokens will find other tokens more or less interesting. And depending on what their values are they're going to find each other interesting to different amounts and I'm going to call those affinities I think. 

And then here we are saying the future cannot communicate with the past. We're going to clamp them. And then when we normalize and sum we're going to aggregate sort of their values depending on how interesting they find each other. 

And so that's the preview for self attention. And basically long story short from this entire section is that you can do weighted aggregations of your past elements by having by using matrix multiplication of a lower triangular fashion. And then the elements here in the lower triangular part are telling you how much of each element fuses into this position. 

So we're going to use this trick now to develop the self attention block. So first let's get some quick preliminaries out of the way. First the thing I'm kind of bothered by is that you see how we're passing in vocab size into the constructor? There's no need to do that because vocab size is already defined up top as a global variable. 

So there's no need to pass this stuff around. Next what I want to do is I don't want to actually create I want to create like a level of indirection here where we don't directly go to the embedding for the logits but instead we go through this intermediate phase because we're going to start making that bigger. So let me introduce a new variable nembed. 

It's short for number of embedding dimensions. So nembed here will be say 32. That was a suggestion from GitHub Copilot by the way. 

It also suggested 32 which is a good number. So this is an embedding table and only 32 dimensional embeddings. So then here this is not going to give us logits directly.

Instead this is going to give us token embeddings. That's what I'm going to call it. And then to go from the token embeddings to the logits we're going to need a linear layer. 

So self.lmhead, let's call it, short for language modeling head, is nnlinear from nembed up to vocab size. And then when we swing over here we're actually going to get the logits by exactly what the Copilot says. Now we have to be careful here because this C and this C are not equal. 

This is nembed C and this is vocab size. So let's just say that nembed is equal to C. And then this just creates one spurious layer of indirection through a linear layer but this should basically run. So we see that this runs and this currently looks kind of spurious but we're going to build on top of this. 

Now next up, so far we've taken these indices and we've encoded them based on the identity of the tokens inside IDX. The next thing that people very often do is that we're not just encoding the identity of these tokens but also their position. So we're going to have a second position embedding table here. 

So self.position embedding table is an embedding of block size by nembed. And so each position from 0 to block size minus 1 will also get its own embedding vector. And then here, first let me decode B by T from IDX.shade. And then here we're also going to have a pos embedding which is the positional embedding and this is tor-arrange. 

So this will be basically just integers from 0 to T minus 1. And all of those integers from 0 to T minus 1 get embedded through the table to create a T by C. And then here this gets renamed to just say X and X will be the addition of the token embeddings with the positional embeddings. And here the broadcasting node will work out. So B by T by C plus T by C. This gets right aligned, a new dimension of 1 gets added and it gets broadcasted across batch. 

So at this point X holds not just the token identities but the positions at which these tokens occur. And this is currently not that useful because of course we just have a simple bigram model so it doesn't matter if you're on the fifth position, the second position or wherever it's all translation invariant at this stage. So this information currently wouldn't help but as we work on the self-attention block we'll see that this starts to matter. 

Okay so now we get the crux of self-attention. So this is probably the most important part of this video to understand. We're going to implement a small self-attention for a single individual head as they're called.

So we start off with where we were. So all of this code is familiar. So right now I'm working with an example where I change the number of channels from 2 to 32. 

So we have a 4 by 8 arrangement of tokens and the information at each token is currently 32 dimensional but we just are working with random numbers. Now we saw here that the code as we had it before does a simple weight, a simple average of all the past tokens and the current token. So it's just the previous information and current information is just being mixed together in an average and that's what this code currently achieves and it does so by creating this lower triangular structure which allows us to mask out this weight matrix that we create. 

So we mask it out and then we normalize it and currently when we initialize the affinities between all the different tokens or nodes, I'm going to use those terms interchangeably, so when we initialize the affinities between all the different tokens to be 0 then we see that weight gives us this structure where every single row has these uniform numbers and so that's what then in this matrix multiply makes it so that we're doing a simple average. Now we don't actually want this to be all uniform because different tokens will find different other tokens more or less interesting and we want that to be data dependent. So for example if I'm a vowel then maybe I'm looking for consonants in my past and maybe I want to know what those consonants are and I want that information to flow to me and so I want to now gather information from the past but I want to do it in a data dependent way and this is the problem that self-attention solves. 

Now the way self-attention solves this is the following. Every single node or every single token at each position will emit two vectors. It will emit a query and it will emit a key. 

Now the query vector roughly speaking is what am I looking for and the key vector roughly speaking is what do I contain and then the way we get affinities between these tokens now in a sequence is we basically just do a dot product between the keys and the queries. So my query dot products with all the keys of all the other tokens and that dot product now becomes weigh. And so if the key and the query are sort of aligned they will interact to a very high amount and then I will get to learn more about that specific token as opposed to any other token in the sequence. 

So let's implement this now. We're going to implement a single what's called head of self-attention. So this is just one head. 

There's a hyperparameter involved with these heads which is the head size and then here I'm initializing linear modules and I'm using bias equals false so these are just going to apply a matrix multiply with some fixed weights. And now let me produce a k and q by forwarding these modules on x. So the size of this will now become b by t by 16 because that is the head size and the same here b by t by 16. So this being the head size. 

So you see here that when I forward this linear on top of my x all the tokens in all the positions in the b by t arrangement all of them in parallel and independently produce a key and a query. So no communication has happened yet but the communication comes now. All the queries will dot product with all the keys. 

So basically what we want is we want weigh now or the affinities between these to be query multiplying key but we have to be careful with we can't matrix multiply this we actually need to transpose k but we have to be also careful because these are when you have the batch dimension so in particular we want to transpose the last two dimensions dimension negative 1 and dimension negative 2. So negative 2 negative 1 and so this matrix multiply now will basically do the following b by t by t right so for every row of b we're now going to have a t squared matrix giving us the affinities and these are now the weight so they're not zeros they are now coming from this dot product between the keys and the queries. So this can now run I can I can run this and the weighted aggregation now is a function in a data-dependent manner between the keys and queries of these nodes. So just inspecting what happened here the weigh takes on this form and you see that before weigh was just a constant so it was applied in the same way to all the batch elements but now every single batch elements will have different sort of weigh because every single batch element contains different tokens at different positions and so this is now data-dependent. 

So when we look at just the 0th row for example in the input these are the weights that came out and so you can see now that they're not just exactly uniform and in particular as an example here for the last row this was the 8th token and the 8th token knows what content it has and it knows at what position it's in and now the 8th token based on that creates a query. Hey I'm looking for this kind of stuff I'm a vowel I'm on the 8th position I'm looking for any consonants at positions up to 4 and then all the nodes get to emit keys and maybe one of the channels could be I am a consonant and I am in a position up to 4 and that key would have a high number in that specific channel and that's how the query and the key when they dot product they can find each other and create a high affinity and when they have a high affinity like say this token was pretty interesting to to this 8th token when they have a high affinity then through the softmax I will end up aggregating a lot of its information into my position and so I'll get to learn a lot about it now just this we're looking at way after this has already happened let me erase this operation as well so let me erase the masking and the softmax just to show you the under the hood internals and how that works so without the masking and the softmax way comes out like this right this is the outputs of the dot products and these are the raw outputs and they take on values from negative you know 2 to positive 2 etc so that's the raw interactions and raw affinities between all the nodes but now if I'm a if I'm a fifth node I will not want to aggregate anything from the sixth node seventh node and the eighth node so actually we use the upper triangular masking so those are not allowed to communicate and now we actually want to have a nice distribution so we don't want to aggregate negative 0.11 of this node that's crazy so instead we exponentiate and normalize and now we get a nice distribution that sums to 1 and this is telling us now in the data dependent manner how much of information to aggregate from any of these tokens in the past so that's way and it's not zeros anymore but it's calculated in this way now there's one more part to a single self-attention head and that is that when we do the aggregation we don't actually aggregate the tokens exactly we aggregate we produce one more value here and we call that the value so in the same way that we produced p and query we're also going to create a value and then here we don't aggregate X we calculate a V which is just achieved by propagating this linear on top of X again and then we output way multiplied by V so V is the elements that we aggregate or the vector that we aggregate instead of the raw X and now of course this will make it so that the output here of the single head will be 16 dimensional because that is the head size so you can think of X as kind of like private information to this token if you if you think about it that way so X is kind of private to this token so I'm a fifth token at some and I have some identity and my information is kept in vector X and now for the purposes of the single head here's what I'm interested in here's what I have and if you find me interesting here's what I will communicate to you and that's stored in V and so V is the thing that gets aggregated for the purposes of this single head between the different nodes and that's basically the self attention mechanism this is this is what it does there are a few notes that I would make like to make about attention number one attention is a communication mechanism you can really think about it as a communication mechanism where you have a number of nodes in a directed graph where basically you have edges pointing between nodes like this and what happens is every node has some vector of information and it gets to aggregate information via a weighted sum from all the nodes that point to it and this is done in a data dependent manner so depending on whatever data is actually stored at each node at any point in time now our graph doesn't look like this our graph has a different structure we have eight nodes because the block size is eight and there's always eight tokens and the first node is only pointed to by itself the second node is pointed to by the first node and itself all the way up to the eighth node which is pointed to by all the previous nodes and itself and so that's the structure that our directed graph has or happens happens to have in autoregressive sort of scenario like language modeling but in principle attention can be applied to any arbitrary directed graph and it's just a communication mechanism between the nodes the second note is that notice that there's no notion of space so attention simply acts over like a set of vectors in this graph and so by default these nodes have no idea where they are positioned in the space and that's why we need to encode them positionally and sort of give them some information that is anchored to a specific position so that they sort of know where they are and this is different than for example from convolution because if you run for example a convolution operation over some input there is a very specific sort of layout of the information in space and the convolutional filters sort of act in space and so it's it's not like an attention. An attention is just a set of vectors out there in space they communicate and if you want them to have a notion of space you need to specifically add it which is what we've done when we calculated the relative the positional encode encodings and added that information to the vectors. 

The next thing that I hope is very clear is that the elements across the batch dimension which are independent examples never talk to each other they're always processed independently and this is a batched matrix multiply that applies basically a matrix multiplication kind of in parallel across the batch dimension so maybe it would be more accurate to say that in this analogy of a directed graph we really have because the batch size is four we really have four separate pools of eight nodes and those eight nodes only talk to each other but in total there's like 32 nodes that are being processed but there's sort of four separate pools of eight you can look at it that way. The next note is that here in the case of language modeling we have this specific structure of directed graph where the future tokens will not communicate to the past tokens but this doesn't necessarily have to be the constraint in the general case and in fact in many cases you may want to have all of the nodes talk to each other fully so as an example if you're doing sentiment analysis or something like that with a transformer you might have a number of tokens and you may want to have them all talk to each other fully because later you are predicting for example the sentiment of the sentence and so it's okay for these nodes to talk to each other and so in those cases you will use an encoder block of self-attention and all it means that it's an encoder block is that you will delete this line of code allowing all the nodes to completely talk to each other what we're implementing here is sometimes called a decoder block and it's called a decoder because it is sort of like decoding language and it's got this autoregressive format where you have to mask with the triangular matrix so that notes from the future never talk to the past because they would give away the answer and so basically in encoder blocks you would delete this allow all the nodes to talk in decoder blocks this will always be present so that you have this triangular structure but both are allowed and attention doesn't care attention supports arbitrary connectivity between nodes the next thing I wanted to comment on is you keep me you keep hearing me say attention self-attention etc there's actually also something called cross attention what is the difference so basically the reason this attention is self-attention is because the keys queries and the values are all coming from the same source from X so the same source X produces keys queries and values so these nodes are self-attending but in principle attention is much more general than that so for example in encoder decoder transformers you can have a case where the queries are produced from X but the keys and the values come from a whole separate external source and sometimes from encoder blocks that encode some context that we'd like to condition on and so the keys and the values will actually come from a whole separate source those are nodes on the side and here we're just producing queries and we're reading off information from the side so cross attention is used when there's a separate source of nodes we'd like to pull information from into our nodes and it's self attention if we just have nodes that would like to look at each other and talk to each other so this attention here happens to be self-attention but in principle attention is a lot more general okay and the last note at this stage is if we come to the attention is all you need paper here we've already implemented attention so given query key and value we've multiplied the query on a key we've soft maxed it and then we are aggregating the values there's one more thing that we're missing here which is the dividing by one over square root of the head size the DK here is the head size why aren't they doing this why is this important so they call it a scaled attention and it's kind of like an important normalization to basically have the problem is if you have unit Gaussian inputs so zero mean unit variance k and q are unit Gaussian and if you just do way naively then you see that your way actually will be the variance will be on the order of head size which in our case is 16 but if you multiply by 1 over head size square root so this is square root and this is 1 over then the variance of way will be 1 so it will be preserved now why is this important you'll notice that way here will feed into softmax and so it's really important especially at initialization that way be fairly diffuse so in our case here we sort of locked out here and way had a fairly diffuse numbers here so like this now the problem is that because of softmax if weight takes on very positive and very negative numbers inside it softmax will actually converge towards 1-hot vectors and so I can illustrate that here say we are applying softmax to a tensor of values that are very close to 0 then we're gonna get a diffuse thing out of softmax but the moment I take the exact same thing and I start sharpening it making it bigger by multiplying these numbers by 8 for example you'll see that the softmax will start to sharpen and in fact it will sharpen towards the max so it will sharpen towards whatever number here is the highest and so basically we don't want these values to be too extreme especially at initialization otherwise softmax will be way too peaky and you're basically aggregating information from like a single node every node just aggregates information from a single other node that's not what we want especially at initialization and so the scaling is used just to control the variance at initialization okay so having said all that let's now take our self-attention knowledge and let's take it for a spin so here in the code I've created this head module and implements a single head of self-attention so you give it a head size and then here it creates the key query and the value linear layers typically people don't use biases in these so those are the linear projections that we're going to apply to all of our nodes now here I'm creating this trill variable trill is not a parameter of the module so in sort of PyTorch naming conventions this is called a buffer it's not a parameter and you have to call it you have to assign it to the module using a register buffer so that creates the trill the lower triangular matrix and when we're given the input X this should look very familiar now we calculate the keys the queries we calculate the attention scores in sideway we normalize it so we're using scaled attention here then we make sure that future doesn't communicate with the past so this makes it a decoder block and then softmax and then aggregate the value and output then here in the language model I'm creating a head in the constructor and I'm calling it self-attention head and the head size I'm going to keep as the same an embed just for now and then here once we've encoded the information with the token embeddings and the position embeddings we're simply going to feed it into the self-attention head and then the output of that is going to go into the decoder language modeling head and create the logits so this is sort of the simplest way to plug in a self-attention component into our network right now I had to make one more change which is that here in the generate we have to make sure that our IDX that we feed into the model because now we're using positional embeddings we can never have more than block size coming in because if IDX is more than block size then our position embedding table is going to run out of scope because it only has embeddings for up to block size and so therefore I added some code here to crop the context that we're going to feed into self so that we never pass in more than block size elements so those are the changes and let's now train the network okay so I also came up to the script here and I decreased the learning rate because the self-attention can't tolerate very very high learning rates and then I also increased number of iterations because the learning rate is lower and then I trained it and previously we were only able to get to up to 2.5 and now we are down to 2.4 so we definitely see a little bit of improvement from 2.5 to 2.4 roughly but the text is still not amazing so clearly the self-attention head is doing some useful communication but we still have a way to go okay so now we've implemented the scale dot product attention now next up in the attention is all you need paper there's something called multi-head attention and what is multi-head attention it's just applying multiple attentions in parallel and concatenating the results so they have a little bit of diagram here I don't know if this is super clear it's really just multiple attentions in parallel so let's implement that fairly straightforward if we want a multi-head attention then we want multiple heads of self-attention running in parallel so in PyTorch we can do this by simply creating multiple heads so however many heads you want and then what is the head size of each and then we run all of them in parallel into a list and simply concatenate all of the outputs and we're concatenating over the channel dimension so the way this looks now is we don't have just a single attention that has a head size of 32 because remember n-embed is 32 instead of having one communication channel we now have four communication channels in parallel and each one of these communication channels typically will be smaller correspondingly so because we have four communication channels we want eight-dimensional self-attention and so from each communication channel we're getting together eight-dimensional vectors and then we have four of them and that concatenates to give us 32 which is the original n-embed and so this is kind of similar to if you're familiar with convolutions this is kind of like a group convolution because basically instead of having one large convolution we do convolution in groups and that's multi-headed self-attention and so then here we just use SA heads self-attention heads instead now I actually ran it and scrolling down I ran the same thing and then we now get this down to 2.28 roughly and the output is still the generation is still not amazing but clearly the validation loss is improving because we were at 2.4 just now and so it helps to have multiple communication channels because obviously these tokens have a lot to talk about they want to find the consonants the vowels they want to find the vowels just from certain positions they want to find any kinds of different things and so it helps to create multiple independent channels of communication gather lots of different types of data and then decode the output now going back to the paper for a second of course I didn't explain this figure in full detail but we are starting to see some components of what we've already implemented we have the positional encodings the token encodings that add we have the masked multi-headed attention implemented now here's another multi-headed attention which is a cross attention to an encoder which we haven't we're not going to implement in this case I'm going to come back to that later but I want you to notice that there's a feedforward part here and then this is grouped into a block that gets repeated again and again now the feedforward part here is just a simple multi-layer perceptron so the multi-headed so here position wise feedforward networks it's just a simple little MLP so I want to start basically in a similar fashion also adding computation into the network and this computation is on a per node level so I've already implemented it and you can see the diff highlighted on the left here when I've added or changed things now before we had the multi-headed self-attention that did the communication but we went way too fast to calculate the logits so the tokens looked at each other but didn't really have a lot of time to think on what they found from the other tokens and so what I've implemented here is a little feedforward single layer and this little layer is just a linear followed by a relu nonlinearity and that's it so it's just a little layer and then I call it feedforward and embed and then this feedforward is just called sequentially right after the self attention so we self attend then we feed forward and you'll notice that the feedforward here when it's applying linear this is on a per token level all the tokens do this independently so the self attention is the communication and then once they've gathered all the data now they need to think on that data individually and so that's what feedforward is doing and that's why I've added it here now when I train this the validation loss actually continues to go down now to 2.24 which is down from 2.28 the output still look kind of terrible but at least we've improved the situation and so as a preview we're going to now start to intersperse the communication with the computation and that's also

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)




