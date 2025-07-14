

（50：00）

它不需要我们进行完美的数学计算，也不必关心你可能想引入神经网络的各种不同类型的神经网络构建模块的激活分布。而且，它显著地稳定了训练过程，这就是为什么这些层非常受欢迎。

现在，批量归一化提供的稳定性实际上是以巨大的代价换来的。这个代价就是，如果你仔细思考这里发生的事情，会发现某种极其奇怪且不自然的现象正在发生。过去的情况是，我们有一个单独的样本输入到神经网络中，然后我们计算其激活值和逻辑值，这是一个确定性的过程，因此你会得到这个样本的某些逻辑值。

后来，由于训练效率的考虑，我们开始批量处理样本，但这些样本最初是独立处理的，纯粹出于效率考量。但现在，由于批量归一化的引入，通过批次的归一化操作，我们在神经网络的前向传播和反向传播过程中，将这些样本在数学上耦合在一起。因此，现在任何一个输入样本的隐藏层激活值（HPREACT）和逻辑输出（logits），不仅取决于该样本及其输入，还受到同一批次中所有其他样本的影响。

这些例子都是随机抽样的。也就是说，比如当你观察HPREACT时，它会输入到H（隐藏状态激活），对于任何一个输入例子来说，H实际上会根据批次中其他例子的存在而略有变化。而且，根据其他偶然出现的例子，H会突然改变，如果你想象抽样不同的例子，它会像抖动一样，因为均值和标准差的统计量会受到影响。

因此，你会得到H的抖动和logits的抖动。你可能会认为这是一个错误或不受欢迎的现象，但奇怪的是，这实际上在神经网络训练中被证明是有益的。而且这是一个附带效应。原因在于，你可以将其视为一种正则化手段。具体来说，输入数据经过处理后得到H，而其他样本的存在会使其产生轻微波动。这种机制实际上是在对每个输入样本进行"填充扩展"，同时引入少量熵值。由于这种填充效应，它本质上构成了一种数据增强形式（我们后续会详细讨论），相当于对输入数据进行小幅扩充和扰动。这使得神经网络更难对这些具体样本产生过拟合。通过引入这些噪声，样本数据得到扩展，从而实现了对神经网络的正则化效果。

这就是为什么欺骗性地作为二阶效应，它实际上是一种正则化手段，这也使得我们更难摆脱批量归一化的使用。因为基本上没有人喜欢批量归一化的这一特性——在数学计算和前向传播过程中，批次中的样本是相互耦合的。这会导致各种奇怪的结果（稍后我们也会深入探讨其中一些），还会引发大量错误等等。正因如此，没人喜欢这个特性，所以人们一直在尝试弃用批量归一化，转而采用其他不会耦合批次样本的归一化技术。例如层归一化、实例归一化、组归一化等等，我们稍后会介绍其中一些方法。

简单来说，批量归一化是最早被引入的归一化层类型。它的效果非常好，并且恰好具有这种正则化效应。它稳定了训练过程，人们一直试图移除它并转向其他归一化技术，但一直难以实现，因为它效果非常好。而它效果如此出色的部分原因，再次归功于这种正则化效应，以及它在控制激活值和其分布方面的高效性。这就是批量归一化的简要故事，我想向大家展示这种耦合带来的另一个奇怪结果。这是我之前在验证集上评估损失时一带而过的奇怪现象之一。

基本上，一旦我们训练好了一个神经网络，我们就希望能在某种环境中部署它，并且希望能够输入一个单独的示例，然后从我们的神经网络中得到一个预测结果。但是，当我们的神经网络在前向传播过程中估计一个批次的均值理解偏差统计量时，我们该如何做到这一点呢？现在的神经网络期望输入的是批次。那么，我们如何输入一个单独的示例并获得合理的结果呢？因此，批归一化论文中的建议如下。

我们在这里想要做的是，在训练之后增加一个步骤，一次性计算并设置批量归一化的均值和标准差，基于整个训练集。为了节省时间，我写了这段代码，我们称之为“校准批量归一化统计量”。基本上，我们使用torch.no_grad告诉PyTorch，这部分操作不会调用.backward()，这样效率会更高一些。

我们将获取训练集，为每一个训练样本计算预激活值，然后一次性估算整个训练集的均值和标准差。接着，我们会得到b_mean和b_std，这些是基于整个训练集估算出的固定数值。在这里，我们不再动态估算，而是直接使用b_mean和b_std。

因此在测试时，我们将固定这些参数并在推理过程中使用它们。现在你可以看到，我们得到了基本一致的结果，但我们获得的好处是，现在也可以处理单个样本了，因为均值和标准差现在已经是固定的张量了。话虽如此，实际上没有人愿意在神经网络训练后再单独进行第二阶段来估计这些均值和标准差，因为大家都想偷懒。

因此，这篇批归一化论文实际上还提出了另一个观点，即我们可以在神经网络训练过程中以动态方式估算均值和标准差。这样，我们只需进行单阶段训练，同时在训练过程中动态估算运行中的均值和标准差。让我们来看看具体是如何实现的。

我基本上取的是我们在这个批次上估计的均值，我称之为b，即第i次迭代的均值。然后这里是b和标准差。b和标准差在第i次迭代时的值，明白吗？均值在这里，标准差在这里。所以，到目前为止我什么都没做。我只是移动了一下位置，并为均值和标准差创建了这些额外的变量，并把它们放在这里。所以，到目前为止没有任何变化，但我们现在要做的是在训练过程中持续跟踪这两个值的运行均值。

那我就从这里开始，创建一个名为 bn_mean_running 的变量，初始化为零，再创建一个 bn_std_running 变量，初始化为1。因为一开始我们初始化 w1 和 b1 的方式会让 hpreact 大致服从单位高斯分布，所以均值会接近零，标准差会接近一。

所以我要初始化这些。但接下来我要更新这些。在PyTorch中，这些运行中的均值和标准差实际上并不属于基于梯度的优化部分。我们永远不会对它们求梯度。它们是在训练过程中单独更新的。因此，我们在这里要做的是使用torch.nograd告诉PyTorch，这里的更新不应该构建计算图，因为不会有反向传播。

但这个运行均值基本上会是当前值的0.999倍加上这个新均值的0.001倍。同样地，bn std running大部分情况下会保持原样，但会朝着当前标准差的方向进行小幅更新。正如你在这里看到的，这个更新是在基于梯度的优化之外和侧面进行的。它并没有使用梯度下降法进行更新，而是以一种类似平滑运行平均的简单方式进行更新。

因此，在网络训练过程中，这些预激活值在反向传播时会不断变化和调整，我们会持续跟踪其典型的均值和标准差，并一次性完成估算。现在运行这段代码时，我会以动态方式持续追踪这些数据。当然，我们希望最终得到的bn_mean_running和bn_std_running，能与之前在此处计算得出的数值高度吻合。

这样一来，我们就不需要第二阶段了，因为我们已经将两个阶段合二为一，可以理解为将它们并排放置。PyTorch中的批归一化层也是以这种方式实现的。在训练过程中会执行完全相同的操作，而在后续推理时，则会使用这些隐藏状态均值和标准差的估计运行值。

那我们等待优化收敛，希望运行均值和标准差大致等于这两个值，然后我们就可以直接在这里使用它，而不需要在最后进行这个显式校准阶段。好了，优化完成了。我将重新运行显式估计，然后显式估计得到的 bn 均值在这里，而优化过程中运行估计得到的 bn 均值你可以看到非常非常相似。

不完全相同，但非常接近。同样，bn std是这样，bn std running是这样。你可以看到，它们再次呈现出相当接近的数值——并非完全相同，但非常相似。因此，在这里，我们可以使用bn_mean_running来代替bn_mean，用bn_std_running代替bn_std，希望验证损失不会受到太大影响。基本上是一样的，这样我们就消除了这个显式校准阶段的需要，因为我们在这里内联进行了校准。好了，我们几乎完成了批归一化。

我还有两点需要说明。第一点，我跳过了关于这里为什么会有这个加epsilon的讨论。这个epsilon通常是一个很小的固定数值，比如默认值为1e-5，它的作用主要是防止当批次方差恰好为零时出现除以零的情况。

在这种情况下，我们通常会遇到分母为零的情况，但由于加上了一个极小值ε，分母会变成一个很小的数字，从而使计算过程更加稳定。因此，你也可以在这里添加一个极小的ε值。实际上，这并不会显著改变最终结果。

在我们的案例中，我打算跳过这一点，因为在我们这个非常简单的例子中不太可能发生这种情况。第二点我想让你注意的是，我们在这里的做法是浪费的，虽然很微妙，但就在我们将偏置项加到HPREACT的这一步，这些偏置项实际上毫无用处。因为我们先把偏置加到HPREACT上，随后又为每个神经元计算均值并减去它。所以无论你在这里添加什么偏置项，都会在这里被减掉，因此这些偏置项根本不起作用。

事实上，它们被减去了，不会影响其余的计算。所以如果你看b1.grad，它实际上会是零，因为它被减去了，实际上没有任何影响。因此，每当你使用批量归一化层时，如果前面有任何权重层，比如线性层或卷积层之类的东西，你最好在这里不使用偏置。

所以你不想使用偏置项，然后在这里你也不想添加它，因为那是虚假的。相反，我们在这里有这个批量归一化的偏置项，而这个批量归一化的偏置项现在负责这个分布的偏置，而不是我们最初在这里的b1。所以基本上，批量归一化层有自己的偏置项，前面的层不需要有偏置项，因为那个偏置项无论如何都会被减去。

所以这是另一个需要注意的小细节。有时候它不会造成什么灾难性的后果。这个b1只会变得毫无用处。它永远不会产生任何梯度。它不会学习。它将保持不变，这纯粹是浪费，但实际上不会对其他方面产生任何影响。

好的，我对代码稍作调整并添加了注释，现在想简单总结一下批归一化层的作用。我们使用批归一化来控制神经网络中激活函数的统计特性。通常会在整个神经网络中穿插使用批归一化层，一般会将其放置在含有乘法运算的层之后——比如线性层或卷积层（这些内容我们后续可能会讲到）。

现在，批归一化内部包含增益和偏置的参数，这些参数通过反向传播进行训练。它还有两个缓冲区，即均值和标准差，分别是运行均值和运行标准差。这些参数并不是通过反向传播训练的，而是通过一种类似于运行均值更新的粗糙方法进行训练的。因此，这些可以看作是批归一化层的参数和缓冲区。

然后它实际上做的是计算输入到批量归一化层的激活值的均值和标准差，针对该批次。接着，它将这批数据居中处理，使其成为单位高斯分布，然后通过学习到的偏置和增益进行偏移和缩放。此外，它还会跟踪输入数据的均值和标准差，并维护这些运行中的均值和标准差。

这将在后续推理过程中使用，这样我们就不必一直重新估计均值和标准差。此外，这还使我们能够在测试时逐个处理样本。这就是批归一化层的作用。这是一个相当复杂的层次结构，但这就是它在内部的工作方式。现在我想给大家展示一个真实的例子。你可以搜索ResNet（残差神经网络），这是一种常用于图像分类的神经网络类型。

当然，我们还没有详细讨论ResNets，所以我不会解释它的所有部分。但现在只需注意，图像输入到这里的ResNet顶部，经过许多具有重复结构的层，最终预测图像中的内容。这种重复结构由这些块组成，这些块在这个深度神经网络中依次堆叠。

现在这段代码，这个基本被串联重复使用的块被称为瓶颈块。这里内容很多。这些都是PyTorch的代码，当然我们还没有全部讲解完，但我想指出其中的一些小部分。在初始化部分，我们会对神经网络进行初始化。所以这里的代码块基本上就是我们正在做的事情。我们正在初始化所有的层。

在前向传播中，我们具体定义了神经网络在获得输入后的行为方式。这里的代码大致体现了我们正在实现的功能。现在这些模块被复制并按顺序堆叠起来，这就是残差网络的基本架构。所以请注意这里发生了什么。Conv1，这些都是卷积层。这些卷积层基本上和线性层是一样的，只不过卷积层不适用于——卷积层用于图像，因此它们具有空间结构。

基本上，这种线性乘法和偏置偏移是在图像块上进行的，而不是在整个输入上。因为这些图像具有空间结构，卷积运算本质上就是wx加b，但它们是在输入的重叠块上进行的。除此之外，它仍然是wx加b。然后我们有一个默认初始化为二维批归一化（BatchNorm）的常规层，即一个二维批归一化层。

然后我们有一个像ReLU这样的非线性函数。所以在这里他们用的是ReLU，而我们在这个例子中使用的是tanh。但两者都是非线性函数，你可以相对互换地使用它们。对于非常深的网络，ReLU通常在实际应用中表现稍好一些。可以看到这里重复出现的模式：卷积、批归一化、ReLU、卷积、批归一化、ReLU，如此循环。

然后这里有一个我们尚未讨论的残差连接。但基本上这与我们这里的模式完全相同。我们有一个权重层，比如卷积层或线性层，接着是批归一化，然后是tanh这个非线性激活函数。但基本上是一个权重层、一个归一化层和非线性层。这就是你在创建这些深度神经网络时堆叠的模式。正如这里所做的那样。

还有一点我想让你注意的是，这里当他们初始化卷积层时，比如conv1x1，其深度就在这里。它正在初始化一个nn.conv2d，这是PyTorch中的一个卷积层。这里有一堆关键字参数，我暂时不会解释。但你看这里 bias equals false（偏置设为假）。设置 bias equals false 的原因和我们不使用偏置的情况完全一致。注意到我是如何彻底避免使用偏置的吗？

使用偏置项是多余的，因为在这个权重层之后有一个批归一化（BatchNormalization）。批归一化会减去这个偏置项，然后拥有自己的偏置项。因此没有必要引入这些多余的参数。这不会影响性能。只是没有用。因为他们有卷积和批归一化的主题，所以这里不需要偏置，因为这里面已经有一个偏置了。

顺便说一下，这个例子很容易找到。只需搜索“ResNetPyTorch”，就能看到这个示例。这基本上算是PyTorch中残差神经网络的标准实现。你可以在这里找到相关内容。当然，我还没有涵盖这些部分的许多内容。我还想简单介绍一下这些PyTorch层的定义及其参数。

现在，我们不讨论卷积层，而是来看一下线性层，因为这里我们使用的是线性层。这是一个线性层。我还没有讲到卷积的部分。但正如我提到的，卷积本质上就是作用在图像小块上的线性层。线性层执行的是wx加b的计算，只不过这里他们把w称为转置矩阵。因此它的计算方式wx加b和我们这里所做的非常相似。

要初始化这一层，你需要知道输入单元数（fan_in）和输出单元数（fan_out）。这样才能初始化权重w。这就是所谓的输入单元数和输出单元数，它们决定了权重矩阵的尺寸应该有多大。你还需要传入是否要使用偏置项。如果将其设为false，则该层内不会包含偏置项。在我们的案例中，如果该层后面跟着一个归一化层（如批量归一化层），你可能就需要这样做。

这样你就可以从根本上消除偏差。现在，关于初始化部分，如果我们往下看，这里显示的是这个线性层内部使用的变量。我们的线性层有两个参数：权重和偏置。同样地，它们有一个权重和一个偏置。他们在讨论默认情况下如何初始化这些参数。默认情况下，PyTorch会通过获取输入维度（fan_in）并取其平方根的倒数来初始化权重。

然后，他们使用的是均匀分布，而不是正态分布。所以基本上是一样的。但他们用的是1，而不是5除以3。所以这里没有计算增益。增益仅为1。但除此之外，它正好是输入平方根的倒数，正如我们这里所展示的。因此，1除以k的平方根就是权重的比例。但在绘制数字时，默认情况下他们并没有使用高斯分布。

他们默认使用的是均匀分布。因此，他们从负k的平方根到k的平方根之间均匀抽取。但这与我们在这节课中看到的动机完全相同。他们这样做的原因是，如果你有一个大致高斯分布的输入，这将确保从这一层输出的也是一个大致高斯分布的输出。

你基本上是通过将权重按1除以输入节点数的平方根进行缩放来实现这一点。这就是这个操作的作用。第二件事是批量归一化层。那么让我们看看在PyTorch中这是如何实现的。这里我们有一个一维的批归一化层，正是我们这里所使用的。同时还有一些关键字参数传入其中。

因此我们需要知道特征的数量。对我们来说，这个数字是200。这是必要的，这样我们才能在这里初始化这些参数。增益、偏置以及用于运行均值和标准差的缓冲区。然后他们需要知道这里的epsilon值。默认情况下，这个值是1e-5。通常不需要对此做太多调整。

然后他们需要了解动量。这里所说的动量，正如他们所解释的，基本上用于这些运行平均值和运行标准差。默认情况下，这里的动量是0.1。在这个例子中，我们使用的动量是0.001。基本上，有时候你可能需要调整这个值。

大致来说，如果批量大小非常大，通常你会看到，当你为每个批量大小估算均值和标准差时，如果批量足够大，你得到的结果大致相同。因此，你可以使用稍高的动量，比如0.1。但对于像32这样小的批量大小，这里的均值和标准差可能会略有不同，因为我们只用32个样本来估算均值和标准差。所以数值会有很大的波动。

如果你的动量是0.1，可能不足以让这个值稳定下来并收敛到整个训练集的实际均值和标准差。所以基本上，如果你的批次大小非常小，0.1的动量可能是有风险的。这可能会导致运行均值和标准差在训练过程中波动过大。

但实际上它并没有正确收敛。Affine等于true决定了这个批量归一化层是否具有这些可学习的仿射参数，即增益和偏置。这一项几乎总是保持为true。其实我不太明白为什么要把这个设为false。然后track running stats是决定PyTorch的批量归一化层是否执行这一操作。你可能想跳过运行统计的一个原因是，比如你想在最后阶段把它们作为第二阶段来估算，就像这样。

在这种情况下，你肯定不希望批归一化层进行这些额外却无用的计算。最后，我们还需要知道批归一化层运行在哪个设备上——CPU还是GPU，以及数据类型应该是半精度、单精度、双精度等等。以上就是批归一化层的相关内容。

否则，它们会链接到论文。这是我们实施的相同公式。一切都一样，完全按照我们在这里所做的。好的，以上就是我这次讲座想讲的全部内容。其实，我最想强调的是理解神经网络中激活值和梯度及其统计数据的重要性。这一点变得越来越重要，特别是当...

更大、更广、更深。我们主要观察了输出层的分布情况，发现如果出现两个自信的错误预测，由于最后一层的激活值过于混乱，最终可能导致这种曲棍球杆状的损失曲线。而如果修正这个问题，训练结束时就能获得更优的损失值，因为你的训练过程避免了无效劳动。

然后我们还发现需要控制激活值。我们不希望它们被压缩为零或爆炸到无穷大。正因为如此，神经网络中的所有这些非线性可能会带来很多麻烦。

基本上，你希望整个神经网络中的一切都能保持相当均匀的状态。你希望整个神经网络中的激活大致呈高斯分布。然后我们讨论了，好吧，如果我们希望激活大致呈高斯分布，那么在神经网络初始化时，我们该如何调整这些权重矩阵和偏置，以便我们不会出现，你知道的，一切都尽可能受控的情况。

这给了我们很大的推动和改进。然后我谈到了这个策略对于更深层次的神经网络实际上是不可行的。因为当你拥有更深层次的神经网络，包含许多不同类型的层时，要精确设置权重和偏置，使得整个神经网络中的激活大致均匀，变得非常非常困难。

于是，我引入了归一化层的概念。现在，在实践中人们会使用多种归一化层，包括批量归一化、层归一化、实例归一化和组归一化。

我们还没有涉及大部分内容，但我已经介绍了第一个。还有我认为最早出现的那个，叫做批量归一化。我们了解了批量归一化的工作原理。这是一个可以分散应用于深度神经网络各层的技术。其核心理念是：如果你想获得近似高斯分布的激活值，只需对激活值计算均值和标准差，然后进行数据中心化处理。这种操作之所以可行，是因为中心化过程本身是可微分的。

除此之外，我们实际上还不得不添加很多花哨的功能。这让你对批量归一化层的复杂性有了一定的了解。因为现在我们正在对数据进行中心化处理，这很好。

但突然之间，我们需要增益和偏置。现在这些参数变得可训练了。接着，由于我们将所有训练样本耦合在一起，问题突然变成了：如何进行推理？在推理阶段，我们需要在整个训练集上一次性估算这些均值和标准差，然后在推理时使用这些值。

但随后没人喜欢做第二阶段。因此，在训练期间，我们将所有内容都整合到批量归一化层中，并尝试以运行的方式估算这些内容，以使一切变得更简单。这就形成了批量归一化层。正如我所说，没有人喜欢这一层。它会导致大量的错误。直观地说，这是因为它在神经网络的前向传播过程中耦合了示例。

我这一生中曾多次因这一层而自食其果。我不希望你也遭受同样的痛苦。所以，基本上要尽量避免它。这些层的其他替代方案包括组归一化（group normalization）或层归一化（layer normalization）等。这些方法在近年来的深度学习中变得越来越常见。不过我们目前还没有讲到这些内容。

但毫无疑问，批量归一化在2015年左右问世时极具影响力，因为这是首次能够可靠地训练更深层次的神经网络。从根本上说，原因在于该层在控制神经网络激活统计量方面非常有效。这就是迄今为止的情况。

以上就是我想讲的全部内容。希望在未来的课程中，我们可以开始探讨循环神经网络。循环神经网络，正如我们将要看到的，实际上是非常非常深的网络，因为当你优化这些神经网络时，你会展开循环。这就是为什么关于激活统计数据和所有这些归一化层的分析对于良好性能变得非常非常重要的原因。我们下次再见。拜拜。

好吧，我撒谎了。我想让我们再做一个总结作为奖励。我认为再对我在这节课中讲的所有内容做一个总结是有用的。此外，我希望我们先对代码进行一些"火炬化"改造，让它看起来更像你在PyTorch中会遇到的样子。你会看到我将把代码组织成这些模块，比如线性模块和批归一化模块。我把代码放在这些模块里，这样我们就能像在PyTorch中那样构建神经网络了。

我将详细讲解这一过程。我们会先创建神经网络，然后像之前那样进行优化循环。还有一件事我想在这里做的是，我想看看激活统计数据，包括前向传播和反向传播。然后这里我们有评估和采样，和之前一样。所以让我倒回到最上面，稍微放慢一点。

所以我在这里创建了一个线性层。你会发现torch.nn中有许多不同类型的层，其中之一就是线性层。Torch.nn.linear函数接收输入特征数量、输出特征数量、是否包含偏置项、以及该层要放置的设备与数据类型等参数。我将省略后两个参数，但其他部分完全相同。我们需要指定fan-in（即输入数量）、fan-out（输出数量）以及是否使用偏置项。

在这一层内部，有一个权重和一个偏置项，如果你愿意这么称呼的话。通常的做法是使用比如从高斯分布中抽取的随机数来初始化权重。然后这里就是我们在这节课中已经讨论过的初始化方法。

这是一个很好的默认设置，也是我认为PyTorch采用的默认方式。默认情况下，偏置项通常初始化为零。现在，当你调用这个模块时，它基本上会计算w乘以x加上b（如果你有偏置项的话）。然后，当你在这个模块上调用.parameters时，它会返回作为该层参数的张量。接下来，我们还有批归一化层。我已经在这里写好了。

这与 PyTorch 的 nn.batchnorm1d 层非常相似，如图所示。在这里，我主要采用了这三个参数：维度、我们在除法中使用的 epsilon 值，以及用于跟踪这些运行统计量（即运行均值和运行方差）的动量。实际上，PyTorch 还接受更多参数，但我这里假设了一些默认设置。

因此，对我们来说，仿射（affine）将为真。这意味着我们将在归一化后使用伽马（gamma）和贝塔（beta）。跟踪运行统计（track running stats）将为真。因此，我们将跟踪批量归一化中的运行均值和运行方差。默认情况下，我们的设备是CPU，默认数据类型是float32。这些就是默认设置。

否则，我们在这个批归一化层中采用所有相同的参数。首先，我只是将它们保存下来。现在，这里有一些新的东西。默认情况下，.training 属性为 true。PyTorch 的 nn 模块也具有这个 .training 属性。这是因为许多模块（包括批归一化模块）会根据神经网络处于训练模式还是评估模式表现出不同行为——无论你是在训练神经网络，还是在评估模式下运行它来计算评估损失，或是在一些测试样本上进行推理。批归一化就是典型例子：在训练时，我们会使用当前批次估算得到的均值和方差。

但在推理过程中，我们使用的是运行均值和运行方差。同样，如果我们正在进行训练，我们会更新均值和方差。但如果我们在测试阶段，这些值就不会被更新。它们被固定不变。因此，这个标志是必要的，默认情况下为真，就像在PyTorch中一样。现在，一维批归一化的参数就是这里的gamma和beta。

然后，在PyTorch术语中，运行均值和运行方差被称为缓冲区。这些缓冲区在这里明确地使用指数移动平均进行训练。它们不是随机梯度下降反向传播的一部分。因此，它们并不是该层的参数。这就是为什么当我们在这里有参数时，我们只返回gamma和beta。我们不会返回均值和方差。

这是在内部进行的一种训练方式，每次前向传播都使用指数移动平均。这就是初始化的过程。现在，在前向传播中，如果我们正在进行训练，那么我们就使用由批次估计的均值和方差。让我把论文调出来。我们计算了均值和方差。刚才在上面，我是在估计标准差，并且在这里的运行标准差中跟踪标准差，而不是运行方差。

但让我们严格按照论文来。在这里，他们计算的是方差，也就是标准差的平方。而运行方差中跟踪的就是这个，而不是运行标准差。但我相信这两者会非常非常相似。如果我们不进行训练，就会使用运行均值和方差进行归一化。

然后在这里，我正在计算这一层的输出。同时，我将其赋值给一个名为dot out的属性。现在，dot out是我在这些模块中使用的一个东西。这不是你在PyTorch中会看到的内容。我们稍微偏离了它。我创建一个点是因为我想非常容易地维护所有这些变量，以便我们可以对它们进行统计并绘制图表。

但PyTorch和模块不会有dot out属性。最后，在这里，我们再次使用指数移动平均来更新缓冲区，正如我之前提到的，根据提供的动量值。重要的是，你会注意到我正在使用torch.nograd上下文管理器。我这样做是因为如果我们不使用这个，PyTorch就会开始为这些张量构建完整的计算图，因为它预期我们最终会调用dot backward。但我们永远不会在任何包含运行均值和运行方差的内容上调用dot backward。这就是为什么我们需要使用这个上下文管理器，这样我们就不会维护和使用所有这些额外的内存。

这样会让效率更高。而且它只是告诉PyTorch不需要反向传播。我们只有一堆张量。我们想更新它们。仅此而已。然后我们就回来。好的，现在往下滚动，我们来看10H层。它与torch.10H非常非常相似。它的功能不多，正如你所料，它只是计算10H。

这就是 torch.10H。这一层没有参数。但因为这些都是层，现在可以非常容易地将它们堆叠起来，基本上就是一个列表。我们可以进行所有我们习惯的初始化操作。所以我们有了最初的嵌入矩阵。我们有我们的层，可以按顺序调用它们。然后再次使用torch.nograd，这里有一些初始化操作。

所以我们想让输出的softmax不那么自信，就像我们之前看到的那样。除此之外，因为我们在这里使用了一个六层的多层感知器，你可以看到我是如何堆叠线性层、10H、线性层、10H等等的，我将在这里使用这个游戏。我马上就会来玩这个。

所以你会看到当我们改变这个时，统计数据会发生什么变化。最后，参数基本上是嵌入矩阵和所有层中的所有参数。注意这里，我用了双重列表推导式，如果你想这么称呼它的话。


But for every layer in layers, and for every parameter in each of those layers, we are just stacking up all those parameters. Now, in total, we have 46,000 parameters. And I'm telling PyTorch that all of them require gradient.

Then here, we have everything here we are actually mostly used to. We are sampling batch. We are doing forward pass.

The forward pass now is just a linear application of all the layers in order, followed by the cross-entropy. And then in the backward pass, you'll notice that for every single layer, I now iterate over all the outputs. And I'm telling PyTorch to retain the gradient of them.

And then here, we are already used to all the gradients set to none, do the backward to fill in the gradients, do an update using stochastic gradient send, and then track some statistics. And then I am going to break after a single iteration. Now, here in this cell, in this diagram, I'm visualizing the histograms of the four pass activations.

And I'm specifically doing it at the tanh layers. So iterating over all the layers, except for the very last one, which is basically just the softmax layer. If it is a tanh layer, and I'm using a tanh layer just because they have a finite output, negative 1 to 1. And so it's very easy to visualize here.

So you see negative 1 to 1, and it's a finite range and easy to work with. I take the out tensor from that layer into t, and then I'm calculating the mean, the standard deviation, the percent saturation of t. And the way I define the percent saturation is that t dot absolute value is greater than 0.97. So that means we are here at the tails of the tanh. And remember that when we are in the tails of the tanh, that will actually stop gradients.

So we don't want this to be too high. Now, here I'm calling torch dot histogram. And then I am plotting this histogram.

So basically, what this is doing is that every different type of layer, and they all have a different color, we are looking at how many values in these tensors take on any of the values below on this axis here. So the first layer is fairly saturated here at 20%. So you can see that it's got tails here.

But then everything sort of stabilizes. And if we had more layers here, it would actually just stabilize at around the standard deviation of about 0.65. And the saturation would be roughly 5%. And the reason that this stabilizes and gives us a nice distribution here is because gain is set to 5 over 3. Now, here, this gain, you see that by default, we initialize with 1 over square root of fan in.

But then here during initialization, I come in and I iterate over all the layers. And if it's a linear layer, I boost that by the gain. Now, we saw that 1. So basically, if we just do not use a gain, then what happens? If I redraw this, you will see that the standard deviation is shrinking.

And the saturation is coming to 0. And basically, what's happening is the first layer is pretty decent. But then further layers are just shrinking down to 0. And it's happening slowly, but it's shrinking to 0. And the reason for that is when you just have a sandwich of linear layers alone, then initializing our weights in this manner, we saw previously, would have conserved the standard deviation of 1. But because we have this interspersed tanh layers in there, these tanh layers are squashing functions. And so they take your distribution and they slightly squash it.

And so some gain is necessary to keep expanding it to fight the squashing. So it just turns out that 5 over 3 is a good value. So if we have something too small like 1, we saw that things will come towards 0. But if it's something too high, let's do 2. Then here we see that... Well, let me do something a bit more extreme because... So it's a bit more visible.

Let's try 3. Okay, so we see here that the saturations are trying to be way too large. Okay, so 3 would create way too saturated activations. So 5 over 3 is a good setting for a sandwich of linear layers with tanh activations.

And it roughly stabilizes the standard deviation at a reasonable point. Now, honestly, I have no idea where 5 over 3 came from in PyTorch when we were looking at the coming initialization. I see empirically that it stabilizes this sandwich of linear and tanh and that the saturation is in a good range.

But I don't actually know if this came out of some math formula. I tried searching briefly for where this comes from, but I wasn't able to find anything. But certainly we see that empirically these are very nice ranges.

Our saturation is roughly 5%, which is a pretty good number. And this is a good setting of the gain in this context. Similarly, we can do the exact same thing with the gradients.

So here is a very same loop if it's a tanh, but instead of taking the layer dot out, I'm taking the grad. And then I'm also showing the mean and the standard deviation. And I'm plotting the histogram of these values.

And so you'll see that the gradient distribution is fairly reasonable. And in particular, what we're looking for is that all the different layers in this sandwich has roughly the same gradient. Things are not shrinking or exploding.

So we can, for example, come here and we can take a look at what happens if this gain was way too small. So this was 0.5. Then you see the, first of all, the activations are shrinking to zero, but also the gradients are doing something weird. The gradient started off here, and then now they're like expanding out.

And similarly, if we, for example, have a too high of a gain, so like 3, then we see that also the gradients have, there's some asymmetry going on where as you go into deeper and deeper layers, the activations are also changing. And so that's not what we want. And in this case, we saw that without the use of batch norm, as we are going through right now, we have to very carefully set those gains to get nice activations in both the forward pass and the backward pass.

Now, before we move on to batch normalization, I would also like to take a look at what happens when we have no tanh units here. So erasing all the tanh nonlinearities, but keeping the gain at 5 over 3, we now have just a giant linear sandwich. So let's see what happens to the activations.

As we saw before, the correct gain here is 1. That is the standard deviation preserving gain. So 1.667 is too high. And so what's going to happen now is the following.

I have to change this to be linear. So we are, because there's no more tanh layers. And let me change this to linear as well.

So what we're seeing is the activations started out on the blue and have, by layer 4, become very diffuse. So what's happening to the activations is this. And with the gradients on the top layer, the activation, the gradient statistics are the purple.

And then they diminish as you go down deeper in the layers. And so basically, you have an asymmetry in the neural net. And you might imagine that if you have very deep neural networks, say like 50 layers or something like that, this is not a good place to be.

So that's why before best normalization, this was incredibly tricky to set. In particular, if this is too large of a gain, this happens. And if it's too little of a gain, then this happens.

So the opposite of that basically happens. Here we have a shrinking and a diffusion, depending on which direction you look at it from. And so certainly, this is not what you want.

And in this case, the correct setting of the gain is exactly 1, just like we're doing at initialization. And then we see that the statistics for the forward and the backward paths are well-behaved. And so the reason I want to show you this is that basically, getting neural nets to train before these normalization layers and before the use of advanced optimizers like Adam, which we still have to cover, and residual connections and so on, training neural nets basically looked like this.

It's like a total balancing act. You have to make sure that everything is precisely orchestrated, and you have to care about the activations and the gradients and their statistics. And then maybe you can train something.

But it was basically impossible to train very deep networks. And this is fundamentally the reason for that. You'd have to be very, very careful with your initialization.

The other point here is, you might be asking yourself, I'm not sure if I covered this, why do we need these 10H layers at all? Why do we include them and then have to worry about the gain? And the reason for that, of course, is that if you just have a stack of linear layers, then certainly we're getting very easily nice activations and so on. But this is just a massive linear sandwich. And it turns out that it collapses to a single linear layer in terms of its representation power.

So if you were to plot the output as a function of the input, you're just getting a linear function. No matter how many linear layers you stack up, you still just end up with a linear transformation. All the wx plus b's just collapse into a large wx plus b with slightly different w's and slightly different b. But interestingly, even though the forward pass collapses to just a linear layer, because of backpropagation and the dynamics of the backward pass, the optimization actually is not identical.

You actually end up with all kinds of interesting dynamics in the backward pass because of the way the chain rule is calculating it. And so optimizing a linear layer by itself and optimizing a sandwich of 10 linear layers, in both cases, those are just a linear transformation in the forward pass, but the training dynamics would be different. And there's entire papers that analyze, in fact, like infinitely layered linear layers and so on.

And so there's a lot of things that you can play with there. But basically, the 10-H nonlinearities allow us to turn this sandwich from just a linear function into a neural network that can, in principle, approximate any arbitrary function. Okay, so now I've reset the code to use the linear 10-H sandwich like before, and I've reset everything.

So the gain is 5 over 3. We can run a single step of optimization, and we can look at the activation statistics of the forward pass and the backward pass. But I've added one more plot here that I think is really important to look at when you're training your neural nets and to consider. And ultimately, what we're doing is we're updating the parameters of the neural net.

So we care about the parameters and their values and their gradients. So here, what I'm doing is I'm actually iterating over all the parameters available, and then I'm only restricting it to the two-dimensional parameters, which are basically the weights of these linear layers. And I'm skipping the biases, and I'm skipping the gammas and the betas just for simplicity.

But you can also take a look at those as well. But what's happening with the weights is instructive by itself. So here we have all the different weights, their shapes.

So this is the embedding layer, the first linear layer, all the way to the very last linear layer. And then we have the mean, the standard deviation of all these parameters. The histogram, and you can see that it actually doesn't look that amazing.

So there's some trouble in paradise. Even though these gradients look okay, there's something weird going on here. I'll get to that in a second.

And the last thing here is the gradient-to-data ratio. So sometimes I like to visualize this as well because what this gives you a sense of is what is the scale of the gradient compared to the scale of the actual values? And this is important because we're going to end up taking a step update that is the learning rate times the gradient onto the data. And so if the gradient has too large of a magnitude, if the numbers in there are too large compared to the numbers in data, then you'd be in trouble.

But in this case, the gradient-to-data is our low numbers. So the values inside grad are 1000 times smaller than the values inside data in these weights, most of them. Now, notably, that is not true about the last layer.

And so the last layer actually here, the output layer, is a bit of a troublemaker in the way that this is currently arranged. Because you can see that the last layer here in pink takes on values that are much larger than some of the values inside the neural net. So the standard deviations are roughly 1 and negative 3 throughout, except for the last layer, which actually has roughly 1 and negative 2 standard deviation of gradients.

And so the gradients on the last layer are currently about 100 times greater, sorry, 10 times greater than all the other weights inside the neural net. And so that's problematic because in the simple stochastic gradient descent setup, you would be training this last layer about 10 times faster than you would be training the other layers at initialization. Now, this actually kind of fixes itself a little bit if you train for a bit longer.

So for example, if i greater than 1000, only then do a break. Let me reinitialize. And then let me do it 1000 steps.

And after 1000 steps, we can look at the forward pass. Okay, so you see how the neurons are saturating a bit. And we can also look at the backward pass.

But otherwise, they look good. They're about equal, and there's no shrinking to zero or exploding to infinities. And you can see that here in the weights, things are also stabilizing a little bit.

So the tails of the last pink layer are actually coming in during the optimization. But certainly, this is a little bit troubling, especially if you are using a very simple update rule, like stochastic gradient descent, instead of a modern optimizer like Adam. Now I'd like to show you one more plot that I usually look at when I train neural networks.

And basically, the gradient to data ratio is not actually that informative, because what matters at the end is not the gradient to data ratio, but the update to the data ratio, because that is the amount by which we will actually change the data in these tensors. So coming up here, what I'd like to do is I'd like to introduce a new update to data ratio. It's going to be list, and we're going to build it out every single iteration.

And here, I'd like to keep track of basically the ratio every single iteration. So without any gradients, I'm comparing the update, which is learning rate times the gradient. That is the update that we're going to apply to every parameter.

So see, I'm iterating over all the parameters. And then I'm taking the basically standard deviation of the update we're going to apply, and divide it by the actual content, the data of that parameter and its standard deviation. So this is the ratio of basically how great are the updates to the values in these tensors.

Then we're going to take a log of it. And actually, I'd like to take a log 10, just so it's a nicer visualization. So we're going to be basically looking at the exponents of this division here.

And then that item to pop out the float. And we're going to be keeping track of this for all the parameters and adding it to this UD tensor. So now let me reinitialize and run 1000 iterations.

We can look at the activations, the gradients, and the parameter gradients as we did before. But now I have one more plot here to introduce. And what's happening here is we're iterating over all the parameters.

And I'm constraining it again, like I did here to just the weights. So the number of dimensions in these sensors is two. And then I'm basically plotting all of these update ratios over time.

So when I plot this, I plot those ratios, and you can see that they evolve over time during initialization to take on certain values. And then these updates are like start stabilizing, usually during training. Then the other thing that I'm plotting here is I'm plotting here like an approximate value that is a rough guide for what it roughly should be.

And it should be like roughly 1 and negative 3. And so that means that basically there's some values in this tensor, and they take on certain values. And the updates to them at every single iteration are no more than roughly 1,000th of the actual magnitude in those tensors. If this was much larger, like for example, if the log of this was like say negative 1, this is actually updating those values quite a lot.

They're undergoing a lot of change. But the reason that the final layer here is an outlier is because this layer was artificially shrunk down to keep the softmax unconfident. So here, you see how we multiply the weight by 0.1 in the initialization to make the last layer prediction less confident? That artificially made the values inside that tensor way too low, and that's why we're getting temporarily a very high ratio.

But you see that that stabilizes over time once that weight starts to learn. But basically, I like to look at the evolution of this update ratio for all my parameters, usually. And I like to make sure that it's not too much above 1 and negative 3, roughly.

So around negative 3 on this log plot. If it's below negative 3, usually that means that the parameters are not training fast enough. So if our learning rate was very low, let's do that experiment.

Let's initialize. And then let's actually do a learning rate of say 1 and negative 3 here. So 0.001. If your learning rate is way too low, this plot will typically reveal it.

So you see how all of these updates are way too small. So the size of the update is basically 10,000 times in magnitude to the size of the numbers in that tensor in the first place. So this is a symptom of training way too slow.

So this is another way to sometimes set the learning rate and to get a sense of what that learning rate should be. And ultimately, this is something that you would keep track of.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

A little bit on the higher side, because you see that we're above the black line of negative 3, we're somewhere around negative 2.5, it's like okay, but everything is like somewhat stabilizing, and so this looks like a pretty decent setting of learning rates and so on. But this is something to look at, and when things are miscalibrated you will see very quickly. So for example, everything looks pretty well behaved, right? But just as a comparison, when things are not properly calibrated, what does that look like? Let me come up here, and let's say that for example, what do we do? Let's say that we forgot to apply this fan-in normalization, so the weights inside the linear layers are just a sample from a Gaussian in all the stages. 

What happens to our, how do we notice that something's off? Well, the activation plot will tell you, whoa, your neurons are way too saturated, the gradients are going to be all messed up, the histogram for these weights are going to be all messed up as well, and there's a lot of asymmetry. And then if we look here, I suspect it's all going to be also pretty messed up. So you see there's a lot of discrepancy in how fast these layers are learning, and some of them are learning way too fast. 

So negative 1, negative 1.5, those are very large numbers in terms of this ratio. Again, you should be somewhere around negative 3 and not much more about that. So this is how miscalibrations of your neural nets are going to manifest, and these kinds of plots here are a good way of bringing those miscalibrations to your attention, and so you can address them.

Okay, so far we've seen that when we have this linear tanh sandwich, we can actually precisely calibrate the gains and make the activations, the gradients, and the parameters, and the updates all look pretty decent. But it definitely feels a little bit like balancing of a pencil on your finger, and that's because this gain has to be very precisely calibrated. So now let's introduce batch normalization layers into the mix, and let's see how that helps fix the problem.

So here I'm going to take the BatchNormalization1D class, and I'm going to start placing it inside. And as I mentioned before, the standard typical place you would place it is between the linear layer, so right after it, but before the non-linearity. But people have definitely played with that, and in fact you can get very similar results even if you place it after the non-linearity. 

And the other thing that I wanted to mention is it's totally fine to also place it at the end, after the last linear layer and before the loss function. So this is potentially fine as well. And in this case, this would be output, would be vocab size. 

Now because the last layer is BatchNorm, we would not be changing the weight to make the softmax less confident, we'd be changing the gamma. Because gamma, remember, in the BatchNorm is the variable that multiplicatively interacts with the output of that normalization. So we can initialize this sandwich now, we can train, and we can see that the activations are going to of course look very good, and they are going to necessarily look good, because now before every single tanh layer, there is a normalization in the BatchNorm. 

So this is, unsurprisingly, all looks pretty good. It's going to be standard deviation of roughly 0.65, 2%, and roughly equal standard deviation throughout the entire layers. So everything looks very homogeneous. 

The gradients look good, the weights look good in their distributions, and then the updates also look pretty reasonable. We're going above negative 3 a little bit, but not by too much. So all the parameters are training at roughly the same rate here.

But now what we've gained is, we are going to be slightly less brittle with respect to the gain of these. So for example, I can make the gain be, say, 0.2 here, which was much, much slower than what we had with the tanh, but as we'll see, the activations will actually be exactly unaffected, and that's because of, again, this explicit normalization. The gradients are going to look okay, the weight gradients are going to look okay, but actually the updates will change.

And so even though the forward and backward paths to a very large extent look okay, because of the backward paths of the BatchNorm and how the scale of the incoming activations interacts in the BatchNorm and its backward paths, this is actually changing the scale of the updates on these parameters. So the gradients of these weights are affected. So we still don't get a completely free path to pass in arbitrary weights here, but everything else is significantly more robust in terms of the forward, backward, and the weight gradients. 

It's just that you may have to retune your learning rate if you are changing sufficiently the scale of the activations that are coming into the BatchNorms. So here, for example, we changed the gains of these linear layers to be greater, and we're seeing that the updates are coming out lower as a result. And then finally, we can also, if we are using BatchNorms, we don't actually need to necessarily, let me reset this to one so there's no gain, we don't necessarily even have to normalize by fan-in sometimes. 

So if I take out the fan-in, so these are just now random Gaussian, we'll see that because of BatchNorm, this will actually be relatively well behaved. So this look, of course, in the forward path look good, the gradients look good, the weight updates look okay, a little bit of fat tails on some of the layers, and this looks okay as well. But as you can see, we're significantly below negative three, so we'd have to bump up the learning rate of this BatchNorm so that we are training more properly.

And in particular, looking at this, roughly looks like we have to 10x the learning rate to get to about 1e negative three. So we'd come here, and we would change this to be update of 1.0. And if I reinitialize, then we'll see that everything still, of course, looks good. And now we are roughly here, and we expect this to be an okay training run.

So long story short, we are significantly more robust to the gain of these linear layers, whether or not we have to apply the fan-in. And then we can change the gain, but we actually do have to worry a little bit about the update scales and making sure that the learning rate is properly calibrated here. But the activations of the forward, backward paths and the updates are looking significantly more well behaved, except for the global scale that is potentially being adjusted here. 

Okay, so now let me summarize. There are three things I was hoping to achieve with this section. Number one, I wanted to introduce you to BatchNormalization, which is one of the first modern innovations that we're looking into that helped stabilize very deep neural networks and their training. 

And I hope you understand how the BatchNormalization works and how it would be used in a neural network. Number two, I was hoping to PyTorchify some of our code and wrap it up into these modules. So like Linear, BatchNorm 1D, 10H, etc. 

These are layers or modules, and they can be stacked up into neural nets like Lego building blocks. And these layers actually exist in PyTorch. And if you import TorchNN, then you can actually, the way I've constructed it, you can simply just use PyTorch by prepending NN. 

to all these different layers. And actually everything will just work, because the API that I've developed here is identical to the API that PyTorch uses. And the implementation also is basically, as far as I'm aware, identical to the one in PyTorch. 

And number three, I tried to introduce you to the diagnostic tools that you would use to understand whether your neural network is in a good state dynamically. So we are looking at the statistics and histograms and activation of the forward pass activations, the backward pass gradients. And then also we're looking at the weights that are going to be updated as part of stochastic gradient ascent. 

And we're looking at their means, standard deviations, and also the ratio of gradients to data, or even better, the updates to data. And we saw that typically we don't actually look at it as a single snapshot frozen in time at some particular iteration. Typically people look at this as over time, just like I've done here. 

And they look at these update to data ratios, and they make sure everything looks okay. And in particular, I said that 1 in negative 3, or basically negative 3 on the log scale, is a good rough heuristic for what you want this ratio to be. And if it's way too high, then probably the learning rate or the updates are a little too big. 

And if it's way too small, then the learning rate is probably too small. So that's just some of the things that you may want to play with when you try to get your neural network to work very well. There's a number of things I did not try to achieve. 

I did not try to beat our previous performance, as an example, by introducing the BatchNorm layer. Actually, I did try, and I found that I used the learning rate finding mechanism that I've described before. I tried to train the BatchNorm layer, BatchNorm neural net, and I actually ended up with results that are very, very similar to what we've obtained before. 

And that's because our performance now is not bottlenecked by the optimization, which is what BatchNorm is helping with. The performance at this stage is bottlenecked by what I suspect is the context length of our context. So currently, we are taking three characters to predict the fourth one, and I think we need to go beyond that. 

And we need to look at more powerful architectures, like recurrent neural networks and transformers, in order to further push the like probabilities that we're achieving on this dataset. And I also did not try to have a full explanation of all of these activations, the gradients, and the backward pass, and the statistics of all these gradients. And so you may have found some of the parts here unintuitive, and maybe you're slightly confused about, okay, if I change the gain here, how come that we need a different learning rate? And I didn't go into the full detail, because you'd have to actually look at the backward pass of all these different layers and get an intuitive understanding of how all that works. 

And I did not go into that in this lecture. The purpose really was just to introduce you to the diagnostic tools and what they look like, but there's still a lot of work remaining on the intuitive level to understand the initialization, the backward pass, and how all of that interacts. But you shouldn't feel too bad, because honestly, we are getting to the cutting edge of where the field is. 

We certainly haven't, I would say, solved initialization, and we haven't solved backpropagation. And these are still very much an active area of research. People are still trying to figure out what is the best way to initialize these networks, what is the best update rule to use, and so on. 

So none of this is really solved, and we don't really have all the answers to all these cases. But at least we're making progress, and at least we have some tools to tell us whether or not things are on the right track for now. So I think we've made positive progress in this lecture, and I hope you enjoyed that, and I will see you next time. 

Hi everyone. So today we are once again continuing our implementation of NACOR. Now so far we've come up to here, Montalio perceptrons, and our neural net looked like this, and we were implementing this over the last few lectures. 

Now I'm sure everyone is very excited to go into recurrent neural networks and all of their variants, and how they work, and the diagrams look cool, and it's very exciting and interesting, and we're going to get a better result. But unfortunately I think we have to remain here for one more lecture. And the reason for that is we've already trained this Montalio perceptron, right, and we are getting pretty good loss, and I think we have a pretty decent understanding of the architecture and how it works.

But the line of code here that I take an issue with is here, loss dot backward. That is, we are taking PyTorch autograd and using it to calculate all of our gradients along the way. And I would like to remove the use of loss dot backward, and I would like us to write our backward pass manually on the level of tensors. 

And I think that this is a very useful exercise for the following reasons. I actually have an entire blog post on this topic, but I like to call backpropagation a leaky abstraction. And what I mean by that is backpropagation doesn't just make your neural networks just work magically. 

It's not the case that you can just stack up arbitrary Lego blocks of differentiable functions and just cross your fingers and backpropagate and everything is great. Things don't just work automatically. It is a leaky abstraction in the sense that you can shoot yourself in the foot if you do not understand its internals. 

It will magically not work or not work optimally, and you will need to understand how it works under the hood if you're hoping to debug it and if you are hoping to address it in your neural net. So this blog post here from a while ago goes into some of those examples. So for example, we've already covered them, some of them already. 

For example, the flat tails of these functions and how you do not want to saturate them too much because your gradients will die. The case of dead neurons, which I've already covered as well. The case of exploding or vanishing gradients in the case of recurrent neural networks, which we are about to cover. 

And then also you will often come across some examples in the wild. This is a snippet that I found in a random code base on the internet, where they actually have a very subtle but pretty major bug in their implementation. And the bug points at the fact that the author of this code does not actually understand backpropagation. 

So what they're trying to do here is they're trying to clip the loss at a certain maximum value. But actually what they're trying to do is they're trying to clip the gradients to have a maximum value instead of trying to clip the loss at a maximum value. And indirectly, they're basically causing some of the outliers to be actually ignored. 

Because when you clip a loss of an outlier, you are setting its gradient to zero. And so have a look through this and read through it. But there's basically a bunch of subtle issues that you're going to avoid if you actually know what you're doing. 

And that's why I don't think it's the case that because PyTorch or other frameworks offer autograd, it is okay for us to ignore how it works. Now, we've actually already covered autograd and we wrote micrograd. But micrograd was an autograd engine only on the level of individual scalars. 

So the atoms were single individual numbers. And I don't think it's enough. And I'd like us to basically think about backpropagation on the level of tensors as well. 

And so in a summary, I think it's a good exercise. I think it is very, very valuable. You're going to become better at debugging neural networks and making sure that you understand what you're doing. 

It is going to make everything fully explicit. So you're not going to be nervous about what is hidden away from you. And basically, in general, we're going to emerge stronger.

And so let's get into it. A bit of a fun historical note here is that today, writing your backward pass by hand and manually is not recommended. And no one does it except for the purposes of exercise.

But about 10 years ago in deep learning, this was fairly standard and, in fact, pervasive. So at the time, everyone used to write their own backward pass by hand manually, including myself. And it's just what you would do. 

So we used to write backward pass by hand. And now everyone just calls lost backward. We've lost something. 

I want to give you a few examples of this. So here's a 2006 paper from Geoff Hinton and Ruslan Slavdinov in science that was influential at the time. And this was training some architectures called restricted Boltzmann machines. 

And basically, it's an autoencoder trained here. And this is from roughly 2010. I had a library for training restricted Boltzmann machines. 

And this was at the time written in MATLAB. So Python was not used for deep learning pervasively. It was all MATLAB. 

And MATLAB was this scientific computing package that everyone would use. So we would write MATLAB, which is a programming language as well. But it had a very convenient tensor class. 

And it was this computing environment. And you would run here. It would all run on the CPU, of course. 

But you would have very nice plots to go with it and a built-in debugger. And it was pretty nice. Now, the code in this package in 2010 that I wrote for fitting restricted Boltzmann machines to a large extent is recognizable. 

But I wanted to show you how you would, well, I'm creating the data and the xy batches. I'm initializing the neural net. So it's got weights and biases just like we're used to. 

And then this is the training loop where we actually do the forward pass. And then here, at this time, didn't even necessarily use backpropagation to train neural networks. So this, in particular, implements contrastive divergence, which estimates a gradient. 

And then here, we take that gradient and use it for a parameter update along the lines we're used to. Yeah, here. But you can see that basically people are meddling with these gradients directly and inline and themselves. 

It wasn't that common to use an autograd engine. Here's one more example from a paper of mine from 2014 called the fragment embeddings. And here, what I was doing is I was aligning images and text. 

And so it's kind of like a clip, if you're familiar with it. But instead of working on the level of entire images and entire sentences, it was working on the level of individual objects and little pieces of sentences. And I was embedding them and then calculating very much like a clip-like loss.

And I dug up the code from 2014 of how I implemented this. And it was already in NumPy and Python. And here, I'm implementing the cost function. 

And it was standard to implement not just the cost, but also the backward pass manually. So here, I'm calculating the image embeddings, sentence embeddings, the loss function. I calculate the scores. 

This is the loss function. And then once I have the loss function, I do the backward pass right here. So I backward through the loss function and through the neural net. 

And I append regularization. So everything was done by hand manually. And you would just write out the backward pass.

And then you would use a gradient checker to make sure that your numerical estimate of the gradient agrees with the one you calculated during the backpropagation. So this was very standard for a long time. But today, of course, it is standard to use an autograd engine. 

But it was definitely useful. And I think people understood how these neural networks work on a very intuitive level. And so I think it's a good exercise again.

And this is where we want to be. So just as a reminder from our previous lecture, this is the Jupyter Notebook that we implemented at the time. And we're going to keep everything the same. 

So we're still going to have a two-layer multilayer perceptron with a batch normalization layer. So the forward pass will be basically identical to this lecture. But here, we're going to get rid of loss.backward. And instead, we're going to write the backward pass manually. 

Now here's the starter code for this lecture. We are becoming a backprop ninja in this notebook. And the first few cells here are identical to what we are used to.

So we are doing some imports, loading the data set, and processing the data set. None of this changed. Now here, I'm introducing a utility function that we're going to use later to compare the gradients. 

So in particular, we are going to have the gradients that we estimate manually ourselves. And we're going to have gradients that PyTorch calculates. And we're going to be checking for correctness, assuming, of course, that PyTorch is correct.

Then here, we have the initialization that we are quite used to. So we have our embedding table for the characters, the first layer, second layer, and a batch normalization in between. And here's where we create all the parameters. 

Now, you will note that I changed the initialization a little bit to be small numbers. So normally, you would set the biases to be all 0. Here, I am setting them to be small random numbers. And I'm doing this because if your variables are initialized to exactly 0, sometimes what can happen is that can mask an incorrect implementation of a gradient. 

Because when everything is 0, it simplifies and gives you a much simpler expression of the gradient than you would otherwise get. And so by making it small numbers, I'm trying to unmask those potential errors in these calculations. You'll also notice that I'm using b1 in the first layer. 

I'm using a bias despite batch normalization right afterwards. So this would typically not be what you do because we talked about the fact that you don't need a bias. But I'm doing this here just for fun because we're going to have a gradient with respect to it. 

And we can check that we are still calculating it correctly even though this bias is spurious. So here, I'm calculating a single batch. And then here, I am doing a forward pass. 

Now, you'll notice that the forward pass is significantly expanded from what we are used to. Here, the forward pass was just here. Now, the reason that the forward pass is longer is for two reasons. 

Number one, here, we just had an f dot cross entropy. But here, I am bringing back a explicit implementation of the loss function. And number two, I've broken up the implementation into manageable chunks. 

So we have a lot more intermediate tensors along the way in the forward pass. And that's because we are about to go backwards and calculate the gradients in this backpropagation from the bottom to the top. So we're going to go upwards. 

And just like we have, for example, the logprops tensor in a forward pass, in a backward pass, we're going to have a dlogprops, which is going to store the derivative of the loss with respect to the logprops tensor. And so we're going to be prepending d to every one of these tensors and calculating it along the way of this backpropagation. So as an example, we have a b in raw here. 

We're going to be calculating a db in raw. So here, I'm telling PyTorch that we want to retain the grad of all these intermediate values, because here in exercise one, we're going to calculate the backward pass. So we're going calculate all these d variables and use the CMP function I've introduced above to check our correctness with respect to what PyTorch is telling us. 

This is going to be exercise one, where we sort of backpropagate through this entire graph. Now, just to give you a very quick preview of what's going to happen in exercise two and below, here we have fully broken up the loss and backpropagated through it manually in all the little atomic pieces that make it up. But here, we're going to collapse the loss into a single cross-entropy cull.

And instead, we're going to analytically derive, using math and paper and pencil, the gradient of the loss with respect to the logits. And instead of backpropagating through all of its little chunks one at a time, we're just going to analytically derive what that gradient is, and we're going to implement that, which is much more efficient, as we'll see in a bit. Then we're going to do the exact same thing for batch normalization. 

So instead of breaking up BatchNorm into all the little tiny components, we're going to use pen and paper and mathematics and calculus to derive the gradient through the BatchNorm layer. So we're going to calculate the backward pass through BatchNorm layer in a much more efficient expression, instead of backward propagating through all of its little pieces independently. So that's going to be exercise three.

And then in exercise four, we're going to put it all together. And this is the full code of training this two-layer MLP. And we're going to basically insert our manual backprop, and we're going to take out loss.backward. And you will basically see that you can get all the same results using fully your own code. 

And the only thing we're using from PyTorch is the torch.tensor to make the calculations efficient. But otherwise, you will understand fully what it means to forward and backward the neural net and train it. And I think that'll be awesome. 

So let's get to it. Okay, so I ran all the cells of this notebook all the way up to here. And I'm going to erase this.

And I'm going to start implementing backward pass starting with DLogProps. So we want to understand what should go here to calculate the gradient of the loss with respect to all the elements of the LogProps tensor. Now, I'm going to give away the answer here. 

But I wanted to put a quick note here that I think will be most pedagogically useful for you is to actually go into the description of this video and find the link to this Jupyter notebook. You can find it both on GitHub, but you can also find Google Colab with it. So you don't have to install anything, you will just go to a website on Google Colab. 

And you can try to implement these derivatives or gradients yourself. And then if you are not able to come to my video and see me do it. And so work in tandem and try it first yourself and then see me give away the answer. 

And I think that'll be most valuable to you. And that's how I recommend you go through this lecture. So we are starting here with DLogProps.

Now, DLogProps will hold the derivative of the loss with respect to all the elements of LogProps. What is inside LogProps? The shape of this is 32 by 27. So it's not going to surprise you that DLogProps should also be an array of size 32 by 27, because we want the derivative of the loss with respect to all of its elements. 

So the sizes of those are always going to be equal. Now, how does LogProps influence the loss? Loss is negative LogProps indexed with range of n and yb and then the mean of that. Now, just as a reminder, yb is just basically an array of all the correct indices. 

So what we're doing here is we're taking the LogProps array of size 32 by 27. Right. And then we are going in every single row. 

And in each row, we are plugging out the index 8 and then 14 and 15 and so on. So we're going down the rows. That's the iterator range of n. And then we are always plugging out the index of the column specified by this tensor yb. 

So in the zeroth row, we are taking the eighth column. In the first row, we're taking the 14th column, etc. And so LogProps at this plucks out all those LogProps probabilities of the correct next character in a sequence. 

So that's what that does. And the shape of this or the size of it is, of course, 32, because our batch size is 32. So these elements get plucked out, and then their mean and the negative of that becomes loss. 

So I always like to work with simpler examples to understand the numerical form of the derivative. What's going on here is once we've plucked out these examples, we're taking the mean and then the negative. So the loss basically, I can write it this way, is the negative of, say, a plus b plus c. And the mean of those three numbers would be, say, negative, divide three. 

That would be how we achieve the mean of three numbers a, b, c, although we actually have 32 numbers here. And so what is basically the loss by, say, like dA, right? Well, if we simplify this expression mathematically, this is negative 1 over 3 of a plus negative 1 over 3 of b plus negative 1 over 3 of c. And so what is d loss by dA? It's just negative 1 over 3. And so you can see that if we don't just have a, b, and c, but we have 32 numbers, then d loss by d, every one of those numbers is going to be 1 over n more generally, because n is the size of the batch, 32 in this case. So d loss by d logprobs is negative 1 over n in all these places. 

Now, what about the other elements inside logprobs? Because logprobs is a large array. You see that logprobs.shank is 32 by 27. But only 32 of them participate in the loss calculation. 

So what's the derivative of all the other most of the elements that do not get plucked out here? Well, their loss intuitively is 0. Sorry, their gradient intuitively is 0. And that's because they do not participate in the loss. So most of these numbers inside this tensor does not feed into the loss. And so if we were to change these numbers, then the loss doesn't change, which is the equivalent of saying that the derivative of the loss with respect to them is 0. They don't impact it.

So here's a way to implement this derivative, then. We start out with tors.zeros of shape 32 by 27. Or let's just say, instead of doing this, because we don't want to hard code numbers, let's do tors.zeros like logprobs. 

So basically, this is going to create an array of zeros exactly in the shape of logprobs. And then we need to set the derivative of negative 1 over n inside exactly these locations. So here's what we can do. 

dlogprobs indexed in the identical way will be just set to negative 1 over 0 divide n, right? Just like we derived here. So now let me erase all of this reasoning. And then this is the candidate derivative for dlogprobs. 

Let's uncomment the first line and check that this is correct. Okay. So CMP ran. 

And let's go back to CMP. And you see that what it's doing is it's calculating if the calculated value by us, which is dt, is exactly equal to t.grad as calculated by PyTorch. And then this is making sure that all of the elements are exactly equal, and then converting this to a single Boolean value. 

Because we don't want a Boolean tensor, we just want a Boolean value. And then here, we are making sure that, okay, if they're not exactly equal, maybe they are approximately equal because of some floating point issues, but they're very, very close. So here we are using torch.allclose, which has a little bit of a wiggle available, because sometimes you can get very, very close. 

But if you use a slightly different calculation, because of floating point arithmetic, you can get a slightly different result. So this is checking if you get an approximately close.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Close result. And then here we are checking the maximum, basically the value that has the highest difference, and what is the difference in the absolute value difference between those two. And so we are printing whether we have an exact equality, an approximate equality, and what is the largest difference. 

And so here, we see that we actually have exact equality. And so therefore, of course, we also have an approximate equality. And the maximum difference is exactly zero. 

So basically, rdlogprops is exactly equal to what PyTorch calculated to be logprops.grad in its backpropagation. So, so far, we're doing pretty well. Okay, so let's now continue our backpropagation. 

We have that logprops depends on props through a log. So all the elements of props are being element-wise applied log2. Now, if we want deep props then, remember your micrograd training, we have like a log node. 

It takes in props and creates logprops. And deep props will be the local derivative of that individual operation, log, times the derivative of the loss with respect to its output, which in this case is dlogprops. So what is the local derivative of this operation? Well, we are taking log element-wise.

And we can come here and we can see, well, from alpha is your friend, that d by dx of log of x is just simply 1 over x. So therefore, in this case, x is props. So we have d by dx is 1 over x, which is 1 over props. And then this is the local derivative. 

And then times we want to chain it. So this is chain rule, times dlogprops. Then let me uncomment this and let me run the cell in place. 

And we see that the derivative of props as we calculated here is exactly correct. And so notice here how this works. Props is going to be inverted and then element-wise multiplied here. 

So if your props is very, very close to 1, that means your network is currently predicting the character correctly, then this will become 1 over 1. And dlogprops just gets passed through. But if your probabilities are incorrectly assigned, so if the correct character here is getting a very low probability, then 1.0 dividing by it will boost this and then multiply by dlogprops. So basically what this line is doing intuitively is it's taking the examples that have a very low probability currently assigned and it's boosting their gradient. 

You can look at it that way. Next up is countSumInv. So we want the derivative of this. 

Now let me just pause here and kind of introduce what's happening here in general because I know it's a little bit confusing. We have the logits that come out of the neural net. Here what I'm doing is I'm finding the maximum in each row and I'm subtracting it for the purpose of numerical stability. 

And we talked about how if you do not do this, you run into numerical issues if some of the logits take on too large values because we end up exponentiating them. So this is done just for safety numerically. Then here's the exponentiation of all the logits to create our counts.

And then we want to take the sum of these counts and normalize so that all the probs sum to 1. Now here instead of using 1 over countSum, I use raised to the power of negative 1. Mathematically they are identical. I just found that there's something wrong with the PyTorch implementation of the backward passive division and it gives a weird result, but that doesn't happen for star star negative 1, so I'm using this formula instead. But basically all that's happening here is we've got the logits, we want to exponentiate all of them, and we want to normalize the counts to create our probabilities. 

It's just that it's happening across multiple lines. So now here we want to first take the derivative, we want to back propagate into countSumIf and then into counts as well. So what should be the countSumIf? Now we actually have to be careful here because we have to scrutinize and be careful with the shapes. 

So counts.shape and then countSumInf.shape are different. So in particular counts is 32 by 27, but this countSumIf is 32 by 1. And so in this multiplication here we also have an implicit broadcasting that PyTorch will do because it needs to take this column tensor of 32 numbers and replicate it horizontally 27 times to align these two tensors so it can do an element-wise multiply. So really what this looks like is the following using a toy example again. 

What we really have here is just props is counts times countsSumIf, so it's c equals a times b, but a is 3 by 3 and b is just 3 by 1, a column tensor. And so PyTorch internally replicated these elements of b and it did that across all the columns. So for example b1 which is the first element of b would be replicated here across all the columns in this multiplication. 

And now we're trying to back propagate through this operation to countsSumInf. So when we are calculating this derivative it's important to realize that this looks like a single operation but actually is two operations applied sequentially. The first operation that PyTorch did is it took this column tensor and replicated it across all the columns basically 27 times.

So that's the first operation, it's a replication. And then the second operation is the multiplication. So let's first back prop through the multiplication. 

If these two arrays were of the same size and we just have a and b, both of them 3 by 3, then how do we back propagate through a multiplication? So if we just have scalars and not tensors, then if you have c equals a times b, then what is the derivative of c with respect to b? Well it's just a. And so that's the local derivative. So here in our case undoing the multiplication and back propagating through just the multiplication itself, which is element-wise, is going to be the local derivative, which in this case is simply counts, because counts is the a. So this is the local derivative, and then times, because of the chain rule, d props. So this here is the derivative, or the gradient, but with respect to replicated b. But we don't have a replicated b, we just have a single b column. 

So how do we now back propagate through the replication? And intuitively this b1 is the same variable and it's just reused multiple times. And so you can look at it as being equivalent to a case we've encountered in micrograd. And so here I'm just pulling out a random graph we used in micrograd.

We had an example where a single node has its output feeding into two branches of basically the graph until the loss function. And we're talking about how the correct thing to do in the backward pass is we need to sum all the gradients that arrive at any one node. So across these different branches the gradients would sum. 

So if a node is used multiple times, the gradients for all of its uses sum during backpropagation. So here b1 is used multiple times in all these columns, and therefore the right thing to do here is to sum horizontally across all the rows. So we want to sum in dimension one, but we want to retain this dimension so that the countSumInv and its gradient are going to be exactly the same shape. 

So we want to make sure that we keep them as true so we don't lose this dimension. And this will make the countSumInv be exactly shaped 32 by 1. So revealing this comparison as well and running this, we see that we get an exact match. So this derivative is exactly correct.

And let me erase this. Now let's also backpropagate into countS, which is the other variable here to create props. So from props to countSumInv, we just did that. 

Let's go into countS as well. So dcounts will be, dcounts is our a, so dc by da is just b, so therefore it's countSumInv, and then times chain rule dprops. Now countSumInv is 32 by 1, dprops is 32 by 27.

So those will broadcast fine and will give us dcounts. There's no additional summation required here. There will be a broadcasting that happens in this multiply here because countSumInv needs to be replicated again to correctly multiply dprops. 

But that's going to give the correct result. So as far as this single operation is concerned. So we've backpropagated from props to countS, but we can't actually check the derivative of countS. 

I have it much later on. And the reason for that is because countSumInv depends on countS. And so there's a second branch here that we have to finish because countSumInv backpropagates into countSum, and countSum will backpropagate into countS. 

And so countS is a node that is being used twice. It's used right here into props, and it goes through this other branch through countSumInv. So even though we've calculated the first contribution of it, we still have to calculate the second contribution of it later. 

Okay, so we're continuing with this branch. We have the derivative for countSumInv. Now we want the derivative of countSum. 

So dcountSum equals, what is the local derivative of this operation? So this is basically an element-wise 1 over countSum. So countSum raised to the power of negative 1 is the same as 1 over countSum. If we go to Wolfram Alpha, we see that x to the negative 1, d by dx of it, is basically negative x to the negative 2, right? Negative 1 over s squared is the same as negative x to the negative 2. So dcountSum here will be, local derivative is going to be negative countSum to the negative 2, that's the local derivative, times chain rule, which is dcountSumInv. 

So that's dcountSum. Let's uncomment this and check that I am correct. Okay, so we have perfect equality. 

And there's no sketchiness going on here with any shapes because these are of the same shape. Okay, next up, we want to back propagate through this line. We have that countSum is count.sum along the rows. 

So I wrote out some help here. We have to keep in mind that counts, of course, is 32 by 27, and countSum is 32 by 1. So in this back propagation, we need to take this column of derivatives and transform it into a array of derivatives, two-dimensional array. So what is this operation doing? We're taking in some kind of an input, like say a 3-by-3 matrix A, and we are summing up the rows into a column tensor B, B1, B2, B3, that is basically this. 

So now we have the derivatives of the loss with respect to B, all the elements of B. And now we want to derive the loss with respect to all these little A's. So how do the B's depend on the A's is basically what we're after. What is the local derivative of this operation? Well, we can see here that B1 only depends on these elements here. 

The derivative of B1 with respect to all of these elements down here is 0. But for these elements here, like A11, A12, etc., the local derivative is 1, right? So DB1 by DA11, for example, is 1. So it's 1, 1, and 1. So when we have the derivative of the loss with respect to B1, the local derivative of B1 with respect to these inputs is 0s here, but it's 1 on these guys. So in the chain rule, we have the local derivative times the derivative of B1. And so because the local derivative is 1 on these three elements, the local derivative multiplying the derivative of B1 will just be the derivative of B1. 

And so you can look at it as a router. Basically, an addition is a router of gradient. Whatever gradient comes from above, it just gets routed equally to all the elements that participate in that addition. 

So in this case, the derivative of B1 will just flow equally to the derivative of A11, A12, and A13. So if we have a derivative of all the elements of B in this column tensor, which is D counts sum that we've calculated just now, we basically see that what that amounts to is all of these are now flowing to all these elements of A, and they're doing that horizontally. So basically, what we want is we want to take the D counts sum of size 32 by 1, and we just want to replicate it 27 times horizontally to create 32 by 27 array. 

So there's many ways to implement this operation. You could, of course, just replicate the tensor, but I think maybe one clean one is that D counts is simply torch.once like, so just two-dimensional arrays of once in the shape of counts, so 32 by 27, times D counts sum. So this way, we're letting the broadcasting here basically implement the replication. 

You can look at it that way. But then we have to also be careful, because D counts was all already calculated. We calculated earlier here, and that was just the first branch, and we're now finishing the second branch. 

So we need to make sure that these gradients add, so plus equals. And then here, let's comment out the comparison, and let's make sure, crossing fingers, that we have the correct result. So PyTorch agrees with us on this gradient as well. 

Okay, hopefully we're getting a hang of this now. Counts is an element-wise exp of normlogits. So now we want denormlogits, and because it's an element-wise operation, everything is very simple. 

What is the local derivative of e to the x? It's famously just e to the x, so this is the local derivative. That is the local derivative. Now, we already calculated it, and it's inside counts, so we might as well potentially just reuse counts. 

That is the local derivative times D counts. Funny as that looks. Counts times D counts is the derivative on the normlogits.

And now let's erase this, and let's verify, and it looks good. So that's normlogits. Okay, so we are here on this line now, denormlogits. 

We have that, and we're trying to calculate D logits and D logit maxes, so backpropagating through this line. Now, we have to be careful here because the shapes, again, are not the same, and so there's an implicit broadcasting happening here. So normlogits has the shape of 32 by 27. 

Logits does as well, but logit maxes is only 32 by 1, so there's a broadcasting here in the minus. Now, here I tried to sort of write out a toy example again. We basically have that this is our C equals A minus B, and we see that because of the shape, these are 3 by 3, but this one is just a column.

And so, for example, every element of C, we have to look at how it came to be. And every element of C is just the corresponding element of A minus basically that associated B. So it's very clear now that the derivatives of every one of these C's with respect to their inputs are 1 for the corresponding A, and it's a negative 1 for the corresponding B. And so, therefore, the derivatives on the C will flow equally to the corresponding A's and then also to the corresponding B's, but then in addition to that, the B's are broadcast, so we'll have to do the additional sum just like we did before. And of course, the derivatives for B's will undergo a minus because the local derivative here is negative 1. So dC 32 by dB 3 is negative 1. So let's just implement that. 

Basically, dlogits will be exactly copying the derivative on normlogits. So dlogits equals dnormlogits, and I'll do a dot clone for safety, so we're just making a copy. And then we have that dlogit maxis will be the negative of dnormlogits because of the negative sign. 

And then we have to be careful because logit maxis is a column, and so just like we saw before, because we keep replicating the same elements across all the columns, then in the backward pass, because we keep reusing this, these are all just like separate branches of use of that one variable. And so therefore we have to do a sum along one, we'd keep them equals true, so that we don't destroy this dimension. And then dlogit maxis will be the same shape. 

Now we have to be careful because this dlogits is not the final dlogits, and that's because not only do we get gradient signal into logits through here, but logit maxis is a function of logits, and that's a second branch into logits. So this is not yet our final derivative for logits, we will come back later for the second branch. For now, dlogit maxis is the final derivative. 

So let me uncomment this CMP here, and let's just run this. And logit maxis, if PyTorch agrees with us. So that was the derivative into, through this line. 

Now before we move on, I want to pause here briefly, and I want to look at these logit maxis and especially their gradients. We've talked previously in the previous lecture that the only reason we're doing this is for the numerical stability of the softmax that we are implementing here. And we talked about how if you take these logits for any one of these examples, so one row of this logits tensor, if you add or subtract any value equally to all the elements, then the value of the probs will be unchanged. 

You're not changing the softmax. The only thing that this is doing is it's making sure that exp doesn't overflow. And the reason we're using a max is because then we are guaranteed that each row of logits, the highest number, is zero. 

And so this will be safe. And so basically what that has repercussions. If it is the case that changing logit maxis does not change the probs, and therefore does not change the loss, then the gradient on logit maxis should be zero.

Right? Because saying those two things is the same. So indeed we hope that this is very, very small numbers. Indeed we hope this is zero. 

Now, because of floating point sort of wonkiness, this doesn't come out exactly zero. Only in some of the rows it does. But we get extremely small values, like 1e-9 or 10. 

And so this is telling us that the values of logit maxis are not impacting the loss, as they shouldn't. It feels kind of weird to backpropagate through this branch, honestly, because if you have any implementation of like f.crossentropy in PyTorch, and you block together all these elements, and you're not doing the backpropagation piece by piece, then you would probably assume that the derivative through here is exactly zero. So you would be sort of skipping this branch, because it's only done for numerical stability. 

But it's interesting to see that even if you break up everything into the full atoms, and you still do the computation as you'd like with respect to numerical stability, the correct thing happens. And you still get very, very small gradients here, basically reflecting the fact that the values of these do not matter with respect to the final loss. Okay, so let's now continue backpropagation through this line here. 

We've just calculated the logit maxis, and now we want to backprop into logits through this second branch. Now here, of course, we took logits, and we took the max along all the rows, and then we looked at its values here. Now the way this works is that in PyTorch, this thing here, the max returns both the values, and it returns the indices at which those values to count the maximum value. 

Now, in the forward pass, we only used values, because that's all we needed. But in the backward pass, it's extremely useful to know about where those maximum values occurred. And we have the indices at which they occurred. 

And this will, of course, help us do the backpropagation. Because what should the backward pass be here in this case? We have the logit tensor, which is 32 by 27. And in each row, we find the maximum value. 

And then that value gets plucked out into logit maxis. And so intuitively, basically, the derivative flowing through here then should be 1 times the local derivative is 1 for the appropriate entry that was plucked out, and then times the global derivative of the logit maxis. So really, what we're doing here, if you think through it, is we need to take the delogit maxis, and we need to scatter it to the correct positions in these logits from where the maximum values came.

And so I came up with one line of code that does that. Let me just erase a bunch of stuff here. So the line of, you could do it very similar to what we've done here, where we create a zeros, and then we populate the correct elements. 

So we use the indices here, and we would set them to be 1. But you can also use one-hot. So f.one-hot, and then I'm taking the logit max over the first dimension, dot indices, and I'm telling PyTorch that the dimension of every one of these tensors should be 27. And so what this is going to do is, okay, I apologize, this is crazy.

PLT.I am sure of this. It's really just an array of where the maxis came from in each row, and that element is 1, and all the other elements are 0. So it's a one-hot vector in each row, and these indices are now populating a single 1 in the proper place. And then what I'm doing here is I'm multiplying by the logit maxis. 

And keep in mind that this is a column of 32 by 1, and so when I'm doing this times the logit maxis, the logit maxis will broadcast, and that column will get replicated, and then element-wise multiply will ensure that each of these just gets routed to whichever one of these bits is turned on. And so that's another way to implement this kind of an operation, and both of these can be used. I just thought I would show an equivalent way to do it. 

And I'm using plus equals because we already calculated the logits here, and this is now the second branch. So let's look at logits and make sure that this is correct, and we see that we have exactly the correct answer. Next up, we want to continue with logits here. 

That is an outcome of a matrix multiplication and a bias offset in this linear layer. So I've printed out the shapes of all these intermediate tensors. We see that logits is of course 32 by 27, as we've just seen. 

Then the h here is 32 by 64, so these are 64-dimensional hidden states. And then this w matrix projects those 64-dimensional vectors into 27 dimensions, and then there's a 27-dimensional offset, which is a one-dimensional vector. Now we should note that this plus here actually broadcasts, because h multiplied by w2 will give us a 32 by 27.

And so then this plus b2 is a 27-dimensional vector here. Now in the rules of broadcasting, what's going to happen with this bias vector is that this one-dimensional vector of 27 will get aligned with a padded dimension of one on the left, and it will basically become a row vector, and then it will get replicated vertically 32 times to make it 32 by 27, and then there's an element-wise multiply. Now the question is how do we backpropagate from logits to the hidden states, the weight matrix w2, and the bias b2? And you might think that we need to go to some matrix calculus, and then we have to look up the derivative for a matrix multiplication, but actually you don't have to do any of that, and you can go back to first principles and derive this yourself on a piece of paper. 

And specifically what I like to do, and what I find works well for me, is you find a specific small example that you then fully write out, and then in the process of analyzing how that individual small example works, you will understand the broader pattern, and you'll be able to generalize and write out the full general formula for how these derivatives flow in an expression like this. So let's try that out. So pardon the low-budget production here, but what I've done here is I'm writing it out on a piece of paper. 

Really what we are interested in is we have a multiply b plus c, and that creates a d, and we have the derivative of the loss with respect to d, and we'd like to know what the derivative of the loss is with respect to a, b, and c. Now these here are little two-dimensional examples of a matrix multiplication. 2 by 2 times a 2 by 2 plus a 2, a vector of just two elements, c1 and c2, gives me a 2 by 2. Now notice here that I have a bias vector here called c, and the bias vector is c1 and c2, but as I described over here, that bias vector will become a row vector in the broadcasting, and will replicate vertically. So that's what's happening here as well. 

c1 c2 is replicated vertically, and we see how we have two rows of c1 c2 as a result. So now when I say write it out, I just mean like this. Basically break up this matrix multiplication into the actual thing that's going on under the hood.

So as a result of matrix multiplication and how it works, d11 is the result of a dot product between the first row of a and the first column of b. So a11 b11 plus a12 b21 plus c1, and so on so forth for all the other elements of d. And once you actually write it out, it becomes obvious this is just a bunch of multiplies and adds, and we know from micrograd how to differentiate multiplies and adds. And so this is not scary anymore, it's not just matrix multiplication, it's just tedious unfortunately, but this is completely tractable. We have dl by d for all of these, and we want dl by all these little other variables. 

So how do we achieve that, and how do we actually get the gradients? Okay, so the low budget production continues here. So let's for example derive the derivative of the loss with respect to a11. We see here that a11 occurs twice in our simple expression, right here, right here, and influences d11 and d12.

So what is dl by d a11? Well, it's dl by d11 times the local derivative of d11, which in this case is just b11, because that's what's multiplying a11 here. And likewise here, the local derivative of d12 with respect to a11 is just b12, and so b12 will, in the chain rule therefore, multiply dl by d12. And then because a11 is used both to produce d11 and d12, we need to add up the contributions of both of those sort of chains that are running in parallel, and that's why we get a plus just adding up those two contributions, and that gives us dl by d a11. 

We can do the exact same analysis for the other one, for all the other elements of a, and when you simply write it out, it's just super simple taking the gradients on expressions like this. You find that this matrix dl by da that we're after, if we just arrange all of them in the same shape as a takes, so a is just too much matrix, so dl by da here will be also just the same shape tensor with the derivatives now, so dl by da11, etc. And we see that actually we can express what we've written out here as a matrix multiply, and so it just so happens that dl by, that all of these formulas that we've derived here by taking gradients can actually be expressed as a matrix multiplication.

In particular, we see that it is the matrix multiplication of these two matrices, so it is the dl by d, and then matrix multiplying b, but b transpose actually. So you see that b21 and b12 have changed place, whereas before we had of course b11, b12, b21, b22. So you see that this other matrix b is transposed. 

And so basically what we have, long story short, just by doing very simple reasoning here, by breaking up the expression in the case of a very simple example, is that dl by da is, which is this, is simply equal to dl by dd matrix multiplied with b transpose. So that is what we have so far. Now we also want the derivative with respect to b and c. Now for b, I'm not actually doing the full derivation, because honestly it's not deep, it's just annoying, it's exhausting. 

You can actually do this analysis yourself. You'll also find that if you take this

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

You will find that dL by dB is also a matrix multiplication. In this case, you have to take the matrix A and transpose it, and matrix multiply that with dL by dD, and that's what gives you dL by dB. And then here for the offsets C1 and C2, if you again just differentiate with respect to C1, you will find an expression like this, and C2, an expression like this, and basically you'll find that dL by dC is simply, because they're just offsetting these expressions, you just have to take the dL by dD matrix of the derivatives of D, and you just have to sum across the columns, and that gives you the derivatives for C. So, long story short, the backward pass of a matrix multiply is a matrix multiply, and instead of, just like we had D equals A times B plus C, in a scalar case, we sort of like arrive at something very, very similar, but now with a matrix multiplication instead of a scalar multiplication.

So, the derivative of D with respect to A is dL by dD matrix multiply B transpose, and here it's A transpose multiply dL by dD. But in both cases, it's a matrix multiplication with the derivative and the other term in the multiplication. And for C, it is a sum.

Now, I'll tell you a secret. I can never remember the formulas that we just derived for backpropagating from matrix multiplication, and I can backpropagate through these expressions just fine. And the reason this works is because the dimensions have to work out.

So, let me give you an example. Say I want to create DH, then what should DH be? Number one, I have to know that the shape of DH must be the same as the shape of H, and the shape of H is 32 by 64. And then the other piece of information I know is that DH must be some kind of matrix multiplication of dLogits with W2.

And dLogits is 32 by 27, and W2 is 64 by 27. There is only a single way to make the shape work out in this case, and it is indeed the correct result. In particular here, H needs to be 32 by 64.

The only way to achieve that is to take a dLogits and matrix multiply it with... You see how I have to take W2, but I have to transpose it to make the dimensions work out. So, W2 transpose. And it's the only way to matrix multiply those two pieces to make the shapes work out, and that turns out to be the correct formula.

So, if we come here, we want DH, which is dA, and we see that dA is dL by dD matrix multiply B transpose. So, that's dLogits multiply, and B is W2. So, W2 transpose, which is exactly what we have here.

So, there's no need to remember these formulas. Similarly, now if I want dW2, well, I know that it must be a matrix multiplication of dLogits and H. And maybe there's a few transpose... Like, there's one transpose in there as well, and I don't know which way it is. So, I have to come to W2, and I see that its shape is 64 by 27.

And that has to come from some matrix multiplication of these two. And so, to get a 64 by 27, I need to take H. I need to transpose it. And then I need to matrix multiply it.

So, that will become 64 by 32. And then I need to matrix multiply the 32 by 27. And that's going to give me a 64 by 27.

So, I need to matrix multiply this with dLogits.shape, just like that. That's the only way to make the dimensions work out. And just use matrix multiplication.

And if we come here, we see that that's exactly what's here. So, A transpose. A for us is H. Multiply it with dLogits.

So, that's W2. And then dB2 is just the vertical sum. And actually, in the same way, there's only one way to make the shapes work out.

I don't have to remember that it's a vertical sum along the 0th axis, because that's the only way that this makes sense. Because B2 shape is 27. So, in order to get a dLogits here, it's 32 by 27.

So, knowing that it's just sum over dLogits in some direction, that direction must be 0, because I need to eliminate this dimension. So, it's this. So, this is kind of like the hacky way.

Let me copy, paste, and delete that. And let me swing over here. And this is our backward pass for the linear layer, hopefully.

So, now let's uncomment these three. And we're checking that we got all the three derivatives correct. And run.

And we see that H, W2, and B2 are all exactly correct. So, we backpropagated through a linear layer. Now, next up, we have derivative for the H already.

And we need to backpropagate through tanh into HPREACT. So, we want to derive dHPREACT. And here, we have to backpropagate through a tanh.

And we've already done this in micrograd. And we remember that tanh is a very simple backward formula. Now, unfortunately, if I just put in d by dx of tanh of x into both from alpha, it lets us down.

It tells us that it's a hyperbolic secant function squared of x. It's not exactly helpful. But luckily, Google Image Search does not let us down. And it gives us the simpler formula.

And in particular, if you have that A is equal to tanh of z, then dA by dz, backpropagating through tanh, is just 1 minus A squared. And take note that 1 minus A squared, A here is the output of the tanh, not the input to the tanh, z. So, the dA by dz is here formulated in terms of the output of that tanh. And here also, in Google Image Search, we have the full derivation.

If you want to actually take the actual definition of tanh and work through the math to figure out 1 minus tanh squared of z. So, 1 minus A squared is the local derivative. In our case, that is 1 minus the output of tanh squared, which here is h. So, it's h squared. And that is the local derivative.

And then times the chain rule, dh. So, that is going to be our candidate implementation. So, if we come here, and then uncomment this.

Let's hope for the best. And we have the right answer. Okay, next up, we have dhpreact.

And we want to backpropagate into the gain, the bnRaw, and the bnBias. So, here, this is the bash norm parameters, bnGain and bnBias, inside the bash norm, that take the bnRaw, that is exact unit Gaussian, and then scale it and shift it. And these are the parameters of the bash norm.

Now, here, we have a multiplication. But it's worth noting that this multiply is very, very different from this matrix multiply here. Matrix multiply are dot products between rows and columns of these matrices involved.

This is an element-wise multiply. So, things are quite a bit simpler. Now, we do have to be careful with some of the broadcasting happening in this line of code, though.

So, you see how bnGain and bnBias are 1 by 64. But hpreact and bnRaw are 32 by 64. So, we have to be careful with that and make sure that all the shapes work out fine and that the broadcasting is correctly backpropagated.

So, in particular, let's start with dbnGain. So, dbnGain should be, and here, this is, again, element-wise multiply. And whenever we have a times b equals c, we saw that the local derivative here is just, if this is a, the local derivative is just the b, the other one.

So, the local derivative is just bnRaw and then times chain rule. So, dhpreact. So, this is the candidate gradient.

Now, again, we have to be careful because bnGain is of size 1 by 64. But this here would be 32 by 64. And so, the correct thing to do in this case, of course, is that bnGain, here is a rule vector of 64 numbers, it gets replicated vertically in this operation.

And so, therefore, the correct thing to do is to sum because it's being replicated. And therefore, all the gradients in each of the rows that are now flowing backwards need to sum up to that same tensor dbnGain. So, we have to sum across all the zero, all the examples, basically, which is the direction in which this gets replicated.

And now, we have to be also careful because bnGain is of shape 1 by 64. So, in fact, I need to keep them as true. Otherwise, I would just get 64.

Now, I don't actually really remember why the bnGain and the bnBias, I made them be 1 by 64. But the biases, b1 and b2, I just made them be one-dimensional vectors. They're not two-dimensional tensors.

So, I can't recall exactly why I left the gain and the bias as two-dimensional. But it doesn't really matter as long as you are consistent and you're keeping it the same. So, in this case, we want to keep the dimension so that the tensor shapes work.

Next up, we have bnRaw. So, dbnRaw will be bnGain multiplying dhPreact. That's our chain rule.

Now, what about the dimensions of this? We have to be careful, right? So, dhPreact is 32 by 64. bnGain is 1 by 64. So, it will just get replicated to create this multiplication, which is the correct thing because in a forward pass, it also gets replicated in just the same way.

So, in fact, we don't need the brackets here. We're done. And the shapes are already correct.

And finally, for the bias, very similar. This bias here is very, very similar to the bias we saw in the linear layer. And we see that the gradients from hPreact will simply flow into the biases and add up because these are just offsets.

And so, basically, we want this to be dhPreact, but it needs to sum along the right dimension. And in this case, similar to the gain, we need to sum across the zeroth dimension, the examples, because of the way that the bias gets replicated vertically. And we also want to have keepDim as true.

And so, this will basically take this and sum it up and give us a 1 by 64. So, this is the candidate implementation. It makes all the shapes work.

Let me bring it up down here. And then let me uncomment these three lines to check that we are getting the correct result for all the three tensors. And indeed, we see that all of that got back propagated correctly.

So, now we get to the batch norm layer. We see how here bnGain and bnBias are the parameters, so the backpropagation ends. But bnRaw now is the output of the standardization.

So, here what I'm doing, of course, is I'm breaking up the batch norm into manageable pieces so we can backpropagate through each line individually. But basically, what's happening is bnMeanI is the sum. So, this is the bnMeanI.

I apologize for the variable naming. bnDiff is x minus mu. bnDiff2 is x minus mu squared here inside the variance.

bnVar is the variance, so sigma squared. This is bnVar. And it's basically the sum of squares.

So, this is the x minus mu squared and then the sum. Now, you'll notice one departure here. Here, it is normalized as 1 over m, which is the number of examples.

Here, I am normalizing as 1 over n minus 1 instead of m. And this is deliberate, and I'll come back to that in a bit when we are at this line. It is something called the Bessel's correction. But this is how I want it in our case.

bnVar inv then becomes basically bnVar plus epsilon. Epsilon is my negative 5. And then it's 1 over square root is the same as raising to the power of negative 0.5, right? Because 0.5 is square root. And then negative makes it 1 over square root.

So, bnVar inv is 1 over this denominator here. And then we can see that bnRaw, which is the x hat here, is equal to the bnDiff, the numerator, multiplied by the bnVar inv. And this line here that creates hPreact was the last piece we've already backpropagated through it.

So, now what we want to do is we are here, and we have bnRaw. And we have to first backpropagate into bnDiff and bnVar inv. So, now we're here, and we have dbnRaw.

And we need to backpropagate through this line. Now, I've written out the shapes here. And indeed, bnVar inv is a shape 1 by 64.

So, there is a broadcasting happening here that we have to be careful with. But it is just an element-wise simple multiplication. By now, we should be pretty comfortable with that.

To get dbnDiff, we know that this is just bnVar inv multiplied with dbnRaw. And conversely, to get dbnVar inv, we need to take bnDiff and multiply that by dbnRaw. So, this is the candidate.

But, of course, we need to make sure that broadcasting is obeyed. So, in particular, bnVar inv multiplying with dbnRaw will be okay and give us 32 by 64, as we expect. But dbnVar inv would be taking a 32 by 64, multiplying it by 32 by 64.

So, this is a 32 by 64. But, of course, this bnVar inv is only 1 by 64. So, the second line here needs a sum across the examples.

And because there's this dimension here, we need to make sure that keepDim is true. So, this is the candidate. Let's erase this.

And let's swing down here and implement it. And then let's comment out dbnVar inv and dbnDiff. Now, we'll actually notice that dbnDiff, by the way, is going to be incorrect.

So, when I run this, bnVar inv is correct. bnDiff is not correct. And this is actually expected, because we're not done with bnDiff.

So, in particular, when we slide here, we see here that bnRaw is a function of bnDiff. But, actually, bnVar inv is a function of bnVar, which is a function of bnDiff2, which is a function of bnDiff. So, it comes here.

So, bdnDiff, these variable names are crazy. I'm sorry. It branches out into two branches, and we've only done one branch of it.

We have to continue our backpropagation and eventually come back to bnDiff. And then we'll be able to do a plus equals and get the actual correct gradient. For now, it is good to verify that cmp also works.

It doesn't just lie to us and tell us that everything is always correct. It can, in fact, detect when your gradient is not correct. So, that's good to see as well.

Okay. So, now we have the derivative here. And we're trying to backpropagate through this line.

And because we're raising to a power of negative 0.5, I brought up the power rule. And we see that basically we have that the bnVar will now be... We bring down the exponent. So, negative 0.5 times x, which is this.

And now raise to the power of negative 0.5 minus 1, which is negative 1.5. Now, we would have to also apply a small chain rule here in our head, because we need to take further derivative of bnVar with respect to this expression here inside the bracket. But because this is an element-wise operation and everything is fairly simple, that's just 1. And so, there's nothing to do there. So, this is the local derivative.

And then times the global derivative to create the chain rule. This is just times the bnVar. So, this is our candidate.

Let me bring this down and uncomment the check. And we see that we have the correct result. Now, before we backpropagate through the next line, I want to briefly talk about the note here, where I'm using the Bessel's correction, dividing by n minus 1 instead of dividing by n, when I normalize here the sum of squares.

Now, you'll notice that this is a departure from the paper, which uses 1 over n instead, not 1 over n minus 1. There, m is our n. And so, it turns out that there are two ways of estimating variance of an array. One is the biased estimate, which is 1 over n. And the other one is the unbiased estimate, which is 1 over n minus 1. Now, confusingly, in the paper, this is not very clearly described. And also, it's a detail that kind of matters, I think.

They are using the biased version at training time. But later, when they are talking about the inference, they are mentioning that when they do the inference, they are using the unbiased estimate, which is the n minus 1 version, basically, for inference, and to calibrate the running mean and the running variance, basically. And so, they actually introduce a train-test mismatch, where in training, they use the biased version, and in test time, they use the unbiased version.

I find this extremely confusing. You can read more about the Bessel's correction, and why dividing by n minus 1 gives you a better estimate of the variance in the case where you have population sizes or samples for a population that are very small. And that is indeed the case for us, because we are dealing with mini-batches.

And these mini-batches are a small sample of a larger population, which is the entire training set. And so, it just turns out that if you just estimate it using 1 over n, that actually almost always underestimates the variance. And it is a biased estimator, and it is advised that you use the unbiased version and divide by n minus 1. And you can go through this article here that I liked, that actually describes the full reasoning.

And I'll link it in the video description. Now, when you calculate the torsion variance, you'll notice that they take the unbiased flag, whether or not you want to divide by n or n minus 1. Confusingly, they do not mention what the default is for unbiased, but I believe unbiased by default is true. I'm not sure why the docs here don't cite that.

Now, in the batch norm 1D, the documentation, again, is kind of wrong and confusing. It says that the standard deviation is calculated via the biased estimator, but this is actually not exactly right. And people have pointed out that it is not right in a number of issues since then.

Because actually, the rabbit hole is deeper, and they follow the paper exactly. And they use the biased version for training. But when they're estimating the running standard deviation, they are using the unbiased version.

So again, there's the train test mismatch. So long story short, I'm not a fan of train test discrepancies. I basically kind of consider the fact that we use the biased version, the training time, and the unbiased test time.

I basically consider this to be a bug. And I don't think that there's a good reason for that. They don't really go into the detail of the reasoning behind it in this paper.

So that's why I basically prefer to use the Bessel's correction in my own work. Unfortunately, BatchNorm does not take a keyword argument that tells you whether or not you want to use the unbiased version or the biased version in both training tests. And so therefore, anyone using BatchNormalization basically, in my view, has a bit of a bug in the code.

And this turns out to be much less of a problem if your mini batch sizes are a bit larger. But still, I just find it kind of unpalatable. So maybe someone can explain why this is okay.

But for now, I prefer to use the unbiased version consistently, both during training and at test time. And that's why I'm using 1 over n minus 1 here. Okay, so let's now actually backpropagate through this line.

So the first thing that I always like to do is I like to scrutinize the shapes first. So in particular here, looking at the shapes of what's involved, I see that bnvar shape is 1 by 64. So it's a row vector and bndiff2.shape is 32 by 64.

So clearly here, we're doing a sum over the 0th axis to squash the first dimension of the shapes here using a sum. So that right away actually hints to me that there will be some kind of a replication or broadcasting in the backward pass. And maybe you're noticing the pattern here.

But basically, anytime you have a sum in the forward pass, that turns into a replication or broadcasting in the backward pass along the same dimension. And conversely, when we have a replication or a broadcasting in the forward pass, that indicates a variable reuse. And so in the backward pass, that turns into a sum over the exact same dimension.

And so hopefully, you're noticing that duality, that those two are kind of like the opposites of each other in the forward and the backward pass. Now, once we understand the shapes, the next thing I like to do always is I like to look at a toy example in my head to sort of just like understand roughly how the variable dependencies go in the mathematical formula. So here, we have a two-dimensional array at the end of 2, which we are scaling by a constant.

And then we are summing vertically over the columns. So if we have a 2x2 matrix A, and then we sum over the columns and scale, we would get a row vector b1, b2. And b1 depends on A in this way, where it's just sum the scale of A. And b2 in this way, where it's the second column summed and scaled.

And so looking at this basically, what we want to do now is we have the derivatives on b1 and b2, and we want to back propagate them into A's. And so it's clear that just differentiating in your head, the local derivative here is 1 over n minus 1 times 1 for each one of these A's. And basically, the derivative of b1 has to flow through the columns of A. Scaled by 1 over n minus 1. And that's roughly what's happening here.

So intuitively, the derivative flow tells us that dbn diff2 will be the local derivative of this operation. And there are many ways to do this, by the way, but I like to do something like this. Torch.onceLike of bn diff2.

So I'll create a large array two-dimensional of ones. And then I will scale it. So 1.0 divided by n minus 1. So this is an array of 1 over n minus 1. And that's sort of like the local derivative.

And now for the chain rule, I will simply just multiply it by dbn var. And notice here what's going to happen. This is 32 by 64, and this is just 1 by 64.

So I'm letting the broadcasting do the replication, because internally in PyTorch, basically dbn var, which is 1 by 64 row vector, will, in this multiplication, get copied vertically until the two are of the same shape. And then there will be an element-wise multiply. And so the broadcasting is basically doing the replication.

And I will end up with the derivatives of dbn diff2 here. So this is the candidate solution. Let's bring it down here.

Let's uncomment this line where we check it. And let's hope for the best. And indeed, we see that this is the correct formula.

Next up, let's differentiate here into bn diff. So here we have that bn diff is element-wise squared to create bn diff2. So this is a relatively simple derivative, because it's a simple element-wise operation.

So it's kind of like the scalar case. And we have that dbn diff should be, if this is x squared, then the derivative of this is 2x, right? So it's simply 2 times bn diff. That's the local derivative.

And then times chain rule. And the shape of these is the same. They are of the same shape.

So times this. So that's the backward pass for this variable. Let me bring that down here.

And now we have to be careful, because we already calculated dbn diff, right? So this is just the end of the other branch coming back to bn diff. Because bn diff will already backpropagate it to way over here from bn raw. So we now completed the second branch.

And so that's why I have to do plus equals. And if you recall, we had an incorrect derivative for bn diff before. And I'm hoping that once we append this last missing piece, we have the exact correctness.

So let's run. And bn diff2, bn diff now actually shows the exact correct derivative. So that's comforting.

OK, so let's now backpropagate through this line here. The first thing we do, of course, is we check the shapes. And I wrote them out here.

And basically, the shape of this is 32 by 64. H prebn is the same shape. But bn mean i is a row vector, 1 by 64.

So this minus here will actually do broadcasting. And so we have to be careful with that. And as a hint to us, again, because of the duality, a broadcasting in the forward pass means a variable reuse.

And therefore, there will be a sum in the backward pass. So let's write out the backward pass here now. Backpropagate into the H prebn.

Because these are the same shape, then the local derivative for each one of the elements here is just 1 for the corresponding element in here. So basically, what this means is that the gradient just simply copies. It's just a variable assignment.

It's equality. So I'm just going to clone this tensor just for safety to create an exact copy. Of dbn diff.

And then here, to backpropagate into this one, what I'm inclined to do here is dbn mean i will basically be what is the local derivative? Well, it's negative torch dot once like of the shape of bn diff, right? And then times the derivative here, dbn diff. And this here is the backpropagation for the replicated bn mean i. So I still have to backpropagate through the replication in the broadcasting. And I do that by doing a sum.

So I'm going to take this whole thing, and I'm going to do a sum over the zeroth dimension, which was the replication. So if you scrutinize this, by the way, you'll notice that this is the same shape as that. And so what I'm doing here doesn't actually make that much sense because it's just an array of ones multiplying dbn diff.

So in fact, I can just do this, and that is equivalent. So this is the candidate backward pass. Let me copy it here.

And then let me comment out this one. And this one. Enter.

And it's wrong. Damn. Actually, sorry, this is supposed to be wrong.

And it's supposed to be wrong because we are backpropagating from a bn diff into h prebn. But we're not done because bn mean i depends on h prebn. And there will be a second portion of that derivative coming from this second branch.

So we're not done yet. And we expect it to be incorrect. So there you go.

So let's now backpropagate from bn mean i into h prebn. And so here again, we have to be careful because there's a broadcasting along, or there's a sum along the 0th dimension. So this will turn into broadcasting in the backward pass now.

And I'm going to go a little bit faster on this line because it is very similar to the line that we had before, multiplied in the past, in fact. So dh prebn will be, the gradient will be scaled by 1 over n. And then basically this gradient here on dbn mean i is going to be scaled by 1 over n. And then it's going to flow across all the columns and deposit itself into dh prebn. So what we want is this thing scaled by 1 over n. Let me put the constant up front here.

So scale down the gradient. And now we need to replicate it across all the rows here. So I like to do that by torch.oncelike of basically h prebn.

And I will let the broadcasting do the work of replication. So like that. So this is dh prebn.

And hopefully we can plus equals that.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)



(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

So this here is broadcasting, and then this is the scaling. So this should be correct. Okay.

So that completes the backpropagation of the Bastrom layer, and we are now here. Let's backpropagate through the linear layer one here. Now, because everything is getting a little vertically crazy, I copy-pasted the line here, and let's just backpropagate through this one line.

So first, of course, we inspect the shapes, and we see that this is 32 by 64. mcat is 32 by 30, w1 is 30 by 64, and b1 is just 64. So as I mentioned, backpropagating through linear layers is fairly easy just by matching the shapes.

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