
我们看到损失看起来是这样的。我们看到训练和验证损失大约在2.16左右。在这里，我对代码进行了一些重构，以便评估任意拆分。因此，你只需传入一个字符串，指定你想要评估的拆分方式。

然后在这里，根据训练集、验证集或测试集，我进行索引并获取正确的分割。接着是网络的前向传播、损失评估和打印输出。这样只是为了让它看起来更美观。你会注意到这里我使用了一个装饰器 `torch.nograd`，你也可以查阅相关文档了解它。简单来说，这个装饰器的作用是告诉 Torch：在这个函数中发生的任何操作都不需要计算梯度。因此，Torch 不会为这些操作记录任何梯度信息，也就不会为后续的反向传播做任何准备工作。

在这里你会注意到我使用了一个装饰器 torch.nograd，你也可以查阅并阅读它的文档。基本上，这个装饰器在函数上的作用是，Torch 会认为这个函数中发生的任何操作都不需要梯度。因此，它不会进行任何为了跟踪所有梯度以准备最终反向传播所需的记录工作。

这就像是这里创建的所有张量的requires_grad属性都被设为False。这样一来效率就大大提高了，因为你是在告诉Torch：我不会对这些计算调用backward()方法，所以你不需要在底层维护计算图。这就是它的作用。

你也可以使用带有 Torch.nograd 的上下文管理器，具体用法可以查阅相关资料。接下来，我们像之前一样从模型中进行采样，只需对神经网络进行一次前向传播，获取概率分布，从中采样，调整上下文窗口，并重复这一过程直到获得特殊的结束标记。可以看到，现在从模型中采样的单词看起来已经顺眼多了。

虽然效果还不算惊艳，这些名字也不完全像真实姓名，但比起我们之前使用的二元模型已经好多了。这就是我们的起点。现在，我首先要仔细研究的是初始化过程。我能看出我们的网络在初始化时配置得非常不当，存在多处问题，但让我们先从第一个开始。看这里的第零次迭代，也就是第一次迭代，我们记录到的损失值是27，然后迅速下降到大约1或2左右。所以我可以断定初始化完全搞砸了，因为这个值实在太高了。

在神经网络训练过程中，通常你都会对初始化时的预期损失值有个大致概念。这个预期值主要取决于损失函数和问题设置。就这个案例而言，27的损失值显然超出预期范围。我预计数字会低得多，我们可以一起计算。基本上在初始化时，我们希望对于任何一个训练样本，接下来可能出现27个字符。在初始化阶段，我们没有理由认为某些字符比其他字符更有可能出现。

因此，我们最初预计得到的概率分布会是一个均匀分布，即对所有27个字符赋予大致相等的概率。所以基本上，我们希望每个字符的概率大约为1/27。这就是我们应该记录的概率。然后损失就是负对数概率。所以我们把这个包装成一个张量，然后可以取它的对数。负对数概率就是我们预期的损失，即3.29，远低于27。

所以现在的情况是，在初始化阶段，神经网络生成的概率分布全都一团糟。有些字符的预测概率极高，有些则极低。本质上就是神经网络错得理直气壮。这就是导致损失值极高的原因。这里用一个更小的四维例子来说明这个问题。假设我们只有四个字符，然后神经网络输出的逻辑值非常非常接近于零。

那么，当我们对所有零取softmax时，得到的概率是一个扩散分布。因此总和为一，且完全均匀。在这种情况下，如果标签是比如说二，实际上标签是二、三、一还是零都无关紧要，因为这是一个均匀分布，我们记录的是完全相同的损失，在这个例子中是1.38。所以这是我们对于一个四维例子所预期的损失。

当然，我能看到，当我们开始调整这些逻辑值时，这里的损失值就会发生变化。所以有可能我们碰巧锁定了一个很高的数值，比如5之类的。那样的话，我们会记录一个很低的损失值，因为初始化时我们恰好给正确的标签分配了正确的概率。

更有可能的是，其他维度的对数概率会很高。接下来会发生的是，我们会开始记录一个更高的损失。可能出现的情况是，对数概率会变成类似这样，它们会呈现极值，而我们记录的损失也会非常高。例如，如果我们有四个Torch.random生成的数，这些是均匀分布的——抱歉，应该是正态分布的数，共四个。这里我们还可以打印出逻辑值、由此产生的概率以及损失值。由于这些逻辑值大部分接近于零，所以产生的损失值是可以接受的。

但假设现在的情况是原来的十倍。你看，因为这些数值更加极端，你猜中正确区间的可能性极低，于是你满怀信心地犯错，并记录下极高的损失值。如果你的逻辑回归输出值更加极端，甚至在初始化阶段就可能出现像无穷大这样极其荒谬的损失值。

所以基本上这样并不好，我们希望网络初始化时logits大致为零。实际上，logits不一定要刚好是零，它们只需要相等即可。例如，如果所有logits都是1，那么由于softmax内部的归一化，结果其实是可以接受的。

但从对称性考虑，我们不希望它是任意正数或负数。我们只希望它全为零，并记录初始化时的预期损失。现在让我们具体看看例子中哪里出了问题。这里我们进行初始化。让我重新初始化神经网络。在这里，让我在第一次迭代后中断。

所以我们只看到了初始损失，也就是27。这个数值实在是太高了。直观来看，现在我们可以预估涉及到的变量了。我们看到这里的logits，如果我们只打印其中一些，比如只打印第一行，会发现这些logits呈现出非常极端的数值。正是这些极端值导致了模型对错误答案产生虚假信心，并使损失值变得非常高。因此这些logits应该更接近于零才对。

那么现在让我们思考一下，如何使这个神经网络输出的逻辑值更接近于零。你可以看到，逻辑值的计算方式是隐藏状态乘以W2再加上B2。首先，目前我们将B2初始化为适当大小的随机值。但由于我们希望初始值大致为零，实际上我们并不想添加随机数的偏置。因此，我在这里添加一个乘以零的项，以确保B2在初始化时基本为零。其次，这是H乘以W2的结果。

因此，如果我们希望逻辑值变得非常非常小，就需要调整W2的权重使其缩小。例如，若将W2所有元素统一缩小为0.1倍，当我再次运行第一轮迭代时，你会发现结果更接近预期值。我们需要的理想值大约是3.29，而当前是4.2。我可以进一步调小参数，现在得到3.32。看，我们正逐步逼近目标值。

现在，你可能在想，我们能不能直接把它设为零？这样在初始化时就能得到我们想要的结果。我通常不这么做的原因是我非常谨慎。我马上会告诉你为什么你不应该把神经网络中的权重W设为零。你通常希望数值较小而不是恰好为零。对于这个特定情况下的输出层，我认为没问题，但我马上会展示如果这样做，问题会迅速出现。所以我们还是用0.01吧。这样的话，我们的损失足够接近，但保留了一些熵。

这并不完全是零。它有一些小的熵，这用于对称性破坏，我们稍后会看到。现在输出的逻辑值更接近于零，一切都很顺利。所以如果我直接删掉这些代码并移除break语句，现在就可以用这个新的初始化方式运行优化程序。让我们看看记录下来的损失值是多少。好的，我这就让它运行起来。

你看到我们一开始表现不错，然后稍微回落了一些。现在的损失曲线不再呈现曲棍球棒形状，因为在曲棍球棒现象中，最初几次迭代的损失变化本质上源于优化过程——它首先压缩逻辑值，然后重新排列它们。我们实际上移除了损失函数中最容易优化的部分，也就是单纯压缩权重的阶段。

因此，一开始我们并没有获得这些容易的收益，我们只是在训练实际的神经网络中获得了一些艰难的收益。所以没有出现曲棍球棒式的增长。但好消息是，首先，初始化的损失符合我们的预期。损失曲线看起来并不像曲棍球棒。这对于你可能训练的任何神经网络都是成立的，也是需要注意的地方。其次，最终的损失实际上有了相当大的改善。

不幸的是，我删除了我们之前在这里的内容。我记得这是2.12，这是2.16。所以我们得到了一个稍微改进的结果。原因在于我们投入了更多的计算周期和时间来优化神经网络，而不是在最初的几千次迭代中仅仅压缩权重，因为在初始化时它们的值一开始就过高。

所以这是需要注意的第一点。现在让我们来看第二个问题。让我重新初始化我们的神经网络，并重新引入break语句。所以我们有一个合理的初始损失。因此，尽管在损失层面上一切看起来都很好，我们得到了预期的结果，但这个神经网络及其初始化内部仍然潜伏着一个更深层次的问题。所以现在逻辑回归的输出是正常的。

现在的问题出在隐藏状态激活值H上。如果我们直接可视化这个向量（即张量H），虽然不太容易看清，但粗略来说问题在于——你会发现大量元素的值都是1或-1。要知道torch.10H这个10H函数是个压缩函数。

它会接收任意数字，并将它们平滑地压缩到负一到正一的范围之内。让我们通过查看H的直方图来更好地理解这个张量内部数值的分布情况。我们可以先这样做。我们可以看到，H 是 32 个样本，每个样本有 200 个激活值。我们可以将其视为负一，以将其展开成一个大的向量。然后我们可以调用 toList 方法，将其转换为一个大的 Python 浮点数列表。

然后我们可以将其传递给plt.hist以生成直方图。我们说要50个分箱，并用分号来抑制大量输出值。我们不需要。因此，我们看到这个直方图，发现绝大多数数值都是负一和一。这说明10H非常非常活跃。我们还可以从根本上探究其原因。我们可以观察输入到10H层的预激活值，并发现这些预激活值的分布范围非常广泛。

这些数值的取值范围在负15到15之间。因此，在10H的折磨测试中，所有数值都会被压缩并限制在负1到1的范围内。这里的许多数值都呈现出非常极端的情况。

现在，如果你对神经网络还不熟悉，可能不会觉得这有什么问题。但如果你精通反向传播的奥秘，并且对这些梯度如何在神经网络中流动有直观的理解，那么当你看到这里10H激活值的分布时，你一定会冒冷汗。让我来告诉你原因。我们必须记住，在反向传播过程中，就像我们在micrograd中看到的那样，我们从损失函数开始进行反向传递，然后通过网络向后流动。

具体来说，我们将通过这个torch.10H进行反向传播。这里的这一层由200个神经元组成，每个神经元对应这些示例中的一个。它实现了一个逐元素的10H。那么让我们看看在反向传播过程中10H会发生什么。

实际上，我们可以回到第一节课中的micrograd代码，看看我们是如何实现tanh函数的。我们看到这里的输入是x，然后我们计算t，也就是x的tanh值。这就是t。t的值在-1到1之间，这是tanh的输出。然后在反向传播过程中，我们如何通过tanh进行反向传播呢？我们取出那个梯度，然后按照链式法则，将其与局部梯度相乘，局部梯度的形式是1减去t的平方。

那么，如果10H的输出非常接近负1或1会发生什么呢？如果你在这里代入t等于1，你会得到0，乘以那个梯度。无论那个梯度是多少，我们都在消除梯度，实际上我们阻止了通过这个10H单元的反向传播。同样地，当t为负1时，这又会变成0，那个梯度就停止了。

直观地说，这很合理，因为这是一个10H神经元。实际情况是，如果它的输出非常接近1，那么我们处于这个10H的尾部。因此，基本上改变输入不会对10H的输出产生太大影响，因为它位于10H的平坦区域。

直观来看，这是合理的，因为这是一个10H神经元。其原理在于：当输出值非常接近1时，我们实际上处于10H函数的尾部区域。在这种情况下，即使改变输入值，也不会对10H的输出产生太大影响，因为它正处于10H函数的平坦区间。

因此，这对损失没有任何影响。实际上，与这个10H神经元相关的权重和偏置不会影响损失，因为这个10H单元的输出处于其平坦区域，没有任何影响。我们可以随意改变它们，而损失不会受到影响。这又是另一种方式来证明梯度确实基本上为零。它消失了。实际上，当t等于0时，我们得到1乘以那个梯度。因此，当10H恰好取值为0时，那个梯度就直接通过了。

简单来说，这个机制的工作原理是：当t等于0时，10H单元处于非活动状态，梯度直接通过。但当你越深入平坦尾部区域，梯度被压缩的程度就越大。实际上你会发现，流经10H的梯度只会不断衰减，其衰减量通过这里的平方关系成比例变化，具体取决于你处于这个10H平坦尾部的深度。

所以这就是这里发生的情况。这里的担忧是，如果所有这些输出H都处于-1和1的平坦区域，那么通过网络流动的梯度就会在这一层被破坏。不过，这里也有一些可取之处，我们可以通过以下方式大致了解这个问题。

我在这里写了一些代码。基本上，我们想做的就是查看H，取其绝对值，并观察它处于平坦区域的频率，比如说大于0.99的情况。最终得到的结果如下。这是一个布尔张量。所以在布尔张量中，如果为真则显示白色，为假则显示黑色。基本上，我们这里展示的是32个样本和200个隐藏神经元。可以看到大部分区域都是白色的。

这告诉我们，所有这些10H神经元都非常非常活跃，而且它们处于平坦的尾部状态。因此，在所有这些情况下，反向梯度都会被破坏。现在，如果这200个神经元中的任何一个出现整列都是白色的情况，那我们就会遇到大麻烦。

因为在这种情况下，我们就会遇到所谓的“死亡神经元”。这可能是一个10H神经元，其权重和偏置的初始化方式可能导致没有任何一个样本能在10H的活跃部分激活它。如果所有样本都落在尾部区域，那么这个神经元就永远学不到东西。

这是一个死神经元。因此，仔细检查并寻找完全白色的列，我们发现情况并非如此。所以我没有看到一个完全是白色的神经元。因此，对于每一个10H神经元来说，我们确实有一些例子可以在10H的活动部分激活它们。所以会有一些梯度流动，这个神经元会学习。神经元会发生变化，会移动，会做一些事情。

但有时你可能会遇到神经元死亡的情况。这种现象的表现形式是：对于一个10H神经元来说，无论你从数据集中输入什么数据，这个10H神经元总是完全激活为1或完全抑制为-1。这样一来它就无法学习了，因为所有的梯度都会归零。

这不仅适用于10H，也适用于神经网络中使用的许多其他非线性函数。因此，我们确实经常使用10H，但sigmoid函数也会有完全相同的问题，因为它是一个压缩神经元。所以sigmoid函数也会有同样的问题，但基本上sigmoid函数实际上也会遇到同样的情况。

同样的情况也适用于ReLU。因此，ReLU在零以下有一个完全平坦的区域。所以，如果你有一个ReLU神经元，那么当它为正时，它就是一个直通。如果预激活为负值，它就会直接关闭。由于该区域完全平坦，因此在反向传播过程中，梯度将被精确归零。所有梯度都会被精确设为零，而不仅仅是一个非常非常小的数值——具体数值取决于t的正负程度。

因此，你可能会得到一个“死亡”的ReLU神经元。死亡ReLU神经元的表现形式是……本质上来说，当一个带有ReLU非线性激活函数的神经元从未被激活时，它就处于这种状态。也就是说，无论你输入数据集中的任何样本，这个神经元都不会被激活，始终处于平坦区域。那么这个ReLU神经元就是一个死亡神经元。它的权重和偏置将永远无法学习，因为它们永远不会获得梯度，因为这个神经元从未被激活过。

这种情况有时会在初始化时发生，因为权重和偏置的设置可能恰好导致某些神经元永远处于"死亡"状态。但在优化过程中也可能出现这种情况。例如，如果学习率设置过高，有时某些神经元会接收到过大的梯度，导致它们偏离数据流形。

然后从那时起，没有任何示例能激活这个神经元，所以这个神经元就永远死掉了。这有点像网络思维中的永久性脑损伤。有时候，如果你的学习率非常高，比如你有一个使用ReLU神经元的神经网络，你训练这个神经网络，最终得到一些损失值。

但实际上你要做的是，遍历整个训练集，对样本进行前向传播，然后你会发现有些神经元从未激活过。这些就是网络中的"死亡神经元"，它们永远不会被激活。通常发生的情况是，在训练过程中，这些ReLU神经元会不断变化、移动等。然后由于某个地方的高梯度，它们偶然被“击倒”，之后再也没有任何东西能激活它们。从那时起，它们就彻底“死”了。

所以这有点像是一些神经元可能会遭受的永久性脑损伤。其他非线性函数，比如Leaky ReLU，就不会有这个问题，因为你可以看到它没有平坦的尾部。你几乎总能得到梯度。ELU也被相当频繁地使用。它也可能存在这个问题，因为它有平坦的部分。所以这是需要注意和关注的事情。

在这种情况下，我们有太多的激活值h达到了极端值。但由于没有白色列的存在，我认为问题不大。事实上，网络经过优化后给出了相当不错的损失值。但这并不是最优选择。尤其是在初始化阶段，这绝对不是你想要的情况。简单来说，流向10h的h预激活过于极端了。

太大了。它在10小时的两侧都造成了过于饱和的分布。这不是你想要的，因为这意味着这些神经元的训练较少，因为它们更新的频率较低。那么我们该如何解决这个问题呢？嗯，h-preactivation是mcat，它来源于c。所以这些都是均匀高斯分布。但随后它会被w1加上b1相乘。而h-preact离零太远，这就是问题的根源。

所以我们希望这种重新激活接近于零，与我们之前对逻辑回归的处理非常相似。因此，在这里，我们实际上希望得到非常非常相似的结果。现在，将偏置设置为一个非常小的数值是可以接受的。我们可以将其乘以001来获得一点熵。我有时喜欢这样做，只是为了在这些10h神经元的初始初始化中增加一点变化和多样性。在实践中我发现这能稍微帮助优化。

然后对于权重，我们也可以直接压缩。那就把所有数值都乘以0.1吧。重新运行第一批数据。现在来看看结果。首先，让我们看看这里。现在你看到了，因为我们将w乘以0.1，得到了一个更好的直方图。这是因为预激活值现在介于负1.5和1.5之间。这样，我们预计白色部分会少得多。

好的，这里没有白色。基本上，这是因为在两个方向上都没有神经元饱和超过0.99。所以这实际上是一个相当不错的状态。也许我们可以稍微提高一点。抱歉，我是在这里改变w1吗？也许我们可以调到0.2。好的，所以像这样的分布可能不错。也许这就是我们初始化应该采用的方式。

那么现在让我把这些擦掉。首先从初始化开始，让我运行完整的优化过程，不中断。让我们看看结果如何。好的，优化完成了。我重新运行了损失函数。这就是我们得到的结果。然后提醒一下，我把之前这节课中看到的所有损失都列了出来。可以看到，我们在这里确实取得了进步。同样提醒一下，刚开始时我们的验证损失是2.17。

通过修正softmax过于自信的错误，我们将误差降至2.13。接着通过解决tanh层过度饱和的问题，误差进一步降至2.10。这种现象的根本原因在于我们的初始化策略更优。因此，我们可以将更多时间用于有效训练，而非无效训练——因为梯度不再被置零。我们无需再耗费计算资源去纠正初始阶段的简单问题（比如softmax的过度自信），也不必反复压缩权重矩阵。

这主要说明了初始化的过程及其对性能的影响，只需了解这些神经网络的内部结构、激活方式和梯度变化即可。现在我们正在处理一个非常小的网络，这只是一个单层的多层感知器。由于网络非常浅，优化问题实际上相当简单且容错性很高。因此，即使我们的初始化非常糟糕，网络最终还是学会了。只是结果稍差一些。

然而，这种情况并不普遍。一旦我们开始使用更深层的网络，比如有50层，情况就会变得复杂得多。而且这些问题会不断累积。因此，如果你的初始化参数设置得足够糟糕，网络实际上可能会陷入根本无法训练的状态。而且网络越深、结构越复杂，对这些错误的容忍度就越低。这绝对是需要警惕、需要仔细检查、需要绘制图表分析、需要谨慎对待的问题。

是的。好的，这对我们来说很有效。但现在的问题是，我们有很多像0.2这样的魔法数字。我是怎么得出这个数字的？如果我有一个有很多层的庞大神经网络，我该如何设置这些数字？显然，没有人会手动做这些事。

实际上，有一些相对原则性的方法来设置这些比例，我现在想向大家介绍一下。让我在这里粘贴一些我准备的代码，以便引发对此的讨论。那么，我在这里所做的是，我们有一些随机输入x，它是从高斯分布中抽取的。

这里有1000个10维的例子。然后我们在这里有一个权重层，也是用高斯分布初始化的，就像我们在这里做的那样。这个隐藏层中的神经元查看10个输入，隐藏层中有200个神经元。接着我们来看这里，就像这个例子中一样，乘法运算，x乘以w，得到这些神经元的预激活值。基本上这里的分析是：假设这些输入是均匀高斯分布，这些权重也是均匀高斯分布。如果我们计算x乘以w（暂时忽略偏置和非线性部分），那么这些高斯分布的均值和标准差会是多少？一开始，输入只是一个标准的高斯分布。

均值为0，标准差为1。标准差再次说明，只是衡量这个高斯分布离散程度的指标。但当我们在这里进行乘法运算后，观察y的直方图时，会发现均值当然保持不变。大约为0，因为这是一个对称操作。

但我们在这里看到标准差已经扩大到了3。输入的标准差本来是1，但现在增长到了3。因此你在直方图中看到的是这个高斯分布正在扩大。我们正在从输入扩展这个高斯分布，而这并不是我们想要的。我们希望大多数神经网络的激活值相对相似，因此整个神经网络大致保持单位高斯分布。那么问题来了，我们该如何调整这些权重w的尺度，以保持这种高斯分布不变？直观来说，如果我在这里将w的元素乘以一个较大的数，比如5，那么这个高斯分布的标准差就会不断增大。现在我们就达到了15。



So basically, these numbers here in the output, y, take on more and more extreme values. But if we scale it down, let's say 0.2, then conversely, this Gaussian is getting smaller and smaller, and it's shrinking. And you can see that the standard deviation is 0.6. And so the question is, what do I multiply by here to exactly preserve the standard deviation to be 1? And it turns out that the correct answer mathematically, when you work out through the variance of this multiplication here, is that you are supposed to divide by the square root of the fan-in. 

The fan-in is basically the number of input elements here, 10. So we are supposed to divide by 10 square root. And this is one way to do the square root. 

You raise it to a power of 0.5. That's the same as doing a square root. So when you divide by the square root of 10, then we see that the output Gaussian, it has exactly standard deviation of 1. Now, unsurprisingly, a number of papers have looked into how to best initialize neural networks. And in the case of multivariate perceptrons, we can have fairly deep networks that have these nonlinearities in between. 

And we want to make sure that the activations are well-behaved, and they don't expand to infinity or shrink all the way to 0. And the question is, how do we initialize the weights so that these activations take on reasonable values throughout the network? Now, one paper that has studied this in quite a bit of detail that is often referenced is this paper by Kaiming et al. called Delving Deep into Rectifiers. Now, in this case, they actually studied convolutional neural networks. 

And they studied especially the ReLU nonlinearity and the pReLU nonlinearity instead of a 10h nonlinearity. But the analysis is very similar. And basically, what happens here is, for them, the ReLU nonlinearity that they care about quite a bit here is a squashing function where all the negative numbers are simply clamped to 0. So the positive numbers are a path through, but everything negative is just set to 0. And because you're basically throwing away half of the distribution, they find in their analysis of the forward activations in the neural net that you have to compensate for that with a gain. 

And so here, they find that basically, when they initialize their weights, they have to do it with a 0-mean Gaussian whose standard deviation is square root of 2 over the Fannin. What we have here is we are initializing the Gaussian with the square root of Fannin. This NL here is the Fannin.

So what we have is square root of 1 over the Fannin because we have the division here. Now, they have to add this factor of 2 because of the ReLU, which basically discards half of the distribution and clamps it at 0. And so that's where you get an initial factor. Now, in addition to that, this paper also studies not just the sort of behavior of the activations in the forward paths of the neural net, but it also studies the backpropagation. 

And we have to make sure that the gradients also are well-behaved because ultimately, they end up updating our parameters. And what they find here through a lot of the analysis that I invite you to read but it's not exactly approachable, what they find is basically, if you properly initialize the forward paths, the backward paths is also approximately initialized up to a constant factor that has to do with the size of the number of hidden neurons in an early and late layer. But basically, they find empirically that this is not a choice that matters too much.

Now, this kind of initialization is also implemented in PyTorch. So if you go to torch.nn.init documentation, you'll find kyming normal. And in my opinion, this is probably the most common way of initializing neural networks now.

And it takes a few keyword arguments here. So number one, it wants to know the mode. Would you like to normalize the activations or would you like to normalize the gradients to be always Gaussian with zero mean and a unit or one standard deviation? And because they find the paper that this doesn't matter too much, most of the people just leave it as the default, which is fan-in. 

And then second, pass in the non-linearity that you are using. Because depending on the non-linearity, we need to calculate a slightly different gain. And so if your non-linearity is just linear, so there's no non-linearity, then the gain here will be one. 

And we have the exact same kind of formula that we've got up here. But if the non-linearity is something else, we're going to get a slightly different gain. And so if we come up here to the top, we see that, for example, in the case of Relu, this gain is a square root of two. 

And the reason it's a square root, because in this paper, you see how the two is inside of the square root. So the gain is a square root of two. In a case of linear or identity, we just get a gain of one. 

In a case of 10H, which is what we're using here, the advised gain is a five over three. And intuitively, why do we need a gain on top of the initialization? It's because 10H, just like Relu, is a contractive transformation. So what that means is you're taking the output distribution from this matrix multiplication, and then you are squashing it in some way.

Now, Relu squashes it by taking everything below zero and clamping it to zero. 10H also squashes it because it's a contractive operation. It will take the tails and it will squeeze them in. 

And so in order to fight the squeezing in, we need to boost the weights a little bit so that we renormalize everything back to unit standard deviation. So that's why there's a little bit of a gain that comes out. Now, I'm skipping through this section a little bit quickly, and I'm doing that actually intentionally. 

And the reason for that is because about seven years ago when this paper was written, you had to actually be extremely careful with the activations and the gradients and their ranges and their histograms. And you had to be very careful with the precise setting of gains and the scrutinizing of the nonlinearities used and so on. And everything was very finicky and very fragile and very properly arranged for the neural net to train, especially if your neural net was very deep. 

But there are a number of modern innovations that have made everything significantly more stable and more well-behaved, and it's become less important to initialize these networks exactly right. And some of those modern innovations, for example, are residual connections, which we will cover in the future, the use of a number of normalization layers, like for example, batch normalization, layer normalization, group normalization. We're going to go into a lot of these as well. 

And number three, much better optimizers, not just stochastic gradient descent, the simple optimizer we're basically using here, but slightly more complex optimizers like RMSProp and especially Adam. And so all of these modern innovations make it less important for you to precisely calibrate the initialization of the neural net. All that being said, in practice, what should we do? In practice, when I initialize these neural nets, I basically just normalize my weights by the square root of the fan-in. 

So basically, roughly what we did here is what I do. Now, if we want to be exactly accurate here, we can go back in it of kind of normal, this is how we would implement it. We want to set the standard deviation to be gain over the square root of fan-in, right? So to set the standard deviation of our weights, we will proceed as follows. 

Basically, when we have Torch.random, and let's say I just create a thousand numbers, we can look at the standard deviation of this, and of course that's one, that's the amount of spread. Let's make this a bit bigger so it's closer to one. So that's the spread of the Gaussian of zero mean and unit standard deviation. 

Now, basically when you take these and you multiply by say 0.2, that basically scales down the Gaussian and that makes it standard deviation 0.2. So basically the number that you multiply by here ends up being the standard deviation of this Gaussian. So here, this is a standard deviation 0.2 Gaussian here when we sample RW1. But we want to set the standard deviation to gain over square root of fan-in, which is fan-in. 

So in other words, we want to multiply by gain, which for 10H is 5 over 3. 5 over 3 is the gain. And then times, or I guess divide square root of the fan-in. And in this example here, the fan-in was 10.

And I just noticed that actually here, the fan-in for W1 is actually an embed times block size, as you will recall is actually 30. And that's because each character is 10-dimensional, but then we have three of them and we concatenate them. So actually the fan-in here was 30 and I should have used 30 here probably. 

But basically we want 30 square root. So this is the number, this is what our standard deviation we want to be. And this number turns out to be 0.3. Whereas here, just by fiddling with it and looking at the distribution and making sure it looks okay, we came up with 0.2. And so instead, what we want to do here is we want to make the standard deviation be 5 over 3, which is our gain, divide this amount times 0.2 square root.

And these brackets here are not that necessary, but I'll just put them here for clarity. This is basically what we want. This is the timing init in our case for 10H nonlinearity.

And this is how we would initialize the neural net. And so we're multiplying by 0.3 instead of multiplying by 0.2. And so we can initialize this way and then we can train the neural net and see what we get. Okay. 

So I trained the neural net and we end up in roughly the same spot. So looking at the validation loss, we now get 2.10. And previously we also had 2.10. There's a little bit of a difference, but that's just the randomness of the process, I suspect. But the big deal, of course, is we get to the same spot, but we did not have to introduce any magic numbers that we got from just looking at histograms and guessing, checking. 

We have something that is semi-principled and will scale us to much bigger networks and something that we can sort of use as a guide. So I mentioned that the precise setting of these initializations is not as important today due to some modern innovations. And I think now is a pretty good time to introduce one of those modern innovations, and that is batch normalization. 

So batch normalization came out in 2015 from a team at Google, and it was an extremely impactful paper because it made it possible to train very deep neural nets quite reliably, and it basically just worked. So here's what batch normalization does and what's implemented. Basically, we have these hidden states, HPREACT, right? And we were talking about how we don't want these pre-activation states to be way too small because then the 10H is not doing anything, but we don't want them to be too large because then the 10H is saturated. 

In fact, we want them to be roughly Gaussian, so zero mean and a unit or one standard deviation, at least at initialization. So the insight from batch normalization paper is, okay, you have these hidden states and you'd like them to be roughly Gaussian, then why not take the hidden states and just normalize them to be Gaussian? And it sounds kind of crazy, but you can just do that because standardizing hidden states so that their unit Gaussian is a perfectly differentiable operation, as we'll soon see. And so that was kind of like the big insight in this paper. 

And when I first read it, my mind was blown because you just normalize these hidden states, and if you'd like unit Gaussian states in your network, at least initialization, you can just normalize them to be unit Gaussian. So let's see how that works. So we're going to scroll to our pre-activations here just before they enter into the 10H.

Now, the idea again is, remember, we're trying to make these roughly Gaussian, and that's because if these are way too small numbers, then the 10H here is kind of inactive. But if these are very large numbers, then the 10H is way too saturated and great in the flow. So we'd like this to be roughly Gaussian. 

So the insight in batch normalization, again, is that we can just standardize these activations so they are exactly Gaussian. So here, HPREACT has a shape of 32 by 200, 32 examples by 200 neurons in the hidden layer. So basically what we can do is we can take HPREACT, and we can just calculate the mean. 

And the mean we want to calculate across the 0th dimension. And we want to also keep them as true so that we can easily broadcast this. So the shape of this is 1 by 200. 

In other words, we are doing the mean over all the elements in the batch. And similarly, we can calculate the standard deviation of these activations. And that will also be 1 by 200. 

Now, in this paper, they have the sort of prescription here. And see, here we are calculating the mean, which is just taking the average value of any neuron's activation. And then their standard deviation is basically kind of like the measure of the spread that we've been using, which is the distance of every one of these values away from the mean, and that squared and averaged. 

That's the variance. And then if you want to take the standard deviation, you would square root the variance to get the standard deviation. So these are the two that we're calculating. 

And now we're going to normalize or standardize these x's by subtracting the mean and dividing by the standard deviation. So basically, we're taking HPreact and we subtract

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

And then we divide by the standard deviation. This is exactly what these two, std and mean, are calculating. Oops, sorry.

This is the mean and this is the variance. You see how the sigma is the standard deviation usually, so this is sigma squared, which the variance is the square of the standard deviation. So this is how you standardize these values.

And what this will do is that every single neuron now, and its firing rate, will be exactly unit Gaussian on these 32 examples, at least, of this batch. That's why it's called batch normalization. We are normalizing these batches.

And then we could, in principle, train this. Notice that calculating the mean and your standard deviation, these are just mathematical formulas. They're perfectly differentiable.

All this is perfectly differentiable and we can just train this. The problem is you actually won't achieve a very good result with this. And the reason for that is we want these to be roughly Gaussian, but only at initialization.

But we don't want these to be forced to be Gaussian always. We'd like to allow the neural net to move this around, to potentially make it more diffuse, to make it more sharp, to make some 10H neurons maybe be more trigger happy or less trigger happy. So we'd like this distribution to move around and we'd like the backpropagation to tell us how the distribution should move around.

And so in addition to this idea of standardizing the activations at any point in the network, we have to also introduce this additional component in the paper here described as scale and shift. And so basically what we're doing is we're taking these normalized inputs and we are additionally scaling them by some gain and offsetting them by some bias to get our final output from this layer. And so what that amounts to is the following.

We are going to allow a batch normalization gain to be initialized at just a once and the once will be in the shape of 1 by n hidden. And then we also will have a bn bias which will be torched at zeros and it will also be of the shape n by 1 by n hidden. And then here the bn gain will multiply this and the bn bias will offset it here.

So because this is initialized to 1 and this to 0, at initialization each neuron's firing values in this batch will be exactly unit Gaussian and will have nice numbers. No matter what the distribution of the HPreact is coming in, coming out it will be unit Gaussian for each neuron and that's roughly what we want at least at initialization. And then during optimization we'll be able to back propagate to bn gain and bn bias and change them so the network is given the full ability to do with this whatever it wants internally.

Here we just have to make sure that we include these in the parameters of the neural net because they will be trained with back propagation. So let's initialize this and then we should be able to train. And then we're going to also copy this line which is the batch normalization layer here on a single line of code and we're going to swing down here and we're also going to do the exact same thing at test time here.

So similar to train time we're going to normalize and then scale and that's going to give us our train and validation loss. And we'll see in a second that we're actually going to change this a little bit but for now I'm going to keep it this way. So I'm just going to wait for this to converge.

Okay so I allowed the neural nets to converge here and when we scroll down we see that our validation loss here is 2.10 roughly which I wrote down here. And we see that this is actually kind of comparable to some of the results that we've achieved previously. Now I'm not actually expecting an improvement in this case and that's because we are dealing with a very simple neural net that has just a single hidden layer.

So in fact in this very simple case of just one hidden layer we were able to actually calculate what the scale of w should be to make these pre-activations already have a roughly Gaussian shape. So the batch normalization is not doing much here. But you might imagine that once you have a much deeper neural net that has lots of different types of operations and there's also for example residual connections which we'll cover and so on it will become basically very very difficult to tune the scales of your weight matrices such that all the activations throughout the neural net are roughly Gaussian.

So that's going to become very quickly intractable. But compared to that it's going to be much much easier to sprinkle batch normalization layers throughout the neural net. So in particular it's common to look at every single linear layer like this one.

This is a linear layer multiplying by a weight matrix and adding a bias. Or for example convolutions which we'll cover later and also perform basically a multiplication with a weight matrix but in a more spatially structured format. It's customary to take this linear layer or convolutional layer and append a batch normalization layer right after it to control the scale of these activations at every point in the neural net.

So we'd be adding these batch normal layers throughout the neural net and then this controls the scale of these activations throughout the neural net. It doesn't require us to do perfect mathematics and care about the activation distributions for all these different types of neural network Lego building blocks that you might want to introduce into your neural net. And it significantly stabilizes the training and that's why these layers are quite popular.

Now the stability offered by batch normalization actually comes at a terrible cost. And that cost is that if you think about what's happening here something terribly strange and unnatural is happening. It used to be that we have a single example feeding into a neural net and then we calculate its activations and its logits and this is a deterministic sort of process so you arrive at some logits for this example.

And then because of efficiency of training we suddenly started to use batches of examples but those batches of examples were processed independently and it was just an efficiency thing. But now suddenly in batch normalization because of the normalization through the batch we are coupling these examples mathematically and in the forward pass and the backward pass of the neural net. So now the hidden state activations HPREACT and your logits for any one input example are not just a function of that example and its input but they're also a function of all the other examples that happen to come for a ride in that batch.

And these examples are sampled randomly. And so what's happening is for example when you look at HPREACT that's going to feed into H the hidden state activations for example for any one of these input examples is going to actually change slightly depending on what other examples there are in the batch. And depending on what other examples happen to come for a ride H is going to change suddenly and it's going to like jitter if you imagine sampling different examples because the statistics of the mean and standard deviation are going to be impacted.

And so you'll get a jitter for H and you'll get a jitter for logits. And you think that this would be a bug or something undesirable but in a very strange way this actually turns out to be good in neural network training. And as a side effect.

And the reason for that is that you can think of this as kind of like a regularizer because what's happening is you have your input and you get your H and then depending on the other examples this is jittering a bit. And so what that does is that it's effectively padding out any one of these input examples and it's introducing a little bit of entropy and because of the padding out it's actually kind of like a form of data augmentation which we'll cover in the future and it's kind of like augmenting the input a little bit and it's jittering it and that makes it harder for the neural net to overfit these concrete specific examples. So by introducing all this noise it actually like pads out the examples and it regularizes the neural net.

And that's one of the reasons why deceivingly as a second order effect this is actually a regularizer and that has made it harder for us to remove the use of batch normalization because basically no one likes this property that the examples in the batch are coupled mathematically and in the forward pass and it leads to all kinds of like strange results we'll go into some of that in a second as well and it leads to a lot of bugs and so on. And so no one likes this property and so people have tried to deprecate the use of batch normalization and move to other normalization techniques that do not couple the examples of a batch. Examples are layer normalization, instance normalization, group normalization and so on and we'll cover some of these later.

But basically long story short batch normalization was the first kind of normalization layer to be introduced. It worked extremely well. It happens to have this regularizing effect.

It stabilized training and people have been trying to remove it and move to some of the other normalization techniques but it's been hard because it just works quite well and some of the reason that it works quite well is again because of this regularizing effect and because it is quite effective at controlling the activations and their distributions. So that's kind of like the brief story of batch normalization and I'd like to show you one of the other weird sort of outcomes of this coupling. So here's one of the strange outcomes that I only glossed over previously when I was evaluating the loss on the validation set.

Basically once we've trained a neural net we'd like to deploy it in some kind of a setting and we'd like to be able to feed in a single individual example and get a prediction out from our neural net. But how do we do that when our neural net now in a forward pass estimates the statistics of the mean understanding deviation of a batch? The neural net expects batches as an input now. So how do we feed in a single example and get sensible results out? And so the proposal in the batch normalization paper is the following.

What we would like to do here is we would like to basically have a step after training that calculates and sets the batch norm mean and standard deviation a single time over the training set. And so I wrote this code here in interest of time and we're going to call what's called calibrate the batch norm statistics. And basically what we do is torch.nograd telling PyTorch that none of this we will call a dot backward on and it's going to be a bit more efficient.

We're going to take the training set get the pre-activations for every single training example and then one single time estimate the mean and standard deviation over the entire training set. And then we're going to get b and mean and b and standard deviation and now these are fixed numbers estimating over the entire training set. And here instead of estimating it dynamically we are going to instead here use b and mean and here we're just going to use b and standard deviation.

And so at test time we are going to fix these clamp them and use them during inference. And now you see that we get basically identical result but the benefit that we've gained is that we can now also forward a single example because the mean and standard deviation are now fixed sort of tensors. That said, nobody actually wants to estimate this mean and standard deviation as a second stage after neural network training because everyone is lazy.

And so this batch normalization paper actually introduced one more idea which is that we can estimate the mean and standard deviation in a running manner during training of the neural net. And then we can simply just have a single stage of training and on the side of that training we are estimating the running mean and standard deviation. So let's see what that would look like.

Let me basically take the mean here that we are estimating on the batch and let me call this b and mean on the ith iteration. And then here this is b and std. b and std at i, okay? And the mean comes here and the std comes here.

So, so far I've done nothing. I've just moved around and I created these extra variables for the mean and standard deviation and I've put them here. So, so far nothing has changed but what we're going to do now is we're going to keep a running mean of both of these values during training.

So let me swing up here and let me create a bn mean underscore running and I'm going to initialize it at zeros and then bn std running which I'll initialize at ones. Because in the beginning because of the way we initialized w1 and b1 hpreact will be roughly unit Gaussian. So the mean will be roughly zero and the standard deviation roughly one.

So I'm going to initialize these. But then here I'm going to update these. And in PyTorch these mean and standard deviation that are running they're not actually part of the gradient-based optimization.

We're never going to derive gradients with respect to them. They're updated on the side of training. And so what we're going to do here is we're going to say with torch.nograd telling PyTorch that the update here is not supposed to be building out a graph because there will be no dot backward.

But this running mean is basically going to be 0.999 times the current value plus 0.001 times this value, this new mean. And in the same way bn std running will be mostly what it used to be. But it will receive a small update in the direction of what the current standard deviation is.

And as you're seeing here this update is outside and on the side of the gradient-based optimization. And it's simply being updated not using gradient descent. It's just being updated using a janky like smooth sort of running mean manner.

And so while the network is training and these pre-activations are sort of changing and shifting around during backpropagation we are keeping track of the typical mean and standard deviation and we're estimating them once. And when I run this now I'm keeping track of this in a running manner. And what we're hoping for of course is that the bn mean underscore running and bn mean underscore std are going to be very similar to the ones that we've calculated here before.

And that way we don't need a second stage because we've sort of combined the two stages and we've put them on the side of each other if you want to look at it that way. And this is how this is also implemented in the batch normalization layer in PyTorch. So during training the exact same thing will happen and then later when you're using inference it will use the estimated running mean of both the mean and standard deviation of those hidden states.

So let's wait for the optimization to converge and hopefully the running mean and standard deviation are roughly equal to these two and then we can simply use it here and we don't need this stage of explicit calibration at the end. Okay so the optimization finished. I'll rerun the explicit estimation and then the bn mean from the explicit estimation is here and bn mean from the running estimation during the optimization you can see is very very similar.

It's not identical but it's pretty close. And in the same way bn std is this and bn std running is this. As you can see that once again they are fairly similar values not identical but pretty close.

And so then here instead of bn mean we can use the bn mean running instead of bn std we can use bn std running and hopefully the validation loss will not be impacted too much. Okay so basically identical and this way we've eliminated the need for this explicit stage of calibration because we are doing it inline over here. Okay so we're almost done with batch normalization.

There are only two more notes that I'd like to make. Number one I've skipped a discussion over what is this plus epsilon doing here. This epsilon is usually like some small fixed number for example 1e negative 5 by default and what it's doing is that it's basically preventing a division by zero in the case that the variance over your batch is exactly zero.

In that case here we normally have a division by zero but because of the plus epsilon this is going to become a small number in the denominator instead things will be more well behaved. So feel free to also add a plus epsilon here of a very small number. It doesn't actually substantially change the result.

I'm going to skip it in our case just because this is unlikely to happen in our very simple example here. And the second thing I want you to notice is that we're being wasteful here and it's very subtle but right here where we are adding the bias into HPREACT these biases now are actually useless because we're adding them to the HPREACT but then we are calculating the mean for every one of these neurons and subtracting it. So whatever bias you add here is going to get subtracted right here and so these biases are not doing anything.

In fact they're being subtracted out and they don't impact the rest of the calculation. So if you look at b1.grad it's actually going to be zero because it's being subtracted out and doesn't actually have any effect. And so whenever you're using batch normalization layers then if you have any weight layers before like a linear or a conv or something like that you're better off coming here and just like not using bias.

So you don't want to use bias and then here you don't want to add it because that's spurious. Instead we have this batch normalization bias here and that batch normalization bias is now in charge of the biasing of this distribution instead of this b1 that we had here originally. And so basically the batch normalization layer has its own bias and there's no need to have a bias in the layer before it because that bias is going to be subtracted out anyway.

So that's the other small detail to be careful with. Sometimes it's not going to do anything catastrophic. This b1 will just be useless.

It will never get any gradient. It will not learn. It will stay constant and it's just wasteful but it doesn't actually really impact anything otherwise.

Okay so I rearranged the code a little bit with comments and I just wanted to give a very quick summary of the batch normalization layer. We are using batch normalization to control the statistics of activations in the neural net. It is common to sprinkle batch normalization layer across the neural net and usually we will place it after layers that have multiplications like for example a linear layer or a convolutional layer which we may cover in the future.

Now the batch normalization internally has parameters for the gain and the bias and these are trained using backpropagation. It also has two buffers. The buffers are the mean and the standard deviation, the running mean and the running mean of the standard deviation.

And these are not trained using backpropagation. These are trained using this janky update of kind of like a running mean update. So these are sort of the parameters and the buffers of batch normalization layer.

And then really what it's doing is it's calculating the mean and the standard deviation of the activations that are feeding into the batch normalization layer over that batch. Then it's centering that batch to be unit Gaussian and then it's offsetting and scaling it by the learned bias and gain. And then on top of that it's keeping track of the mean and standard deviation of the inputs and it's maintaining this running mean and standard deviation.

And this will later be used at inference so that we don't have to re-estimate the mean and standard deviation all the time. And in addition that allows us to basically forward individual examples at test time. So that's the batch normalization layer.

It's a fairly complicated layer but this is what it's doing internally. Now I wanted to show you a little bit of a real example. So you can search ResNet which is a residual neural network and these are common types of neural networks used for image classification.

And of course we haven't covered ResNets in detail so I'm not going to explain all the pieces of it. But for now just note that the image feeds into a ResNet on the top here and there's many many layers with repeating structure all the way to predictions of what's inside that image. This repeating structure is made up of these blocks and these blocks are just sequentially stacked up in this deep neural network.

Now the code for this, the block basically that's used and repeated sequentially in series is called this bottleneck block. And there's a lot here. This is all PyTorch and of course we haven't covered all of it but I want to point out some small pieces of it.

Here in the init is where we initialize the neural net. So this code block here is basically the kind of stuff we're doing here. We're initializing all the layers.

And in the forward we are specifying how the neural net acts once you actually have the input. So this code here is along the lines of what we're doing here. And now these blocks are replicated and stacked up serially and that's what a residual network would be.

And so notice what's happening here. Conv1, these are convolution layers. And these convolution layers basically they're the same thing as a linear layer except convolution layers don't apply, convolution layers are used for images and so they have spatial structure.

And basically this linear multiplication and bias offset are done on patches instead of a map, instead of the full input. So because these images have structure, spatial structure, convolutions just basically do wx plus b but they do it on overlapping patches of the input. But otherwise it's wx plus b. Then we have the normal layer which by default here is initialized to be a BatchNorm in 2D, so a two-dimensional BatchNormalization layer.

And then we have a nonlinearity like ReLU. So instead of, here they use ReLU, we are using tanh in this case. But both are just nonlinearities and you can just use them relatively interchangeably.

For very deep networks, ReLUs typically empirically work a bit better. So see the motif that's being repeated here. We have convolution, BatchNormalization, ReLU, convolution, BatchNormalization, ReLU, etc.

And then here this is a residual connection that we haven't covered yet. But basically that's the exact same pattern we have here. We have a weight layer like a convolution or like a linear layer, BatchNormalization and then tanh which is a nonlinearity.

But basically a weight layer, a normalization layer and nonlinearity. And that's the motif that you would be stacking up when you create these deep neural networks. Exactly as it's done here.

And one more thing I'd like you to notice is that here when they are initializing the conv layers, like conv1x1, the depth for that is right here. And so it's initializing an nn.conv2d which is a convolution layer in PyTorch. And there's a bunch of keyword arguments here that I'm not going to explain yet.

But you see how there's bias equals false. The bias equals false is exactly for the same reason as bias is not used in our case. You see how I erased the use of bias.

And the use of bias is spurious because after this weight layer, there's a BatchNormalization. And the BatchNormalization subtracts that bias and then has its own bias. So there's no need to introduce these spurious parameters.

It wouldn't hurt performance. It's just useless. And so because they have this motif of conv, BatchNormalization, they don't need a bias here because there's a bias inside here.

By the way, this example here is very easy to find. Just do ResNetPyTorch and it's this example here. So this is kind of like the stock implementation of a residual neural network in PyTorch.

And you can find that here. But of course, I haven't covered many of these parts yet. And I would also like to briefly descend into the definitions of these PyTorch layers and the parameters that they take.

Now, instead of a convolutional layer, we're going to look at a linear layer because that's the one that we're using here. This is a linear layer. And I haven't covered convolutions yet.

But as I mentioned, convolutions are basically linear layers except on patches. So a linear layer performs a wx plus b, except here they're calling the w a transpose. So it calculates wx plus b very much like we did here.

To initialize this layer, you need to know the fan in, the fan out. And that's so that they can initialize this w. This is the fan in and the fan out. So they know how big the weight matrix should be.

You need to also pass in whether or not you want a bias. And if you set it to false, then no bias will be inside this layer. And you may want to do that exactly like in our case, if your layer is followed by a normalization layer such as batch norm.

So this allows you to basically disable bias. Now, in terms of the initialization, if we swing down here, this is reporting the variables used inside this linear layer. And our linear layer here has two parameters, the weight and the bias.

In the same way, they have a weight and a bias. And they're talking about how they initialize it by default. So by default, PyTorch will initialize your weights by taking the fan in and then doing 1 over fan in square root.

And then instead of a normal distribution, they are using a uniform distribution. So it's very much the same thing. But they are using a 1 instead of 5 over 3. So there's no gain being calculated here.

The gain is just 1. But otherwise, it's exactly 1 over the square root of fan in, exactly as we have here. So 1 over the square root of k is the scale of the weights. But when they are drawing the numbers, they're not using a Gaussian by default.

They're using a uniform distribution by default. And so they draw uniformly from negative square root of k to square root of k. But it's the exact same thing and the same motivation with respect to what we've seen in this lecture. And the reason they're doing this is if you have a roughly Gaussian input, this will ensure that out of this layer, you will have a roughly Gaussian output.

And you basically achieve that by scaling the weights by 1 over the square root of fan in. So that's what this is doing. And then the second thing is the batch normalization layer.

So let's look at what that looks like in PyTorch. So here we have a one-dimensional batch normalization layer, exactly as we are using here. And there are a number of keyword arguments going into it as well.

So we need to know the number of features. For us, that is 200. And that is needed so that we can initialize these parameters here.

The gain, the bias, and the buffers for the running mean and standard deviation. Then they need to know the value of epsilon here. And by default, this is 1 negative 5. You don't typically change this too much.

Then they need to know the momentum. And the momentum here, as they explain, is basically used for these running mean and running standard deviation. So by default, the momentum here is 0.1. The momentum we are using here in this example is 0.001. And basically, you may want to change this sometimes.

And roughly speaking, if you have a very large batch size, then typically what you'll see is that when you estimate the mean and the standard deviation for every single batch size, if it's large enough, you're going to get roughly the same result. And so therefore, you can use slightly higher momentum, like 0.1. But for a batch size as small as 32, the mean and the standard deviation here might take on slightly different numbers because there's only 32 examples we are using to estimate the mean and standard deviation. So the value is changing around a lot.

And if your momentum is 0.1, that might not be good enough for this value to settle and converge to the actual mean and standard deviation over the entire training set. And so basically, if your batch size is very small, momentum of 0.1 is potentially dangerous. And it might make it so that the running mean and standard deviation is thrashing too much during training.

And it's not actually converging properly. Affine equals true determines whether this batch normalization layer has these learnable affine parameters, the gain and the bias. And this is almost always kept to true.

I'm not actually sure why you would want to change this to false. Then track running stats is determining whether or not batch normalization layer of PyTorch will be doing this. And one reason you may want to skip the running stats is because you may want to, for example, estimate them at the end as a stage two, like this.

And in that case, you don't want the batch normalization layer to be doing all this extra compute that you're not going to use. And finally, we need to know which device we're going to run this batch normalization on, a CPU or a GPU, and what the data type should be, half precision, single precision, double precision, and so on. So that's the batch normalization layer.

Otherwise, they link to the paper. It's the same formula we've implemented. And everything is the same, exactly as we've done here.

OK, so that's everything that I wanted to cover for this lecture. Really, what I wanted to talk about is the importance of understanding the activations and the gradients and their statistics in neural networks. And this becomes increasingly important, especially as

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Bigger, larger, and deeper. We looked at the distributions basically at the output layer, and we saw that if you have two confident mispredictions, because the activations are too messed up at the last layer, you can end up with these hockey stick losses. And if you fix this, you get a better loss at the end of training, because your training is not doing wasteful work.

Then we also saw that we need to control the activations. We don't want them to squash to zero or explode to infinity. And because of that, you can run into a lot of trouble with all of these nonlinearities in these neural nets.

And basically, you want everything to be fairly homogeneous throughout the neural net. You want roughly Gaussian activations throughout the neural net. Then we talked about, okay, if we want roughly Gaussian activations, how do we scale these weight matrices and biases during initialization of the neural net, so that we don't get, you know, so everything is as controlled as possible.

So that gave us a large boost and improvement. And then I talked about how that strategy is not actually possible for much, much deeper neural nets. Because when you have much deeper neural nets with lots of different types of layers, it becomes really, really hard to precisely set the weights and the biases in such a way that the activations are roughly uniform throughout the neural net.

So then I introduced the notion of a normalization layer. Now, there are many normalization layers that people use in practice. Batch normalization, layer normalization, instance normalization, group normalization.

We haven't covered most of them, but I've introduced the first one. And also the one that I believe came out first, and that's called batch normalization. And we saw how batch normalization works.

This is a layer that you can sprinkle throughout your deep neural net. And the basic idea is if you want roughly Gaussian activations, well, then take your activations and take the mean and the standard deviation and center your data. And you can do that because the centering operation is differentiable.

On top of that, we actually had to add a lot of bells and whistles. And that gave you a sense of the complexities of the batch normalization layer. Because now we're centering the data, that's great.

But suddenly we need the gain and the bias. And now those are trainable. And then because we are coupling all the training examples, now suddenly the question is, how do you do the inference? Where to do the inference, we need to now estimate these mean and standard deviation once over the entire training set, and then use those at inference.

But then no one likes to do stage two. So instead, we fold everything into the batch normalization layer during training and try to estimate these in a running manner so that everything is a bit simpler. And that gives us the batch normalization layer.

And as I mentioned, no one likes this layer. It causes a huge amount of bugs. And intuitively, it's because it is coupling examples in the forward pass of the neural net.

And I've shot myself in the foot with this layer over and over again in my life. And I don't want you to suffer the same. So basically, try to avoid it as much as possible.

Some of the other alternatives to these layers are, for example, group normalization or layer normalization. And those have become more common in more recent deep learning. But we haven't covered those yet.

But definitely, batch normalization was very influential at the time when it came out in roughly 2015, because it was kind of the first time that you could train reliably much deeper neural nets. And fundamentally, the reason for that is because this layer was very effective at controlling the statistics of the activations in a neural net. So that's the story so far.

And that's all I wanted to cover. And in the future lectures, hopefully, we can start going into recurrent neural nets. And recurrent neural nets, as we'll see, are just very, very deep networks, because you unroll the loop when you actually optimize these neural nets.

And that's where a lot of this analysis around the activation statistics and all these normalization layers will become very, very important for good performance. So we'll see that next time. Bye.

Okay, so I lied. I would like us to do one more summary here as a bonus. And I think it's useful to have one more summary of everything I've presented in this lecture.

But also, I would like us to start by torchifying our code a little bit, so it looks much more like what you would encounter in PyTorch. So you'll see that I will structure our code into these modules, like a linear module and a batch norm module. And I'm putting the code inside these modules so that we can construct neural networks very much like we would construct them in PyTorch.

And I will go through this in detail. So we'll create our neural net. Then we will do the optimization loop, as we did before.

And then one more thing that I want to do here is I want to look at the activation statistics, both in the forward pass and in the backward pass. And then here we have the evaluation and sampling, just like before. So let me rewind all the way up here and go a little bit slower.

So here I am creating a linear layer. You'll notice that torch.nn has lots of different types of layers. And one of those layers is the linear layer.

Torch.nn.linear takes a number of input features, output features, whether or not we should have bias, and then the device that we want to place this layer on, and the data type. So I will omit these two, but otherwise we have the exact same thing. We have the fan-in, which is the number of inputs, fan-out, the number of outputs, and whether or not we want to use a bias.

And internally inside this layer, there's a weight and a bias, if you like it. It is typical to initialize the weight using, say, random numbers drawn from a Gaussian. And then here's the coming initialization that we've discussed already in this lecture.

And that's a good default, and also the default that I believe PyTorch uses. And by default, the bias is usually initialized to zeros. Now, when you call this module, this will basically calculate w times x plus b, if you have nb.

And then when you also call .parameters on this module, it will return the tensors that are the parameters of this layer. Now, next we have the batch normalization layer. So I've written that here.

And this is very similar to PyTorch's nn.batchnorm1d layer, as shown here. So I'm kind of taking these three parameters here, the dimensionality, the epsilon that we'll use in the division, and the momentum that we will use in keeping track of these running stats, the running mean, and the running variance. Now, PyTorch actually takes quite a few more things, but I'm assuming some of their settings.

So for us, affine will be true. That means that we will be using a gamma and beta after the normalization. The track running stats will be true.

So we will be keeping track of the running mean and the running variance in the batch norm. Our device by default is the CPU, and the data type by default is float32. So those are the defaults.

Otherwise, we are taking all the same parameters in this batch norm layer. So first, I'm just saving them. Now, here's something new.

There's a .training, which by default is true. And PyTorch nn modules also have this attribute, .training. And that's because many modules, and batch norm is included in that, have a different behavior, whether you are training your neural net or whether you are running it in an evaluation mode and calculating your evaluation loss or using it for inference on some test examples. And batch norm is an example of this, because when we are training, we are going to be using the mean and the variance estimated from the current batch.

But during inference, we are using the running mean and running variance. And so also, if we are training, we are updating mean and variance. But if we are testing, then these are not being updated.

They're kept fixed. And so this flag is necessary and by default true, just like in PyTorch. Now, the parameters of batch norm 1D are the gamma and the beta here.

And then the running mean and the running variance are called buffers in PyTorch nomenclature. And these buffers are trained using exponential moving average here explicitly. And they are not part of the backpropagation of stochastic gradient descent.

So they are not parameters of this layer. And that's why when we have parameters here, we only return gamma and beta. We do not return the mean and the variance.

This is trained sort of like internally here, every forward pass using exponential moving average. So that's the initialization. Now, in a forward pass, if we are training, then we use the mean and the variance estimated by the batch.

Let me pull up the paper here. We calculate the mean and the variance. Now, up above, I was estimating the standard deviation and keeping track of the standard deviation here in the running standard deviation instead of running variance.

But let's follow the paper exactly. Here, they calculate the variance, which is the standard deviation squared. And that's what's kept track of in the running variance instead of a running standard deviation.

But those two would be very, very similar, I believe. If we are not training, then we use the running mean and variance. We normalize.

And then here, I'm calculating the output of this layer. And I'm also assigning it to an attribute called dot out. Now, dot out is something that I'm using in our modules here.

This is not what you would find in PyTorch. We are slightly deviating from it. I'm creating a dot out because I would like to very easily maintain all those variables so that we can create statistics of them and plot them.

But PyTorch and modules will not have a dot out attribute. And finally, here, we are updating the buffers using, again, as I mentioned, exponential moving average, given the provided momentum. And importantly, you'll notice that I'm using the torch.nograd context manager.

And I'm doing this because if we don't use this, then PyTorch will start building out an entire computational graph out of these tensors, because it is expecting that we will eventually call it dot backward. But we are never going to be calling dot backward on anything that includes running mean and running variance. So that's why we need to use this context manager, so that we are not sort of maintaining and using all this additional memory.

So this will make it more efficient. And it's just telling PyTorch that they're running no backward. We just have a bunch of tensors.

We want to update them. That's it. And then we return.

Okay, now scrolling down, we have the 10H layer. This is very, very similar to torch.10H. And it doesn't do too much. It just calculates 10H, as you might expect.

So that's torch.10H. And there's no parameters in this layer. But because these are layers, it now becomes very easy to sort of like stack them up into basically just a list. And we can do all the initializations that we're used to.

So we have the initial sort of embedding matrix. We have our layers, and we can call them sequentially. And then again, with torch.nograd, there's some initializations here.

So we want to make the output softmax a bit less confident, like we saw. And in addition to that, because we are using a six-layer multi-layer perceptron here, so you see how I'm stacking linear, 10H, linear, 10H, et cetera, I'm going to be using the game here. And I'm going to play with this in a second.

So you'll see how when we change this, what happens to the statistics. Finally, the parameters are basically the embedding matrix and all the parameters in all the layers. And notice here, I'm using a double list comprehension, if you want to call it that.

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

