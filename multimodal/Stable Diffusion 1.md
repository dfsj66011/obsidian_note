
那么什么是稳定扩散呢？稳定扩散模型是在 2022 年引入的，也就是去年年底，我记得，由德国慕尼黑路德维希-马克西米利安大学的会议组提出，并且它是开源的，预训练的权重可以在互联网上找到，它变得非常有名，因为人们开始用它做很多事情，构建项目和产品，使用稳定扩散技术，稳定扩散最简单的应用之一是进行文本到图像的转换，也就是说，给定一个提示，我们想要生成一张图像，我们还将看到图像到图像的工作原理，以及如何进行图像修复，图像到图像意味着你已经有一张图片，例如一张狗的图片，你希望通过使用提示来稍微改变它，比如，你想让模型给狗加上翅膀，让它看起来像一只飞翔的狗，而图像修复意味着你移除图像的某些部分，例如，你可以移除，我不知道这里的这部分，然后你要求模型用其他合理的，与图像一致的部分来替换它，我们还将看到这是如何工作的，

让我们深入了解生成模型，因为扩散模型就是生成模型，但什么是生成模型呢？生成模型学习数据的概率分布，这样我们就可以从该分布中采样，生成新的数据实例，例如，如果我们有很多猫或狗的图片，或者其他任何东西的图片，我们可以在这个数据集上训练一个生成模型，然后从该分布中采样，生成新的猫或狗的图片，或者其他任何东西的图片，而这正是我们在稳定扩散中所做的事情，我们实际上有很多图像，我们在大量图像上进行训练，然后从该分布中采样，生成训练集中不存在的新图像，但你可能会有疑问，为什么要将数据建模为分布，即概率分布呢？好吧，让我给你举个例子，想象一下，你是一个罪犯，想要生成成千上万的假身份，想象我们生活在一个非常简单的世界里，每个假身份由代表个人特征的变量组成，比如年龄和身高，假设我们只有两个变量来构成一个人，那就是这个人的年龄和身高，在我的例子中，我将使用厘米作为身高单位，我认为美国人可以将其转换为英尺，那么，如果我们是一个有此目标的罪犯，该如何进行呢？好吧，我们可以向政府的统计部门索取一些关于人口年龄和身高的统计数据，这些信息你可以在网上轻松找到，例如，然后我们可以从这个分布中采样，例如，如果我们将人口的年龄建模为一个均值为 40，方差为 30 的高斯分布，好吧，这些数字是编造的，我不知道它们是否反映了现实，而身高的均值为 120 厘米，方差为 100,我们得到这两个分布，然后我们可以从这两个分布中采样, 生成一个假身份，从分布中采样是什么意思，从这种分布中采样意味着投掷一枚硬币, 一枚非常特殊的硬币，它有很大概率落在这个区域，较小概率落在这个区域，更小概率落在这个区域, 几乎零概率落在这个区域。

所以想象一下，我们掷这枚硬币一次，比如为了年龄，它落在这里，所以它是有可能的，不是非常有可能，但相当有可能，所以假设年龄是 3，让我写下来，年龄，比如说，是 3、然后我们再掷一次这枚硬币，硬币落在了，比如说这里，100, 我们说30, 身高, 130厘米，所以, 如你所见、年龄和身高的组合在现实中是非常不可能的,我是说, 不, 三岁孩子身高一米三十, 我是说, 至少不是我认识的那些. 所以这个年龄和身高的组合非常,呃，不可信，所以为了产生可信的组合，我们实际上需要将这两个变量，即年龄和身高，不作为独立变量分别采样，而是作为一个联合分布来建模，通常我们这样表示联合分布，其中每个年龄和身高的组合都有一个与之相关的概率分数，从这种分布中，我们只用一枚硬币采样，例如，这枚硬币有很大概率落在这个区域，较小概率落在这个区域，几乎零概率落在这个区域，假设我们掷硬币，它最终落在这个区域，以获得相应的

假设这是年龄，这是我们得到的相应身高，我们只需要这样做，并假设这些实际上是真实的身高和真实的身高，现在这里的数字实际上并不匹配，但你明白了我们需要对所有变量建模，这实际上也是我们对图像所做的，对于我们的图像，我们创建了一个非常复杂的分布，其中每个像素都是一个分布，所有像素的整体是一个大的联合分布，一旦我们有了联合分布，我们就可以做很多有趣的事情，例如，我们可以边缘化，所以，例如，想象我们有一个关于年龄和身高的联合分布，所以让我们称年龄为 x，让我们称身高为 y，所以如果我们有一个联合分布，这意味着有 p(x,y) 它为每个 x 和 y 的组合定义，我们总是可以通过边缘化另一个变量来计算 p(x)，即单个变量的概率，通过积分 p(x,y) 和 dy，这就是我们边缘化的方式，这意味着对所有可能的 y 进行边缘化，然后我们还可以计算条件概率，例如，我们可以问，在身高超过一米的情况下，年龄在 0 到 3 岁之间的概率是多少？像这样的问题，我们可以通过使用条件概率来进行这种查询，所以这实际上就是我们在生成模型中所做的，我们将我们的数据建模为一个非常大的联合分布，然后我们学习这个分布的参数，因为它是一个非常复杂的分布，所以我们让神经网络学习这个分布的参数.我们的目标当然是学习这个非常复杂的分布, 然后从中采样以生成新数据, 就像之前的罪犯想要通过建模代表一个人身份的非常复杂的分布来生成新的假身份一样.

在我们的例子中, 我们将我们的系统建模为一个联合分布,同时包含一些潜在变量，所以让我描述一下，正如你可能熟悉扩散模型:我们有两个过程.一个称为前向过程, 另一个称为反向过程，前向过程意味着我们有个初始图像，我们称之为 x0, 就是这里这个, 我们向它添加噪声以得到另一张图像, 它与前一张相同，但上面有一些噪声，然后我们取这张带有一点噪声的图像，生成一张与前一张相同，但噪声更多的图像，所以, 正如你所见, 这张图像的噪声更多, 以此类推,直到我们到达最后一个潜在变量 zt，其中 t 等于 1000，这时它变成了完全的噪声，纯噪声! 实际上是 N(0,1), 因为我们处于多元世界，我们的目标实际上是这个过程，这个前向过程是固定的，所以我们定义了如何根据前一张图像构建每张图像的噪声版本，所以我们知道如何添加噪声，并且我们有一个特定的公式, 一个解析公式，来说明如何向图像添加噪声，问题是我们没有解析公试来逆转这个过程，所以我们不知道如何取这张图像并直接去除噪声，没有封闭的公式来实现这一点。

我们学习和训练一个神经网络来执行这个逆过程, 从有噪声的东西中去除噪声，如果你考虑一下，给某物添加噪声比从某物中去除噪声要容易得多，这就是我们为此目的使用神经网络的原因，

现在我们需要深入数学内部, 因为我们将不仅用它来编写代码, 还要编写采样器, 而在采样器中, 一切都与数学有关，我会尽量简化它、所以不要害怕，那么, 让我们开始吧.

好的, 这是来自 DDPM 论文，即 2020 年 Ho 的去噪扩散概率模型，这里我们有两个过程，第一个是前向过程, 这意味着给定原始图像, 我如何生成时间步 t 处的图像的噪声版本? 在这种情况下，实际上，这是联合分布，让我们看看这里这个，这意味着如果我有时间步 t-1 处的图像，我如何获得下一个时间步，即这张图像的更多噪声版本?

好吧, 我们将其定义为以均值为中心的高斯分布, 均值以先前的值为中心，方差由这里的 beta 参数定义，这里的beta参数由我们决定, 它表示在这个噪声化过程中每一步我们想要添加多少噪声，这也被称为噪声化的马尔可夫链, 因为每个变量都以前一个变量为条件，所以要得到 Xt, 我们需要有 Xt-1，正如你从这里看到的, 我们从x0开始, 然后到x1，这里我称之为z1以示区分，但实际上X1等于 Z1，所以 X0是原始图像, 所有接下来的x 都是噪声版本,其中 xt 是最噪声的，所以这被称为噪声化的马尔可夫链, 我们可以这样做，

所以它被我们定义为一个过程, 这是一个添加噪声的高斯序列， 这里有一个有趣的公式，这是一个闭环的封闭公式, 可以从原始图像直接到任意时间步 t 的图像，而不需要计算所有中间图像，使用这种特定的参数化方法，我们可以从原始图像直接到时间步 t 的图像，通过从这个分布中采样，通过这样定义分布，所以有了这个均值和这个方差, 这里的均值依赖于一个参数, alpha, alpha bar, 它实际上依赖于 beta, 所以我们知道它, 不需要学习，而且方差实际上也依赖于 alpha，alpha 是 beta 的函数，所以 beta 也是我们知道的, 这里没有需要学习的参数.

现在让我们看看反向过程, 反向过程意味着我们有一些噪声, 我们想要得到更少噪声的东西, 所以我们想要去除噪声，我们也将其定义为一个高斯分布，带有均值 mu theta 和方差 sigma theta，现在，这个均值和这个方差对我们来说是未知的，我们必须学习它们，我们将使用神经网络来学习这两个参数，实际上，方差我们也会设定为固定的，我们将以这样的方式参数化，使得这个方差实际上是固定的，所以我们假设我们已经知道方差, 并让网络只学习这个分布的均值，

所以回顾一下，我们有一个添加噪声的前向过程，我们对这个过程了如指掌，我们知道如何添加噪声，我们有一个反向过程, 我们不知道如何去噪，所以我们让网络学习如何去噪的参数，好了、现在我们已经定义了这两个过程，我们实际上如何训练模型来实现它呢? 正如你记得的，我们最初的目标实际上是学习我们数据集上的概率分布，所以这里的这个量，但与之前不同的是, 当我们能够边缘化时, 例如在罪犯想要生成身份的情况喜爱，我们可以对所有变量进行边缘化，这里我们不能边缘化，因为我们需要对 x1,x2,xt，xt，一直到 xt 进行边缘化，所以涉及很多变量，而要计算这个积分意味着要计算所有可能的x1和所有可能的x2等等，所以这是一个计算上非常复杂的任务，我们称之为计算上不可行，这些意味着它在理论上可能，但实际上需要永远的时间，所以我们不能使用这个方法，那么我们能做什么呢？

我们想要学习这里的这个量，所以我们想要学习这个参数 theta，以最大化我们在这里看到的似然，我们做的是，我们可以定义，我们找到了这个量的一个下界，所以这个量，即似然，而这个下界被称为 ELBO，如果我们最大化这个下界，它也会最大化似然，让我给你一个并行的例子，来说明最大化下界的含义，例如，想象你有一家公司，你的公司有一些收入，通常收入大于或等于你的公司销售额，所以你有来自销售的收入，也许你还有一些来自银行利息等的收入，但我们肯定可以说，公司的收入大于或等于公司的销售额，所以如果你想最大化你的收入，你可以最大化你的销售额，例如，这是你收入的一个下界，所以如果我们最大化销售额, 我们也会最大化收入, 这就是这里的想法:但在实际操作层面我们该怎么做呢?

这是根据 DDPM 论文定义的 DDPM 扩散模型的训练代码, 基本上思路是, 在我们得到 ELBO 之后, 我们可以将其参数化, 将损失函数定义为这样, 这意味着我们需要学习,我们需要训练一个称为epsilon theta 的网络, 它给定一个有噪声的图像, 所以这里的公式意味着在时间步t 的有噪声图像和添加噪声的时间步一一网络必须预测图像中有多少噪声，即噪声图像，如果我们对这个损失函数进行梯度下降, 这里我们将最大化 ELBO, 同时我们也将最大化我们数据的对数似然. 这就是我们如何训练这种网络的方法. 现在, 我知道这有很多概念需要掌握, 所以不用担心. 现在, 只需记住有一个前向过程和一个反向过程，为了训练这个网络进行反向过程, 我们需要训练一个网络来检测在时间步t 的有噪声图像版本中有多少噪声.

让我告诉你, 一旦我们有了这个已经训练好的网络, 我们实际上如何采样以生成新数据? 所以让我们来这里, 让我们来这里，那么我们如何生成新数据呢? 假设我们已经有了一

我们去除这个噪声, 然后再次询问网络那里有多少噪声, 并将其去除，然后我们问网络:那里有多少噪声一一好的, 去除它. 然后这里有多少噪声一一好的, 去除它. 等等等等, 直到我们到达这一步, 然后这里我们将得到一些新的东西. 所以如果我们从纯噪声开始, 并多次进行这个反向过程，我们将最终得到一些新的东西, 这就是这个生成模型背后的想法: 现在我们知道如何从纯噪声开始生成新数据, 我们也希望能够控制这个去噪过程, 以便我们可以生成我们想要的图像，我的意思是, 我们如何告诉模型从纯噪声开始生成一张猫、狗或房子的图片? 因为到目前为止, 从纯噪声开始并不断去噪, 我们当然会生成一张新图像. 但并不意味着我们可以控制生成哪张新图像. 所以我们需要找到一种方法告诉模型在这个生成过程中我们想要什么, 而想法是从纯噪声开始, 在这个去噪链中一一即噪声化过程中一一我们引入一个信号. 我们称之为提示，提示也可以称为条件信号, 或者也可以称为上下文，无论如何, 它们是同一个概念, 即我们影响模型如何去除噪声, 以便输出朝着我们想要的方向发展.

为了理解这是如何工作的, 让我们再次回顾一下这种网络的训练过程. 因为这对我们来说非常重要:学习这种网络的训练过程, 以便我们可以引入提示. 让我们回到之前.

好的, 正如我之前告诉你的, 我们的最终目标是建模一个分布一一θ, p(θ), 以便我们最大化我们数据的可能性, 并学习这个分布. 我们最大化 ELBO, 即下界.但是我们如何最大化 ELBO 呢? 我们最小化这里的损失，因此, 通过最小化这个损失, 我们最大化 ELBO, 从而学习这个分布. 因为这里的 ELBO 是我们数据分布可能性的下界.

那么这个损失函数是什么呢? 这里的损失函数表明我们需要创建一个模型 $\epsilon_\theta$, 这样如果我们给这个模型一个在特定噪声水平下的噪声化图像, 并且我们还告诉它我们在这个图像中加入了多少噪声, 网络就必须预测有多少噪声. 所以这个 $\epsilon$ 是我们添加的噪声量，这就是我们可以在训练循环上进行梯度下降的地方. 这样我们将学习我们数据上的分布，但是, 正如你所见, 这个分布不包含任何告诉模型什么是猫、什么是狗或什么是房子的信息，模型只是学习如何生成有意义的、类似于我们初始训练数据的图片, 但它不知道图片与提示之间的关系，所以一个想法是:我们能否学习一个关于我们初始数据的联合分布, 包括所有图像和条件信号, 即提示?

嗯, 这也是我们不想要的, 因为我们实际上想学习这个分布, 以便我们可以采样并生成新数据. 我们不想学习联合分布，那会受到上下文太大影响, 模型可能无法学习数据的生成过程，所以我们的最终目标始终是这个，但我们还想找到某种方法, 让模型构建我们想要的东西，这个想法是我们修改这个单元, 所以这里的模型 $\epsilon_\theta$ 将使用, 让我给你展示, 这个单元模型，这个单元将接收一个噪声化的图像作为输入, 比如一只猫，在特定噪声水平下:我们还告诉它我们添加到这只猫的噪声水平，并将两者都输入到单元中，单元必须预测有多少噪声, 这是单元的任务. 如果我们在这里也引入提示信号，即条件信号:会怎么样呢? 这样, 如果我们告诉模型, 你能从这个有这么多噪声的图像中去除噪声吗? 我还告诉你这是一只猫, 所以模型有更多信息来去除噪声，是的, 模型可以这样学习如何去除噪声, 并构建更接近提示的东西，这将使模型具有条件性，这意味着它将表现的像一个条件模型，所以我们需要告诉模型我们想要的条件是什么，以便模型可以以那种特定的方式去除噪声, 使输出朝向那个特定的提示，但同时, 当我们训练模型时、除了给带有提示的图像外，我们还可以有时一一比如说50

所以我们知识在他输入是给他一堆零，这样, 模型将学会既作为条件模型又作为非条件模型来行动，所以模型将学会关注提示, 也学会不关注提示，这样做有什么好处呢?这样我们可以在想要生成新图片时, 进行两个步骤，在第一步, 假设你想生成一张猫的图片, 我们可以这样做.首先让我删除，我们可以进行第一步我们称之为步骤一，我们可以从纯噪声开始，因为正如我之前所说, 要生成新图像，我们从纯噪声开始，我们指示模型，噪声水平是多少，所以一开始是 t 等于1000、即最大噪声水平，我们告诉模型我们想要一只猫，并将其作为输入给单元，单元将预测一些我们需要去除的噪声, 以便将图像移向我们想要的输出，即一只猫, 这就是我们的输出一，我们称之为输出一，

然后我们再进行另一个步骤，所以让我删除这个，然后我们进行男一个步骤、我们称之为步骤二，再次, 我们给出与之前相同的输入噪声, 相同的时间步长作为噪声水平，所以它是具有相同噪声水平的相同噪声, 但我们不提供任何提示，这样，模型将生成一些输出，我们称之为输出二，这是关于如何去除噪声, 以生成我们不知道是什么的东西，但生成的东西属于我们的数据分布, 然后我们以某种方式组合这两个输出，以便我们可以决定我们希望输出更接近提示的程度，这被称为无分类器引导, 所以这种方法被称为无分类器引导，我不会告诉你为什么它被称为无分类器引导, 否则我需要介绍分类器引导, 并且要谈论分类器引导, 我需要介绍基于分数的模型、以理解为什么它被称为这样.

但这个想法是这样的，我们训练一个模型, 在训练时, 有时我们给它提示, 有时我们不给出提示, 这样模型学会忽略提示, 但也学会关注提示，当我们从这个模型中采样时, 我们进行两个步骤，第一次我们给它我们想要的提示, 第二次我们给出相同的噪声但没有我们想要的提示，然后我们线性组合这两个输出一一有条件的和无条件的，通过一个权重, 该权重指示我们希望输出接近我们条件的程度，接近我们的提示，这个值越高, 输出就越接近我们的提示，这个值越低，输出就越不接近我们的提示，这就是无分类器引导背后的思想.

实际上, 我们需要给出某种嵌入. 因此, 模型需要理解这个提示，为了理解提示, 模型需要某种嵌入. 嵌入意味着我们需要一些向量来表示提示的含义, 这些嵌入是通过使用 CLIP 文本编码器提取的，所以在谈论文本编码器之前, 我们先来谈谈 CLIP，所以 CLIP 是由 Open A I构建的模型, 它能够将文本与图像连接起来，基本上，他们拿了一大堆图像，例如, 这张图片及其描述，然后他们拿了另一张图片及其描述，所以图像一与文本一相关联, 文本一是图像一的描述，然后图像二有描述二，图像三有文本三, 这是图像三的描述, 等等，他们构建了这个矩阵, 你可以在这里看到, 它是由第一个图像的嵌入与所有可能的描述的点积组成的，所以图像一与文本一, 图像一与文本二, 图像一与文本三，等等，然后图像二与文本一, 图像二与文本二, 等等.

他们是如何训练的呢? 基本上我们知道图像与文本的对应关系在矩阵的对角线上, 因为图像一与文本一相关联、图像二与文本二相关联, 图像三与文本三相关联，那么他们是如何训练的呢? 基本上他们构建了一个损失函数, 他们希望对角线上的值最大, 而其他所有值都为零, 因为这些值不匹配, 它们不是这些图像的相应描述.

通过这种方式, 模型学会了如何将图像的描述与图像本身结合起来，在稳定扩散中, 我们使用这里的文本编码器, 即 CLIP 的一部分来编码我们的提示, 以获取一些嵌入. 这些嵌入随后被用作我们单元的条件信号, 以去噪图像为我们想要的样子.

嗯，好吧，还有一件事我们需要理解，如我之前所说我们有个正向过程，它向图像添加噪声，然后我们有一个反向过程，它从图像中去除噪声，这个反响过程可以通过使用无分类器引导来条件化，这个反向过程意味着我们需要进行多次噪声化步骤才能到达图像，到达新图像，这也意味着每个步骤都涉及通过带有噪声的图像的单元, 并作为输出获取该图像中存在的噪声量，但如果图像非常da, 比如这里的图像是 512 乘以 512, 这意味着每次在单元中我们都会有一个非常大的矩阵需要通过这个单元，这可能会非常慢，因为它是一个非常大的数据矩阵，单元需要处理，如果我们能以某种方式将图像压缩成更小的东西，使得每次通过单元的时间更少呢? 嗯，想法是，是的, 我们可以用一种叫做变分自编码器的东西来压缩图像，让我们看看变分自编码器是如何工作的。

好的, 稳定扩散实际上被称为潜在扩散模型, 国为我们学习的不是数据的概率分布 Px, 而是使用变分自编码器学习数据的潜在表示. 所以, 基本上, 我们压缩了我们的数据，所以让我们回到正题, 我们将数据压缩成更小的东西, 然后我们学习噪声化过程，使用数据的压缩版本，而不是原始数据，然后我们可以解压缩它来构建原始数据，让我实际展示一下，它在实践层面是如何工作的，

所以想象一下, 你有一些数据, 你想通过互联网发送给你的朋友. 你会怎么做? 你可以发送原始文件, 或者你可以发送压缩文件, 比如你可以用 Win Zip 压缩文件, 然后发送给你的朋友, 朋友收到后可以解压缩并重建原始数据，这正是自编码器的工作. 自编码器是一个网络、给定一个图像, 例如, 在通过编码器后, 会转换成一个维度远小于原始图像的向量. 如果我们使用这个向量并通过解码器运行它, 它将重建回原始图像. 我们可以对许多图像进行这样的操作, 每个图像在这个过程中都会有一个表示，这被称为对应于每个图像的编码. 现在, 自编码器的问题在于, 该模型学习的编码在语义上没有任何意义，因此, 例如, 与猫相关的编码可能与与披萨或建筑相关的编码非常相似，所以, 这些编码之间没有语义关系，为了克服自编码器的这一局限性, 我们引入了变分自编码器, 在这种编码器中, 我们学会了压缩数据, 但同时这些数据是根据多元分布分布的，大多数时候是高斯分布. 我们学习这个分布的均值和标准差, 这个非常复杂的分布. 并且给定潜在表示, 我们总是可以通过解码器来重建原始数据，这就是我们在稳定扩散中也使用的想法.

现在我们可以将所有这些我们见过的东西结合起来, 看看稳定扩散的架构是什么样的. 那么, 让我们从文本到图像的工作原理开始. 现在想象一下, 文本到图像基本上是这样工作的. 想象一下, 你想生成一张戴着眼镜的狗的图片. 所以你从提示开始, 比如"一只戴着眼镜的狗". 然后你会怎么做呢? 我们采样一些噪声 从 $N(0,1)$ 中采样一些噪声. 我们用变分自编码器对其进行编码. 这将给我们这个噪声的潜在表示. 我们称之为 Z. 这当然是一团纯粹的噪声, 但已经被编码器压缩, 然后我们将其发送到单元. 单元的目标是检测有多少噪声, 并且由于我们还将条件信号提供给单元, 单元必须检测噪声. 我们需要去除多少噪声才能使其变成一张符合提示的图片, 也就是一张狗的图片.

所以, 我们通过单元传递它, 连同时间步, 初始时间步, 比如1000, 单元将在这里的输出中检测有多少噪声, 我们的调度器, 稍后我们会看到调度器是什么, 将去除这个噪声, 然后再次将其发送给单元进行第二步的去噪, 再次发送时间步, 这次不是1000, 而是例如980, 因为我们跳过了一些步骤. 然后我们再次处理噪声. 我们检测有多少噪声, 调度器将去除这个噪声, 再次发送回去, 我们多次重复这个过程, 我们不断这样做, 去噪一一经过许多步骤, 直到图像中不再有噪声，在我们完成这些步骤的循环后, 我们得到输出 Z', 这仍然是一个潜在表示, 因为该单元只处理数据的潜在表示, 而不是原始数据.我们通过解码器传递它以获得输出图像. 这就是为什么它被称为潜在扩散模型, 因为单元, 即去噪过程, 总是处理数据的潜在表示. 这就是我们如何生成文本到图像的过程.

我们可以对图像到图像做同样的事情，图像到图像意味着, 例如, 我有一张狗的图片, 我想通过使用提示将这张图片修改成其他东西，例如, 我想让模型给这只狗加上眼镜, 所以我在这里输入图像, 然后说"一只戴着眼镜的狗", 希望模型会给这只狗加上眼镜. 它是如何工作的呢? 我们使用变分自编码器的编码器对图像进行编码, 得到图像的潜在表示. 然后我们在这个潜在表示上添加噪声, 因为正如我们之前看到的, 该单元的任务是对图像进行去噪. 但当然, 我们需要有一些噪声来进行去噪. 所以我们给这张图像添加噪声, 以及我们添加到这张图像的噪声量，所以这里的起始图像指示了单元在构建输出图像时有多少自由度, 因为我们添加的噪声越多, 单元就越有自由度去改变图像. 但我们添加的噪声越少, 模型改变图像的自由度就越小，因为它不能进行剧烈的改变. 如果我们从纯噪声开始, 单元可以做任何它想做的事情. 但如果我们从较少的噪声开始, 单元就只能稍微修改输出图像. 所以我们从多少噪声开始, 就表明我们希望模型在这里对初始图像给予多少关注. 然后我们给出提示. 经过许多步骤, 我们不断去噪、去噪、去噪, 当不再有噪声时, 我们取这个潜在表示, 通过解码器传递它, 然后在这里得到输出图像，这就是图像到图像的工作原理.

现在让我们进入最后一部分, 即如何进行图像修复，图像修复的工作方式与图像到图像类似，但使用了一个遮罩，所以图像修复首先意味着我们有一张图像，我们想要裁剪掉图像的某一部分, 比如这只狗的腿;然后我们希望模型为这只狗生成新的腿, 可能会有一些不同，所以, 正如你所见， 这里的脚与这只狗的腿有些不同，所以我们从这只狗的初始图像开始, 通过编码器传递它，它变成了一个潜在表示，我们在这个潜在表示上添加一些噪声，我们给出一些提示, 告诉模型我们希望它生成什么，所以我只是说"一只运行的狗，因为我希望为这只狗生成新的腿。然后我们将加噪的输入传递给单元，单元将在第一个时间步生成一个输出. 但当然, 没有人告诉模型只预测这个区域，当然, 模型在这里的输出中预测并修改了整个去噪后的图像，但我们在这里的输出中移除了一一我们不在乎它预测了什么，图像中我们已经知道的那个区域的噪声，我们用我们已经知道的图像替换它，并再次通过单元传递它.

基本上, 我们在每个步骤、每个单元的输出中, 用原始图像的已知区域替换已知区域, 以便基本上欺骗模型, 让它以为这些图像细节是模型自己生成的而不是我们，所以每次在这里的这个区域, 在我们将其发送回单元之前，我们通过用原始图像替换单元为我们生成的这个区域的任何输出，来组合单元的输出和现有图像，然后我们将其返回给单元，让我们继续这样做，这样, 模型将只能在这个区域工作、因为这是我们从未在单元的输出中替换的区域，然后在没有更多噪声后, 我们取输出, 将其发送到解码器, 然后它会构建我们在这里看到的图像. 好的, 这就是从架构角度来看稳定扩散的工作原理.

我知道这是一段漫长的旅程, 我不得不介绍许多概念, 但在我们开始构建单元、模型之前, 了解这些概念非常重要, 否则我们甚至不知道如何开始构建稳定扩散.

--------------

现在我们终于开始编写我们的稳定扩散代码了，我们首先要编写的是变分自编码器，因为它在 Unet 之外，也就是扩散模型之外，负责检测和预测图像中存在的噪声量。

变分自编码器的编码器和解码器的工作是将图像或噪声编码为图像或噪声本身的压缩版本，然后可以将这个潜在变量通过 Unet 运行，然后在噪声化的最后一步，我们将这个潜在变量通过解码器，以获得原始的输出图像。

所以编码器的工作实际上是将数据的维度降低为更小的数据，这个想法与 Unet 非常相似，我们从一个非常大的图片开始，在每一步我们不断缩小图像的尺寸，但同时不断增加图像的特征通道。



所以变分自编码器, 让我再给你展示一下这张图.
So. the variational auto encoder, let me show you again. the picture.
在这里我们不是学习如何压缩数据, 我们是在学习一个潜在
Input Recon i
ructed 空间, 而这个潜在空间是多元高斯分布的参数.
Input 所以实际上, 变分自编码器被训练来学习这个分布的均值和
In put 方差, 即和, 这实际上是我们从这个变分自
ariational auto encoder is trained to learn the mu and fhe sigma so themes
In put 编码器的输出中得到的东西, 而不是直接的压缩图像.
ariamce of this distribution and this is actually what we wilgetfnomtheoutpu
Input 如果这还不清楚, 伙计们, 我之前做过一个关于变分自
Input 编码器的视频, 在其中我也展示了为什么, 为什么我们要
Input Recons Input 这样做, 所有的参数重参数化技巧等等. 但现在, 只需
In pu 记住这不仅仅是一个压缩版本的图像, 它实际上是一个分布
Input 然后我们可以从这个分布中采样, 我会展示给你看,
然后我们可以从这个分布中采样, 我会展示给你看.
can sample from this distribution and i will show you how
所以变分自编码器的输出实际上是均值和方差.
So. the :output of the Variational Log Encoder is actually the mean. and. the-variance.
实际上, 它不是方差, 而是对数方差.
And actually it's not the variance but the log variance.
所以均值和对数方差等于to rch. chunkx2维度等于1.
So. the. mean and the log variance is equal to torch. chunk x2dimension. equal. 1
我们也会看到什么是chunk 函数.
We will see also what is the chunk function.
所以我会展示给你看
So I will show you.
所以这基本上将批量大小8-个通道、高度、高度除以
So this basically converts batch size, 8 channels, height y
所以这基本上将批量大小、8 -个通道、高度、高度除以
8、宽度除以8, 这是该编码器最后一层的输出, 也
height. divided by 8width dividedby8which is the output of the last layer. of. this.
所以这个chunk 基本上意味着沿着这个维度将其分成两个张量.
encoder. so this one and we divide it into two tensors so this. chunk-basically means
所以这个chunk 基本上意味着沿着这个维度将其分成两个张量
divide it
所以这个chunk 基本上意味着沿着这个维度将其分成两个张量.
into. two. tensors along this dimension so along this dimension. it will. become. two.
所以两个张量的形状是, 批量大小, 然后高度除以8, 和
tensors. of size along this dimension of size4so two tensors of. shape batch size.
所以两个张量的形状是, 批量大小, 然后高度除以8, 和
所以两个张量的形状是, 批量大小, 然后高度除以8, 和
henheightdividedby8, and width, oops, width. divided by 8
而实际上, 这个的输出基本上代表了均值和方差
And. this, basically, the output of this actually represents the. mean and the
而实际上 这个的输出基本上代表了均值和方差
variance.
而我们想要的是方差, 实际上我们不想要对数方差.
And. what. we do, we don't want the log variance, we want. the-variance, actually.
所以, 为了将对数方差转换成方差, 我们进行指数运算.
So, to. transform the log variance into variance, we do. the. exponentiation..
所以首先我们还需要做的是钳制这个方差, 否则它会变得非常小.
So. the first thing actually we also need to do is to clamp-this. variance because
所以钳制意味着如果方差太小或太大, 我们希望它在我们可
v otherwise it will become very small so clamping means that if. the. variance is. too.
所以钳制意味着如果方差太小或太大, 我们希望它在我们可
small or too big
所以钳制意味着如果方差太小或太大, 我们希望它在我们可
we want it to become within some ranges that we are acceptable for us. so. this
接受的范围内. 这个钳制函数一一对数方差一一告诉
we want it to become within some ranges that we are acceptable for us. so. this.
Py Torch, 如果值太小或太大, 就让它在这个范围内.
clamping function log variance tells the py torch that if. the value is too. small. or.
Py Torch, 如果值太小或太大, 就让它在这个范围内
too big make it within this range nas
这不会改变张量的形状, 所以这里仍然是这个张量. 然后
:and this. doesn't change the shape of the tensors so this still remains this tensor.
所以方差等于对数方差, 点exp, 这意味着取这个的指数
here. and. then we transform the log variance into variance so. the. variance. is. equal. to
所以方差等于对数方差, 点exp, 这意味着取这个的指数
the log
所以方差等于对数方差, 点exp, 这意味着取这个的指数.
variance dot exp which means make the exponential of this so you delete. the. log and
所以你去掉对数, 它就变成了方差, 这也不会改变张量的
it becomes the variance
所以你去掉对数, 它就变成了方差, 这也不会改变张量的
and this. also doesn't change the size of the the shape of. the. tensor and. then. to
然后要从方差计算标准差, 正如你所知, 标准差是方差的
and this. also doesn't change the size of the the shape of. the. tensor and. then. to
平方根. 所以标准差是方差, 点, sqrt, 这也不会改变张量
square root of the variance so standard deviation is the variance dot sqrt-and also.
平方根. 所以标准差是方差, 点, sqrt, 这也不会改变张量
this doesn't change the size of the tensor. okay. now.
的大小.
this doesn't change the size of the tensor. okay. now
好的, 现在正如我之前告诉你的, 这是一个潜在空间.
this doesn't change the size of the tensor. okay. now
好的, 现在正如我之前告诉你的, 这是一个潜在空间.
Input Reconstructed 好的, 现在正如我之前告诉你的, 这是一个潜在空间.
What we want, as told you be fone, this is a latent spaces
Recor Input uc ted
Input truc ted 它是一个多元高斯分布, 有自己的均值和方差.
它是一个多元高斯分布, 有自己的均值和方差.
It's. a multivariate Gaussian, which has its own mean and its own variance.
我们知道均值和方差.
And we know the mean and the variance.
这个均值和这个方差.
This mean and this variance.
我们如何转换?
How do we convert?
我们如何从中采样?
How do we sample from it?
好吧, 我们可以从中采样的是基本上:我们可以从n01 Well,
好吧, 我们可以从中采样的是基本上:我们可以从n01 what. we. can sample from is basically we can sample from. no1 this is. we. if. we have. a
如果我们有一个来自n01 的样本, 我们如何将其转换为给定
what. we. can sample from is basically we can sample from. no1 this is. we. if. we have. a.
均值和方差的样本. 正如你在概率和统计中记得的那样
sample. from. no1 howdoweconvert it into a sample of a given. mu soa given mean-and
均值和方差的样本. 正如你在概率和统计中记得的那样
the given
均值和方差的样本. 正如你在概率和统计中记得的那样
variance. this as if you remember from probability and statistics. if. you have a sample
如果你有一个来自n01 的样本, 你可以通过这种变换将其
variance. this as if you remember from probability and statistics. if. you have a sample
转换为具有给定均值和方差的任何其他高斯样本.
from. n01 you can convert it into any other sample of a gaussian with a given mean and
转换为具有给定均值和方差的任何其他高斯样本.
the variance through this transformation
所以, 如果 Z一一 我们称之为 Z一一等于 N01, 我们
So. if. Z, let'scall it this one, Zis equal to No1, we can transform into another. N
可以通过这种变换将其转换为另一个 N 一 我们称之为×.
let'scall it X, through this transformation.
X等于乙, 嗯, 新分布的均值加上新分布的标准差乘以乙.
Xis. equal. to Z, well, the mean of the new distribution plus. the. standard. deviation
X 等于乙, 嗯, 新分布的均值加上新分布的标准差乘以乙.
of the new distribution multiplied by. Z.
这是变换, 这是概率统计中的公式一一基本上意味着将
this. is the transformation this is the formula from probability-statistics. basically
这个分布转换为具有这个均值和这个方差的分布, 这基本上
means transform this distribution into this one that has this. mean and this variance.
这个分布转换为具有这个均值和这个方差的分布, 这基本上
which
这个分布转换为具有这个均值和这个方差的分布, 这基本上
basically. means sample from this distribution this is why we. are given also. the. noise.
这就是为什么我们也将噪声作为输入, 因为我们需要噪声
basically. means sample from this distribution this is why we are given also. the noise.
生成器使用特定的种子生成噪声.
:as input. because the noise we wanted to come from with. the particular. seed of. the.
生成器使用特定的种子生成噪声
noise
生成器使用特定的种子生成噪声.
generator. so we ask is as an input and we sample from. this. distribution. like. this. x.
所以我们将其作为输入, 并像这样从这个分布中采样:x generator. so we ask is as an input and we sample from. this. distribution. like. this. x.
所以我们将其作为输入, 并像这样从这个分布中采样:x is equal to mean plus standard deviation
等于均值加上标准差乘以噪声.
is equal to mean plus standard deviation
等于均值加上标准差乘以噪声
最后, 还有另一个步骤, 我们需要用一个常数来缩放输出.
Finally, there is also another step that we need to. scale
最后, 还有另一个步骤, 我们需要用一个常数来缩放输出.
最后, 还有另一个步骤, 我们需要用一个常数来缩放输出.
the. output by a constant this constant i found it in the original. repository so. im
这个常数我在原始仓库中找到了, 所以我只是在这里写
just. writing it here without any explanation on why because i actually i also dont.
下来, 没有解释为什么, 因为我实际上也不知道.
know it's
它只是 个他们在最后使用的缩放常数.
know it's
我不知道这是否出于历史原因, 因为他们使用了一些之前有
just. a. scaling constant that they use at the end i don't know. if it's there for
这个常数的模型 或者他们出于某种特定原因引入了它, 但
historical. reason because they use some previous model that. had. this. constant or. they
这个常数的模型, 或者他们出于某种特定原因引入了它, 但
introduced it
这个常数的模型, 或者他们出于某种特定原因引入了它, 但
for. some particular reason but it's a constant that i saw it. in the. original
我在原始仓库中看到了这个常数, 实际上, 如果你检查稳定
for. some particular reason but it's a constant that i saw it. in the. original
扩散模型的原始参数, 也有这个常数. 所以我也在输出上用
扩散模型的原始参数, 也有这个常数. 所以我也在输出上用
stable diffusion model there is also this constant so I'm also. scaling the. output. by
这个常数进行缩放, 然后我们返回x.
this constant and then we return. x. s
现在我们到目前为止构建的内容, 除了我们还没有构建残差块
now what we built so far except that we didn't build the. residual. block and the
和注意力块
:now what we built so far except we didn't build. the. residual. block and the
Recon 和注意力块.
In pu 这里我们构建了变分自编码器的编码器部分和采样部分. 所以
Input 我们取图像, 通过编码器运行它、它变得非常小. 它会
Input 告诉我们均值和方差, 然后我们从那个分布中菜样. 给定
Input Recon ruc ted 均值和方差.
In pu 现在我们需要构建解码器以及残差块和注意力块, 我们会看到
现在我们需要构建解码器以及残差块和注意力块, 我们会看到
mean and the variance now we need to build the decoder. along with. the
现在我们需要构建解码器以及残差块和注意力块, 我们会看到
Input 现在我们需要构建解码器以及残差块和注意力块, 我们会看到
esi dual block and the attention block and what we will see is f hatin then decoder w
Input Recon Is put truc te d 在解码器中我们做的是编码器的反向操作.
esidual block and the attention block and what we will see is that in then decoder
Input 所以我们会减少通道数, 同时增加图像的大小. 让我们进入
所以我们会减少通道数, 同时增加图像的大小. 让我们进入
the. same. time we will increase the size of the image so. let'sgo. to. the decoder. let.
解码器. 让我检查一下是否一切正常, 看起来没问题
v. the. same. time we will increase the size of the image so. let'sgo. to. the decoder. let
解码器. 让我检查一下是否一切正常, 看起来没问题
me. review if everything is fine looks like it is so let'sgo to. the. decoder.
我们还需要定义注意力. 我们需要定义自注意力. 稍后我们
We. also. need to define the attention We need to define the. self. attention. later we.
让我们先定义残差块, 就是我们之前定义的那个, 这样你就
define it Let's define first the residual block the one we defined. before so. that. you.
让我们先定义残差块, 就是我们之前定义的那个, 这样你就
understand
明白什么是残差块了, 然后我们定义之前定义的注意力块
what is. this residual block and Then we define the attention-block that. we defined..
最后我们构建注意力.
before and finally we build the attention so
所以, 好的, 这是由归一化和卷积组成的.
before and finally we build the attention so
所以, 好的, 这是由归一化和卷积组成的.
okay. this is made up of normalization and convolutions like. i said before. there. is. a.
就像我之前说的, 有两种归一化, 一种是组归一化, 然后
two normalization which is the group norm one
就像我之前说的, 有两种归一化, 一种是组归一化, 然后
是另一种组归一化.
是另一种组归一化.
And then there is another group normalization.
使用远程通道到输出通道.
Use remote channels to out channels.
然后我们有一个跳跃连接.
And then we have a skip connection.
跳跃连接基本上意味着你取输入, 跳过一些层, 然后将其与
Skip. connection basically means that you take the input, you skip some layers. and.
跳跃连接基本上意味着你取输入, 跳过一些层, 然后将其与
then you connect it with the output of the. last layer.
最后一层的输出连接起来.
then you connect it with the output of the. last layer.
我们还需要这个残差连接.
And we also need this residual connection.
如果两个通道不同, 我们需要创建另一个中间层.
: If. the two channels are different, we need to create another intermediate layer.
现在我创建它, 稍后我会解释它.
Now l create it, later l explain it.
好的, 让我们创建前向方法, 它接受一个 Torch Tensor 并返回一个
. Ok, let's create the forward method which is a. Torch. Tensor.
好的, 让我们创建前向方法, 它接受一个 Torch Tensor 并返回一个
好的, 让我们创建前向方法, 它接受一个 Torch Tensor 并返回一个
:and returns the torch dot tensor okay the input of this residual layer as you saw
好的, 这个残差层的输入, 就像你之前看到的, 是一个
:and returns the torch dot tensor okay the input of this residual layer as you saw
包含一些通道、 高度和宽度的批次, 这些尺寸可能不同
before is. something that has a batch with some channels and. then height and width
ox(512, 152), 包含一些通道 高度和宽度的批次, 这些尺寸可能不同.
which can be
包含一些通道、 高度和宽度的批次, 这些尺寸可能不同
different it's not always the same sometimes it's512by. 512sometimes it's. half. of.
它并不总是相同的. 有时是512x512, 有时是它的一半, 有时是
different it's not always the same sometimes it's512by. 512sometimes it's. half. of.
所以假设×是批次大小、通道数、高度、宽度.
that. sometimes it's one fourth of that etc so suppose it's. x is batch size.
所以假设×是批次大小、通道数、高度、宽度.
所以假设是批次大小、通道数、高度、宽度.
in channels. height width what we do is we create the skip. connection so. we save. the
我们要做的是创建跳跃连接, 所以我们保存初始输入. 我们
in channels. height width what we do is we create the skip. connection so. we save. the.
称之为残差, 或者residue 等于x.
initial input we call it the residual or residue is equal. to x. we apply-the
我们应用归一化, 第一个, 这不会改变张量的形状.
initial input we call it the residual or residue is equal to x we apply-the.
我们应用归一化, 第一个, 这不会改变张量的形状.
normalization the
我们应用归一化, 第一个, 这不会改变张量的形状.
first one. and this doesn't change the shape of the the tensor. the. normalization
归一化不会改变. 然后我们应用激活函数.
doesn't change then we apply the silo function
这也不会改变张量的大小. 然后我们应用第一个卷积.
this. also doesn't change the size of the tensor then we apply the first convolution
这也不会改变张量的大小, 因为如你所见, 这里我们有核
this. also. doesn't change the size of the tensor because as you can see. here we. have
大小3, 是的, 但有1的填充. 有1的填充
kernel size
大小3, 是的, 但有1的填充. 有1的填充
3yes. but with the padding of 1with thepaddingof1. actually it will not change. the
实际上它不会改变张量的大小, 所以它仍然保持不变. 然后
3yes. but with the padding of 1with thepaddingof1actually it will not change. the
我们再次应用组归一化2.
:size of. the tensor so it will still remain this one then we apply-again the-group
我们再次应用组归一化2.
normalization2
这再次不会改变张量的大小. 然后我们再次应用激活函数.
this. again. doesn't change the size of the tensor then we apply the silo. again. then we
然后我们应用第二个卷积, 最后, 我们应用残差连接, 这
this. again. doesn't change the size of the tensor then we apply the silo again. then we.
然后我们应用第二个卷积, 最后, 我们应用残差连接, 这
apply the convolution number two.
然后我们应用第二个卷积, 最后, 我们应用残差连接, 这
and. finally we apply the residual connection which basically. means. that we. take. x.
但如果输出通道的数量不等于输入通道的数量, 你就不能将
plus. the. residual but if the number of output channels is not. equal to. the input
但如果输出通道的数量不等于输入通道的数量, 你就不能将
channels you
它们相加, 因为这两个维度将不匹配.
channels you
它们相加, 因为这两个维度将不匹配.
can not add. this one with this one because this dimension. will not match. between. the.
它们相加, 因为这两个维度将不匹配.
two
那么我们该怎么做呢? 我们在这里创建这个层, 将输入通道
So what we do, we create this layer here to convert the input. channels to. the. output
转换为×的输出通道, 以便可以进行这个求和
So what we do, we create this layer here to convert the input. channels to. the. output.
转换为×的输出通道, 以便可以进行这个求和
channels of X such that this sum can be done.
所以我们做的是应用这个残差层.
So what we do is we apply this residual layer.
残差层的残差, 像这样.
Residual layer of residual, like this.
这就是一个残差块.
And this is a residual block.
正如我告诉你的, 它只是一系列卷积和组归一化.
As. ltold you, it's just a bunch of convolutions and group normalization.
对于那些熟悉计算机视觉模型, 尤其是 Res Net 的人来说
And for those who are familiar with computer vision models, especially in. Res Net, we
我们会大量使用它.
And for those who are familiar with computer vision models, especially in. Res Net, we.
我们会大量使用它.
use a lot of it.
这是一个非常常见的模块.
It's a very common block.
让我们来构建我们在编码器中也使用过的适应块.
:o Let'sgo build the adaptation block that we used also before. in. the encoder.
这个在这里
This one here.
为了定义注意力, 我们还需要定义自注意力. 所以让我们
and. to. define the attention we also need to define the self-attention so. let'sfirst..
首先构建使用变分自编码器的注意力块, 然后我们定义什么是
build the. attention block which is using the variational auto encoder and. then we.
首先构建使用变分自编码器的注意力块, 然后我们定义什么是
define what is this self-attention. a
自注意力?
define what is this self-attention..
它有一个组归一化. 同样, 在稳定扩散中, 通道数总是
it has a group normalization again the channel is always 32. here in stable. diffusion.
10. 但你也可能想知道什么是组归一化, 对吧? 所以让我们
it has a group normalization again the channel is always 32. here in stable. diffusion.
来回顾 下
but you also. may be wondering what is group normalization right so. let'sgo. to review it.
实际上, 既然我们在这里, 如果你记得我在之前的关于 LLa MA actually since we are here and okay if you remember from my previous slide son. llama
Input 实际上, 既然我们在这里, 如果你记得我在之前的关于 LLa MA
Input ruc ted 的幻灯片中, 让我们来看看我们使用层归一化的地方
With layer normalization we
所以, 首先:什么是归一化? n alize by rows (data items )
With layer normalization we So, first c f all, what is mor malization?
With layer normalization we normalize by rows (data items )
归一化基本上是当我们有一个深度神经网络时网络的每层m2s ))
With layer normalization we
Monmali z atio each layer of the
With layer normalization we al network, each layer r of the
With layer normalization we normalize by rows (data items )
With layer normalization we Now. wha kh
gin distribution so
比如有时一层的输出在0到1之间, 但下步可能data items )
With layer normalization we
在3到5之间, 再下一步可能在·10到5ws(data items )
With layer normalization we
之间, 等等
With layer normalization we normalize by rows (data items )
因此, 如果一层的输出分布发生变化, 那么下一层也会看到tems )
With layer normalization we
与其习惯看到的输入非常不同的输灭g by rows (data items ))
With layer normalization we
With layer normalization we normalize by rows (data items )
这基本上会将下一层的输出推向一个新的分布*进而使损失toms )
With layer normalization we
这基本上使得损失函数振荡过大, 从而使训练变慢:(data items )
With layer normalization we
在批量归一化中, 我们按列进行归一化, 因此统计量均值和tems With layer normalization we
标准差是按列计算的. normalize by rows (data items )
With layer normalization we with batch normalization. we mc
With layer normalization we normalize by rows (data items )
在层归一化中, 它是按行计算的, 因此每个项图独立于其他tems )
With layer normalization we
With layer normalization we normalize by rows (data items )
the others
而组归一化则类似于层归一化, 但不是所有项目的特征
而是分组进行.
With layer normalization we normalize by rows (data items )
所以, 例如, 想象你这里有因个特征g by rows (data items ))
With layer normalization we
With layer normalization we
With layer normalization we normalize by rows (data items )
那么第一组将是 F1和 F2、第二组将是 F3和 P4s(data items )
With layer normalization we will be f3and F4
With layer normalization we normalize by rows (data items )
所以你将有两个均值和两个方差
With layer normalization we
用于第二组
With layer normalization we normalize by rows (data items )
So you will h
但为什么我们要这样使用它? 为什么我们要对这种特征进行toms ))
With layer normalization we
分组? 因为这些特征实际上来自卷积, 正如我们之前看到
的, 让我们回到网站. 想象你这里有一个5x5 的卷积核.
的, 让我们回到网站. 想象你这里有一个5x5 的卷积核.
back to the website imagine you have a kernel of five be re
这里的每个输出实际上来自图像的局部区域.
Each output here actually comes from local area of the image.
因此, 两个接近的特征, 例如彼此靠近的两个事物, 可能
So the two close features, for cx ample, two things that are close to each other may
彼此相关.
be related to each other.
因此, 两个相距较远的事物彼此不相关.
things that are far from each other are not related to each other.
这就是为什么在这种情况下我们可以使用组归一化.
因为彼此接近的特征将具有相似的分布, 或者我们使它们具有
相同的分布, 而相距较远的事物可能不会.
这就是组归一化的基本思想.
但归一化的整体思想是, 我们不希望这些事物波动大大
否则, 损失函数将会波动, 从而使训练变慢.
通过归一化, 我们使训练更快.
with no z malization we make the training faster
所以让我们回到编码.
So-let's go back to coding.
所以我们正在编写注意力模块的代码.
So we were coding the attention block.
所以现在注意力模块有这个组归一化, 还有一个注意力机制
So now the. attention block has this group normalization and also. an attention, which
稍后我们会定义它.
And later we define it.
和通道 好的.
And channels, okay.
这个有一个前向方法.
This one have a forward method.
Torch 点张量.
Torch dot tensor.
x * alf. cm, 2(x)
当然返回一个 I orch 点张量.
Returns, of course, a torch dot tensor
好的, 这个模块的输入是什么?
Okay, what is the input of this block?
这个模块的输入是某种形式的批量大小、 通道数、 高度和
The. input. of this block is something in the form of batch size, number of. channels
这个模块的输入是某种形式的批量大小、通道数、高度和
height and width.
宽度
height and width.
但由于这个注意力模块会在多个位置使用, 我们不定义具体的
But. because it will be used in many positions, this attention. block, we don't define
但由于这个注意力模块会在多个位置使用, 我们不定义具体的
specific size.
所以我们只是说文是某种形式的批量大小、特征或通道
So. we just say that x is something that is batch size, features. or channels, if you
(如果你愿意)、高度和宽度.
So. we just say that x is something that is batch size, features. or channels, if you
(如果你愿意)、高度和宽度
want, height and width.
再次, 我们创建一个残差连接.
Again, we create a residual connection.
我们做的第一件事是提取形状.
And the first thing we do is we extract the. shape.
所以n是批量大小, 通道数, 高度和宽度等于x. shape.
Son. is the. batch'size, the number of channels, the height and the width is. equal. to.
所以n是批量大小, 通道数, 高度和宽度等于x. shape.
x. shape.
然后, 正如我之前告诉你的, 我们对这张图像的所有像素
The n, as. l told you before, we do the self-attention between all. the pixels of this
进行自注意力计算.
The n, as. ltold you before, we do the self-attention between all the pixels of. this
进行自注意力计算.
image.
我会展示给你看.
And I will show you how.
这将转换这里的张量.
This will transform this tensor here.
转换成这里的张量:高度乘以宽度. 所以现在我们有一个
into. this tensor here height mutti plied by width so now we have a sequence. where. each.
序列, 其中每个项目代表一个像素, 因为我们乘以高度和
item. represents a pixel because we multiplied height by width and then we. transpose.
然后进行转置.
it so put
所以稍稍往前一点. 将负一和负二进行转置. 这将把这种
it so put
所以稍稍往前一点. 将负一和负二进行转置. 这将把这种
it back little before transpose the minus one with minus two. this will transform. this
形状转换成这种形状.
shape into this shape
所以我们把这一个放回去, 所以这一个在前, 特征变成最后
so :we put back this one so this one comes before and features becomes. the last. one
的, 这就像我们在tran is former 模型中做注意力时一样.
something like this and okay so as you can see from this. tensor here this is. like
的, 这就像我们在transformer 模型中做注意力时一样.
when we do the
所以在transformer 模型中, 我们有一个词元序列, 每个词元代表
attention in. the transformer model so in the transformer. model we have a sequence. of
所以在transformer 模型中, 我们有一个词元序列, 每个词元代表
tokens each tokens representing for example. a. word.
一个词, 例如, 注意力基本上计算每个词元之间的注意力.
tokens each tokens representing for example. a. word.
个词, 例如, 注意力基本上计算每个词元之间的注意力.
一个词, 例如, 注意力基本上计算每个词元之间的注意力.
and. the attention basically calculates the attention between-each. token so. how. do-two
所以两个词元是如何相互关联的
:tokens are related to each other in this case we can think. of it. as a sequence. of
在这种情况下, 我们可以把它看作是一个像素序列, 每个
:tokens :are related to each other in this case we can. think. of it. as a sequence. of
在这种情况下, 我们可以把它看作是一个像素序列, 每个
pixels each
在这种情况下:我们可以把它看作是一个像素序列, 每个
pixel with its own embedding which is the features. of. that. pixels and. were late
像素都有自己的嵌入, 即该像素的特征, 我们将像素相互
pixels to. each other and then we do the attention which is a self-attention. in which.
然后我们进行注意力计算, 这是一个自注意力, 其中自
pixels to. each other and then we do the attention which is a self-attention. in which
注意力意味着查询、键和值是相同的输入, 这不会改变
self-attention means that the query key and values. are the same input.
注意力意味着查询、键和值是相同的输入, 这不会改变
注意力意味着查询、键和值是相同的输入, 这不会改变
and this doesn't change the shape so this one remains the same then we transpose back.
逆变换, 因为我们把它变成这种形式只是为了进行注意力
and. we. do. the inverse transformation because we put it in. this. form only. to. do. the.
所以现在我们转置, 所以我们把这个转换成特征, 然后是
so. now we transpose'so we take this one and we convert it. into features
所以现在我们转置, 所以我们把这个转换成特征, 然后是
高度和宽度, 然后我们再次通过重新查看张量来移除乘法.
高度和宽度, 然后我们再次通过重新查看张量来移除乘法.
and. then. height and width and then again we remove this multiplication. by viewing
所以nc hw. 所以我们从这里到这里. 然后我们加上残差连接并
again. the tensors on chw so we go from here to here then we. add. the residual
所以nc hw. 所以我们从这里到这里. 然后我们加上残差连接并
connection and we return x that's it.
返回 X 就是这样
connection and we return x that's it
残差连接不会改变输入的大小. 我们返回一个这种形状的
the. residual connection witt not change the size of the of. the input and we return. a
张量. 让我也检查一下这里的残差连接是正确的.
tensor of. this shape here tet me check also the residual connection here. is correct.
好的, 现在我们已经有了注意力模块, 让我们也构建自
okay'now
好的, 现在我们已经有了注意力模块, 让我们也构建自
we. that. have also been'the attention block t et'sbuild also. the self attention since
好的, 现在我们已经有了注意力模块, 让我们也构建自
we are "
注意力, 因为我们正在构建注意力和注意力, 因为在稳定
注意力, 因为我们正在构建注意力和注意力, 因为在稳定
building-the attentions and the attentions because we have. two. kind of attention in
一种是自注意力, 另一种是交叉注意力我们需要构建
:the-stable diffusion one is called the self attention and. one. is. the. cross attention
一种是自注意力, 另一种是交叉注意力, 我们需要构建
and'we "
一种是自注意力, 另一种是交叉注意力, 我们需要构建
need to. build both so let'sgo buit dit in a separate class called attention-and okay -
两者. 所以让我们在一个叫做attention 的独立类中构建它. 好
need to. build both so let'sgo buit d itin a'separate class called. attention and okay
吧, 所以再次, 导入torch.
need to. build both so let'sgo build it in a separate class called. attention and okay
吧, 所以再次, 导入torch.
so again import torch
好的, 我想你们可能想在构建之前复习一下注意力机制.
:o Okay, L think you guys may be want to review the attention before building it.
所以让我们来复习一下.
So let'sgo review it.
我在这里打开了我的关于transformer 模块注意力机制的视频的
I have. here opened my slides from m
我在这里打开了我的关于tra or mer 模块注意力机制的视
幻灯片
幻灯片.
transformer module.
sequence length = number of heads d/ h
-所以自注意力基本上是一种方式, 特别是在语言模型中, 是
size of the e
种让我们将词元相互关联的方式
s H qut nce length = number of heads • d/h
size of the e 所以我们从一个词元序列开始, 每个词元都有一个大小为 D
H quan ce length size of the embedding vee model 的嵌入, 我们将其转换为查询、键和值, 其中自
size of the emo e 注意力中的查询、键和值是相同的短阵, 相同的序列.
= number of heads-d/h
size of the eml 我们将它们乘以 WQ矩阵、即 WQ、 WK和 WV、它们
是参数矩阵.
= number of heads
size of th 然后我们沿着 Dmodel 维度将它们分割成多个头, 这样我们
number of head t 就可以指定我们想要多少个头.
= size of the embedding vec to = number of heads
size of the 在我们的例子中, 我们将要做的注意力实际上只有一个头
number of head t tion that we wild o here is ad 我稍后会展示给你.
= si2e of the embedding vec to sequence length = number of heads 我稍后会展示给你.
show you later.
= number of heads
然后我们为每个头计算注意力. 然后我们通过将这些头连接在
起重新组合
Multi ead( Q. K, V) = Concat(head, head)w
我们将这个连接输出的矩阵乘以另一 一个称为 WO 的短阵
head )w
size of the er
rumber of head
只有 个头而不是多头, 那么我们将不会进行这种分割操作
bef of headt 我们将只进行与w和wo的乘法. 好吧, 这就是首
head)w
size of the 注意力是如何工作的. 所以在自注意力中, 我们从这个相
size of the emo e 的输入矩阵中得到查询、键和值, 这就是我们要构建的
= size of the embedding vecto
s Hqutince length 内容.
Multi f ead( Q. K, V) = Concat(bead, head,)w
d/ho
= number of heads matrix input and this is what we are go in
内容.
所以我们有头的数量, 然后我们有嵌入.
So we have the number of heads, then we have the embedding
那么每个词元的嵌入是什么?
So what is the embedding of each token?
但在我们的例子中, 我们不是在谈论词元, 我们将在谈论
But in our. case, we are not talking about tokens, we will talk about pixels and we
像素, 我们可以认为每个像素的通道数是该像素的嵌入
can. think. that the number of channels of each pixel is the. embedding of. the. pixel.
所以嵌入, 就像在原始的 Transformer 中一样, 嵌入是捕捉词义的
So. the embedding, just like in the original transformer, the. embeddings. are the. kind
所以嵌入, 就像在原始的 Transformer 中一样, 嵌入是捕捉词义的
of vectors that capture the meaning of the. word.
向量.
of vectors that capture the meaning of the. word.
在这种情况下, 我们有通道, 每个通道, 每个像素由许多
In. this case, we have the channels, each channel, each pixel represented. by. many
这里我们也有 W 矩阵的偏置, 这在原始的 Transformer 中是没有的
w Here we. have also the bias for the W matrices which we don't. have in the-original
这里我们也有 W 矩阵的偏置, 这在原始的 Transformer 中是没有的
transformer.
现在让我们定义 W矩阵, 即 WQ、 WQ和 WV.
Nowlet'sdefine the Wmatrices, sow Q, w Qandwv.
我们将它表示为一个大的线性层.
We will represent it as one big linear. layer.
而不是将其表示为三个不同的矩阵, 我们可以说它是一个大的
instead. of. representing it as three different matrices it's. possible. we just. say. that.
3乘以 Dembedding 矩阵, 如果有偏置的话, 偏置在投影
instead. of. representing it as three different matrices it's. possible. we just. say. that.
中, 在投影偏置中. 所以这意味着代表投影, 因为我们是
it's big matrix 3 by Dembedding and the bias is if we want it. so in. projection. in
中, 在投影偏置中. 所以这意味着代表投影, 因为我们是
projection
在应用注意力之前对输入进行投影, 然后有一个自投影, 这
bias. so. this means stands for in projection because we is the projection. of. the input.
是在我们应用注意力之后. 所以 WO 矩阵.
before we. apply the attention and then there is a auto projection which is. after we..
是在我们应用注意力之后. 所以 WO 矩阵.
applied the attention so the Wo matrix
所以, 正如你记得的, 这里的 WO 矩阵实际上是demod al
Soas. you remember here, the Wo matrix is actually demo dal byde modal.
正如你记得的 这里的w O 矩阵实际上是de modal
乘以demod all
输入也是de modal 乘以de modal
输入也是de modal 乘以de modal.
The input is also demo dal by de modal.
而这正是我们所做的.
And this is exactly what we did...
但我们这里有三个, 所以是三乘以de modal.
But we have three of them here, so it's three by-de modal.
然后我们保存了头部的数量.
And then we save the number of heads.
然后我们保存了每个头部的维度.
And then we save the dimension of each. head..
每个头部的维度基本上意味着如果我们有多头, 每个头部将
a:the dimension of each head basically means that if we have multihead each head will
每个头部的维 有多头 每个头部浆
所以我们需要保存
大小是多少?
所以d model 除以头部的数量 嵌入除以头部的数量 让我们
所以d model 除以头部的数量, 嵌入除以头部的数量, 让我们
d model. divided by the number of heads embedding divided. by. the. number. of heads. let's
实现前向传播. 我们也可以应用一个掩码.
d model. divided by the number of heads embedding divided. by. the. number. of heads. let's
实现前向传播. 我们也可以应用一个掩码,
implement the forward we can also apply. a. mask
正如你记得的, 掩码是一种避免将一个特定token 与它之后的
As you remember, the mask is a way to avoid relating tokens, one particular. token.
token 关联, 而只与它之前的token 关联的方法.
:with. the. tokens that come after it, but only with the token. that come. before. it
这被称为因果掩码.
And this is called the causal mask.
如果你真的不明白这里的注意力机制发生了什么, 我强烈建议
v if you really are not understanding what is happening here in the attention i highly
你观看我之前的视频, 因为那里解释得非常清楚, 我会.
a :recommend you watch my previous video because it's explained. very well. and iwill. i..
如果你观看, 不会花太多时间, 我认为你会学到很多.
you watch
如果你观看, 不会花太多时间, 我认为你会学到很多.
it. will take not so much time and i think you will learn a lot so. the first. thing we
所以我们做的第一件事是提取形状.
it. will. take not so much time and i think you will learn a. lot. so. the first. thing we.
所以我们做的第一件事是提取形状.
do is extract the shape then we extract
然后我们提取批次大小、序列长度, 嵌入等于输入形状.
do is extract the shape then we extract
然后我们提取批次大小、序列长度, 嵌入等于输入形状.
然后我们提取批次大小、序列长度, 嵌入等于输入形状.
batch size. sequence length and the embedding is equal to input shape then. we say-that
然后我们说我们将把它转换成另一种形状, 我稍后会告诉你
we. will convert it into another shape that I will show you later why this. is called..
为什么这被称为中间形状.
:we. will convert it into another shape that I will show you later. why this. is called
为什么这被称为中间形状.
the interim shape intermediate shape.
然后我们应用查询、键和值, 我们分割, 我们应用投影
then. we apply the query key and value we split we apply the in. projection. so. the wq.
即w q、wk和wv 矩阵到输入, 并将其转换为
then. we apply the query key and value we split we apply. the. in. projection. so. the wq
所以查询、键和值等于. 我们乘以它, 但然后我们用块
wk and wv. matrix to the input and we convert it into query key-and values. so. query
所以查询、键和值等于. 我们乘以它, 但然后我们用块
key and
所以查询、键和值等于. 我们乘以它, 但然后我们用块
values :are. equal to we multiply it but then we divide it so. with. chunk. as is how you
分割, 正如我之前向你展示的什么是块.
before what is chunk
基本上, 我们将输入与表示wq 、wq 的大矩阵相乘
basically. we will multiply the input with the big matrix that represent wq, wq and we
但然后我们将其分割回三个较小的矩阵.
but. then. we split it back into three smaller matrices this is. the-same as. applying
这相当于应用三个不同的投影, 而不是.
three
这相当于应用三个单独的投影, 但也可以将它们组合成一个大
. different projections instead of.. is the same as applying three separate in
矩阵. 这就是我们要做的.
projections but it's also possible to combine it in one big matrix. this. what we will
矩阵. 这就是我们要做的.
do basically it will convert batch size..
基本上, 它将批次大小、序列长度维度转换为批次大小
do basically it will convert batch size..
基本上, 它将批次大小、序列长度维度转换为批次大小
基本上, 它将批次大小、序列长度维度转换为批次大小
sequence length dimension into batch size sequence length. dimension multiplied by..
序列长度维度乘以三, 然后, 通过使用块, 我们沿着最后
oc three and then by using chunk we split it along the last. dimension. into. three.
一个维度将其分割成三个不同形状的张量:批次大小、序列
:oc three and then by using chunk we split it along the. last. dimension. into. three.
长度和维度.
different tensors of shape batch size sequence length. and. dimension un
好的, 现在我们可以根据头部的数量将查询、键和值分割成
Okay, now we can split the query key and values in the number of. heads, according-to
好的, 现在我们可以根据头部的数量将查询、键和值分割成
the number of heads.
这就是我们构建这种形状的原因, 这意味着将最后一个维度
This. is why we built this shape, which means split the last dimension. into. n. heads.
维度到批次大小、序列长度, 然后是 H, 即头部的数量
andvalues v dotview wonderful this will convert okay. let'swrite. it batch size
维度到批次大小、序列长度, 然后是 H, 即头部的数量
sequence length dimension in tom
维度到批次大小、序列长度, 然后是 H, 即头部的数量
维度到批次大小、序列长度, 然后是 H, 即头部的数量
batch size, sequence length, then H, so the number of heads, and each. dimension
每个维度除以头部的数量.
divided by the number of heads..
所以每个头部将观察整个序列, 但只观察每个token (在这种
So each. head will watch the full sequence, but only a part of the. embedding of. each
情况下是像素)嵌入的一部分, 并将观察这个头部的
So. each. head will watch the full sequence, but only a part of. the. embedding of. each
情况下是像素)嵌入的一部分, 并将观察这个头部的
token, in this case pixel, and will watch this part. of. the head.
所以整个维度, 嵌入除以头部的数量.
So. the full dimension, the embedding divided by. the. number of heads.
然后这将转换它, 因为我们也在转置. 这将转换为批次
:and then. this will convert it because we are also transposing. this. will. convert. it.
所以每个头部将观察整个序列, 但只观察嵌入的一部分.
into batch. size edge sequence length and then dimension edge so. each head. will. watch
所以每个头部将观察整个序列, 但只观察嵌入的一部分.
the all the
所以每个头部将观察整个序列, 但只观察嵌入的一部分.
sequence. but only a part of the embedding we then calculate the the attention. just..
然后我们计算注意力, 就像公式一样.
sequence. but only a part of the embedding we then calculate the. the attention just..
然后我们计算注意力, 就像公式一样.
like the formula so query multiplied by the transpose of. the keys. s.
所以查询乘以键的转置, 即查询矩阵乘以键的转置. 这将
like the formula so query multiplied by the transpose of. the keys. s.
所以查询乘以键的转置, 即查询矩阵乘以键的转置. 这将
所以查询乘以键的转置, 即查询矩阵乘以键的转置. 这将
is. the. query matrix multiplication with the transport of the keys this. will return. a
返回一个大小为批次大小和序列长度乘以序列长度的矩阵.
is. the query matrix multiplication with the transport of the keys this. will. return a
然后我们可以应用掩码.
matrix of. size batch size and sequence length by sequence. length. we can. then apply
然后我们可以应用掩码.
the mask
如你所记, 掩码是我们计算注意力时应用的东西.
As you remember, the mask is something that we apply when. we calculate. the. attention.
如你所记
如果我们不希望两个token 我们基本上会替换它们
的值
在这个矩阵中, 我们在应用soft max 之前用负无究大香
以便soft max 使其变为零
所以我们在这里做的是这个.
So this is what we are doing here
我们首先构建掩码.
We first build the mask.
这将创建一个因果掩码, 基本上是一个掩码, 其中上三角
This. will. create a causal mask, basically a mask where the upper triangle, so above
部分, 即主对角线上方, 由一组成.
the principal diagonal, is made up of one.
One, ones, a lot of ones.
然后我们用负无穷大填充它
And then we fill it up with minus infinity
掩码, 哎呀, 不是掩码, 而是权重.
Masked, oops, not mask, but weight
掩码填充, 但使用掩码, 我们放入负无穷大, 像这样:如
masked. fill but with mask and we put minus infinity like. this as. you remember. the.
你所记, Transformer 的公式是查询乘以键的转置, 然后除以模型的
formula. of the transformer is a query multiplied by the transpose. of. the-keys and.
你所记, Transformer 的公式是查询乘以键的转置, 然后除以模型的
then divided by
你所记, Transformer 的公式是查询乘以键的转置, 然后除以模型的
the square. root of the model so this is what we will do now. so divided. by. the square
所以我们现在要做的是, 除以模型的平方根, 除了头部分
the square. root of the model so this is what we will do now. so divided. by. the square
所以我们现在要做的是, 除以模型的平方根, 除了头部分
root of the model except of the head.
所以我们现在要做的是, 除以模型的平方根, 除了头部分
and :then we. apply the soft max we multiply by the wo matrix we-transpose. back. so. we
然后我们应用soft max, 乘以 WO 矩阵, 转置回来. 所以
and. then we. apply the soft max we multiply by the wo matrix. we. transpose. back. so. we.
我们想要移除, 现在我们想要移除头维度, 所以输出等于.
want to. remove now we want to remove the head dimension so. output is. equal. to.
让我写一些形状. 这是什么?
let me write some shapes, so what is. this?
这等于批次大小, 序列乘以序列, 进行矩阵乘法, 批次大小.
this is equal to batch size sequence by sequence multiplied, so matrix. multiplication
这等于批次大小, 序列乘以序列, 进行矩阵乘法, 批次大小.
with batch size this will result in to batch. size..
这将导致批次大小, h, 序列长度和维度除以h.
with batch size this will result in to batch. size..
这将导致批次大小, h, 序列长度和维度除以h.
然后我们转置这个, 这将导致:所以我们从这个开始, 它
h sequence length and dimension divided by h this we then. transpose and. this. will.
然后我们转置这个, 这将导致:所以我们从这个开始, 它
result into so we start with this one and it. becomes
变成等等, 我这里放了太多括号.
result into so we start with this one and it. becomes
变成等等, 我这里放了太多括号.
变成等等, 我这里放了太多括号.
Wait, I put too many parentheses here
批次大小, 序列长度, 等等, 和维度.
Batch size, sequence length, etch, and dimensions.
Ok.
Okay
然后我们可以重塑为输入, 像初始形状, 所以这个.
Then we can reshape as the input, like the initial shape, so. this one.
然后我们应用输出投影, 所以我们乘以wo矩阵, 好的
cand then we apply the output projection so we multiply by the. wo matrix okay
这是自注意力.
这是自注意力
This is the self
现在让我们回到继续构建解码器
Now let'sgo back to'continue building the decoder.
目前我们已经构建了注意力块和残差块, 但我们还需要构建
For. now we have built the attention block and the residual. block, but we-need. to
目前我们已经构建了注意力块和残差块, 但我们还需要构建
build the decoder.
而且这也是我们将一个接一个应用的模块序列. 我们从卷积
and. also. this one is a sequence of modules that we will apply one after another we
形状, 形状会变化, 但你已经理解了编码器中的概念.
start with. the convolution just like before now i will not write. again the. shape. the.
形状, 形状会变化, 但你已经理解了编码器中的概念.
shapes
我们在编码器中
change but you got the idea in th
让我在这里给你展示一下, 这里,
Recor ln put uc ted
Input 在编码器中, 我们不断缩小图像的尺寸, 直到它变得很小.
Recor ln put uc ted
Input truc ted 在解码器中, 我们需要恢复到图像的原始尺寸.
In the decoder we need to neturn to the original size of the image
Input 所以我们从潜在维度开始, 然后恢复到图像的原始维度.
Input 所以我们从潜在维度开始, 然后恢复到图像的原始维度.
So west an wich the latent dimension. and we. return. to. the. origin a dimension
Input Recon i
truc ted
卷积.
Convolution.
我们从四个通道开始, 最终也输出四个通道.
So we start with four channels and we output. four channels.
然后我们再进行一次卷积操作
Then we have another convolution
我们达到500.
Wegoto500.
然后我们有一个残差块, 和之前一样.
Then we have a residual block, just like. before.
然后我们有一个注意力模块
Then we have an attention block.
然后我们有一系列的残差块, 总共有四个.
Then we have a bunch of residual blocks, and we have four. of. them.
让我复制一下.
Let me copy.
Ok.
Okay.
现在, 残差块, 让我在这里写一些形状.
Now, the residual blocks, let me write some. shapes here.
在这里, 我们达到了一种状态, 其中我们有补丁大小, 有
a0*ck(512, 512),
Here. we arrived to a situation in which we have patch size, we have 512 features. and.
512 个特征, 图像的大小仍然没有增长, 因为我们没有
Here. we arrived to a situation in which we have patch size, we have 512 features. and
任何会使其增长的卷积.
the. size. of the image still didn't grow because we don't have any convolution. that
任何会使其增长的卷积.
will make it grow.
这个当然会保持不变, 因为它是一个残差块等等
This one, of course, will remain the same because it's a residual. block and etc
现在, 为了增加图像的大小, 月 所以现在图像实际上是高度
Now, to. increase the size of the image, so now the image is. actually. height divided.
除以8, 记住, 高度是512, 这是我们正在处理的图像的
by8, which height, as you remember, is512, the size. of. the image that we are
除以8, 记住, 高度是512, 这是我们正在处理的图像的
working with.
所以这里的这个维度是64乘64.
So this dimension here is 64by. 64.
我们如何增加它?
How can we increase it
我们使用一个称为上采样的模块.
We use one module called up sample...
上采样. 我们必须把它想象成当我们调整图像大小时. 所以
the up sample we have to think of it like um when were size an image so. imagine. you.
想象一下, 你有一张64乘64的图像, 你想把它
theup sample we have to think of it like um when were size an image so. imagine. you
转换成128乘128.
have-an image that is 64by64 and you want to transform. into 128by128. the the. up.
上采样会做到这一点, 就像我们调整图像大小时一样.
have-an image that is 64by64 and you want to transform. into 128by128. the theup
上采样会做到这一点, 就像我们调整图像大小时一样.
sample will
上采样会做到这一点, 就像我们调整图像大小时一样.
o do it. just like when were size an image so it will replicate the. pixels. will um.
所以它会复制像素, 嗯, 两次, 所以在维度上, 比如向
do it. just like when were size an image so it will replicate the. pixels. will um
所以它会复制像素, 嗯, 两次, 月 所以在维度上, 比如向
所以它会复制像素, 嗯, 两次, 所以在维度上, 比如向
v. twice so. along the dimensions right and down for example. twice so that. the. total.
右和向下, 两次, 这样像素的总数, 高度和宽度
A:v:twice so along the dimensions right and down for example. twice so that. the. total.
实际上是翻倍的. 这就是上采样
amount. of pixels that the height and the width actually doubles. and this. is. the. up
实际上是翻倍的. 这就是上采样
sample
基本上, 它只是复制每个像素, 以便沿着每个维度的这个
:o basically it wil I just replicate each pixel so that by this. scale. factor along each
基本上, 它只是复制每个像素, 以便沿着每个维度的这个
dimension so this one becomes batch. size
比例因子, 所以这个变成了批次大小, 除以8. 除以8
dimension so this one becomes batch. size.
比例因子, 所以这个变成了批次大小, 除以8. 除以8
比例因子, 所以这个变成了批次大小, 除以8. 除以8
divideby8. withdivideby 8becomes as we see here 8dividedby. 4 and with. divided
然后我们有一个卷积残差块, 所以我们有卷积:2 D, 512
by4. then we have a convolution residual blocks so we have. convolutions 2 D..
然后我们有一个卷积, 残差块, 所以我们有卷积:2 D, 512
到512. 然后我们有一些512乘500的小块, 但在这种
到512. 然后我们有一些512乘500的小块, 但在这种
11. to512. then we have little blocks of 512by500 but in. this case we. have three. of
所以我们还有一次会使图像大小翻倍的上采样, 比例因子为2 them. then. we have another up sample this will again double. the size. of the. image so. we
所以我们还有一次会使图像大小翻倍的上采样, 比例因子为2.
have
所以我们还有一次会使图像大小翻倍的上采样, 比例因子为2 another one that will double the size of the image and by-a. scale factor. of two. so
所以让我们这样写现在变成了除以2, 所以它会加倍
divideby. 4with512channels so let'swrite it like this will. become divide. by. 2now
图像的大小. 所以现在我们的图像是256乘256.
so. it will. double the size of the image sonow ourimageis. 256by 256then-again. we
然后我们再次有一个卷积, 然后我们又有三个残差块, 但
so it will. double the size of the image so now ourimageis256by 256then-again. we
这次我们减少了特征的数量, 所以是256, 然后是256到256.
这次我们减少了特征的数量, 所以是256, 然后是256到256.
but. this. time we reduce the number of features so256and. then. it's. 256. to. 256. okay
好的, 然后我们又有一次上采样, 这会再次使图像大小
but. this. time we reduce the number of features so256and. then. it's. 256. to. 256. okay
翻倍, 这次我们从除以2到除以2, 直到原始大小
:then we. have another up samplings which will again double. the size of. the image
翻倍, 这次我们从除以2到除以2, 直到原始大小
翻倍, 这次我们从除以2到除以2, 直到原始大小
wand. this. time we will go from divide by 2todivide by2up. to the original. size. and.
然后我们又有一次卷积, 这次是256, 因为这是新的特征
because the. number of channels has changed we are not 512 anymore okay and then. we :
然后我们又有一次卷积, 这次是256, 因为这是新的特征
have
然后我们又有一次卷积, 这次是256, 因为这是新的特征
another. convolution this case with 256 because it's the. new number of. features.
然后我们又有一些残差块, 它们会减少特征的数量.
w : Then we. have another bunch of residual blocks that will decrease. the number. of.
然后我们又有一些残差块, 它们会减少特征的数量.
features.
所以我们从256到128.
Sowegoto256to128.
我们最后有一个组归一化.
We have finally a group norm..
32是组的大小, 所以在计算u和之前, 我们将
32is. the-group size, so we group features in groups of 32. before. calculating the. mu
特征分组为32 个一组进行归一化.
and the sigma before normalizing.
我们将通道数定义为128, 这是我们拥有的特征数量.
And we define the number of channels as128, which is the. number of. features that. we
我们将通道数定义为128, 这是我们拥有的特征数量.
have.
所以这个组归一化会将这128个特征分成32个一组.
Sothis. group normalization will divide the se128 features into groups. of 32
然后我们应用 SIL U.
Then we apply the SIL U.
最后的卷积会将数据转换成具有三个通道的图像, 即 RGB the. final. convolution that will transform into an image with. the. three channels. so.
通过应用这里的卷积, 这些卷积不会改变输出的尺寸.
rgb. by applying these convolutions here which doesn't change the size of. the output.
所以我们从一个图像开始, 它是批次大小, 128高度宽度
sowe will
所以我们从一个图像开始, 它是批次大小, 128高度宽度.
go:from an. image that is batch size 128 height width why. height. width because. after.
因为在最后一次上采样后, 我们恢复到原始尺寸, 变成只有
go :from an. image that is batch size 128 height width why. height. width because. after..
三个通道的图像. 这就是我们的解码器.
现在我们可以写前向方法了
Now we can write the forward method.
如果我在中间放了很多空格, 我很抱歉, 但这样不容易
I'm. sorry. if I'm putting a lot of spaces between here, but. otherwise it's. easy. to get.
迷失, 也能清楚我们在哪里
lost and not understand where we are.
所以这里解码器的输入是我们的潜在变量, 即批次大小4
So. here. the input of the decoder is our latent, so it's batchsize. 4, height. divided
高度除以8, 宽度除以8.
by 8, width divided by 8.
正如你记得的, 在编码器中我们最后做的就是按这个常数
as you. remember here in the encoder the last thing we do. is. be scaled. by. this
缩放, 所以我们抵消这个缩放, 也就是反向这个缩放215
asyou remember here in the encoder the last thing we do. is. be scaled. by. this
然后我们通过解码器运行它, 然后返回×, 即批次大小
A constant so we nullify this scaling so we reverse this scaling 215. and then. we run. it
然后我们通过解码器运行它, 然后返回×, 即批次大小
through the decoder and then return. x
然后我们通过解码器运行它, 然后返回×, 即批次大小
然后我们通过解码器运行它, 然后返回×, 即批次大小
which is batch size, tree, height and width.
树, 高度和宽度.
which is batch size, tree, height and width.
sis, 1, mlgmt, xiesg
让我也写下这个解码器的输入, 就是这个.
Let me also write the input of this decoder, which. is. this one.
我们已经有了.
511, 512)
We already have it.
好的, 这是我们的变分自编码器.
Okay, this is our variational auto encoder.
到目前为止, 让我们回顾一下.
So far, let'sgo review.
到目前为止, 让我们回顾
Recor ln put uc ted
我们正在构建稳定扩散的架构
We are building. our architecture of the stable diffusion
到目前为止, 我们已经构建了编码器和解码器.
So far we hame built the encoder and the decoder
但现在我们必须构建单元, 然后是 CLIP 文本编码器, 最后
是连接所有这些部分的管道.
final l we ha wefo build the pipeline that will connect all of the s
这将是一段漫长的旅程, 但实际上构建这些东西很有趣
因为你可以学到它们工作的每一个细节,
接下来我们要构建的是文本编码器.
detail of hew the sy work so the next thing that we
所以这是一个 CLIP 编码器, 它允许我们将提示编码成嵌入
然后我们可以将其输入到这个单元模型中. 让我们构建这个
CLIP 编码器, 当然, 我们将使用预训练版本.
通过下载词汇表, 我会向你展示它是如何工作的. 让我们
通过下载词汇表, 我会向你展示它是如何工作的. 让我们
wil show yom how it. works. so. lets. start we-go. to visual. studio e ode wee neate
开始吧. 我们去 Visual Studio Code yom how it works. so. lets. start we-go to visual. studio @ode
开始吧. 我们去 Visual l Studio Code.
. i will show you how it wonks so let'sstart we go to visual studio. code we. create a.
我们在st 文件夹中创建一个名为clippy 的新文件, 在这里
i will show you how it works so let'sstart we go to visual studio. code we. create. a
我们开始导入通常的东西, 我们还导入了自注意力, 因为
我们将会用到它. 所以, 基本上, CLIP 是一个非常类似于
我们将会用到它. 所以, 基本上, CLIP 是一个非常类似于
and. we. also import self-attention because we will be using. it so. basically. clip. is. a.
Transformer 编码器层的层.
wand. we. also import self-attention because we will be using. it so. basically. clip. is. a
所以, 正如你记得的 Transformer, 让我在这里向你展示 Transformer.
layer very. similar to the encoder layer of the transformer. so. as you. remember. the.
所以, 正如你记得的 Transformer, 让我在这里向你展示 Transformer.
transformer let me show you here.
所以, 正如你记得的 Tran
sforme r, 让我在这里向你展示
Transfor
mere
这是 Transformer 的编码器层 力机制和前馈网络组成
size of the 有许多这样的块一个接一个地应用. 我们还有一个表示句子
size of the 中每个词位置的东西, 在 CLIP 中我们也会有类似的东西
= size of the em oedding vecto
Mslti Heat( Q, K, V) = Concat (head, head )w = number of heads
所以我们需要构建一个非常类似的东西, 实际上这也是为什么
Transformer 模型非常成功的原因
size of the em be number of head t 自的. 所以让我们开始构建它. 首先, 我将构建模塑的
for Rh
目的. 所以让我们开始构建它. 首先, 我将构建模型的
o for. this purpose and so let'sgo to build it the first thing we. wil. build. I will.
所以让我们构建 CLIP, 它有一些嵌入. 嵌入允许我们将词转换
build. first the skeleton of the model and then we will build each-block. so. let's
所以让我们构建 CLIP, 它有一些嵌入. 嵌入允许我们将词转换
build clip
所以让我们构建 CLIP, 它有一些嵌入. 嵌入允许我们将词转换
and this has. some embeddings the embeddings allow us to convert. the tokens so. as. you.
所以, 正如你记得的, 当你有一个由文本组成的句子时
and this has. some embeddings the embeddings allow us to convert. the tokens. so. as. you.
首先你将其转换为数字, 其中每个数字表示词汇表中词的
remember in when you have a sentence made up of text. first you convert it into
位置, 然后你将其转换为嵌入, 其中每个嵌入表示一个大小
each. number indicates the position of the token inside of the vocabulary. and. then-you
为512 的向量(在原始的 Transformer 中), 但在 CLIP 中
original transformer but here in clip it's the size. is. 768
为512 的向量(在原始的 Transformer 中), 但在 CLIP 中
为512 的向量(在原始的 Transformer 中), 但在 CLi P 中
and each. vector represents kind of the meaning of the word or the token. captures. so
大小是768, 每个向量表示词或词元的某种意义. 这就是
and. each. vector represents kind of the meaning of the word or the token. captures. so.
嵌入, 稍后我们会定义它. 我们需要词汇表的大小.
:e this. is. an embedding and later we define it we need the vocabulary size. the
词汇表大小是49408. 我是直接从文件中获取的
:othis is. an embedding and later we define it we need the vocabulary size. the -.
词汇表大小是49408. 我是直接从文件中获取的
12. itook it directly from the file this is the embedding size and. the sequence.
这是嵌入大小和序列长度, 我们可以拥有的最大序列长度
49408i took it directly from the file this is the embedding size. and. the sequence
因为我们需要使用填充, 所以是77. 因为我们实际上应该
length. the. maximum sequence length that we can have because-we need. to. use. the.
因为我们需要使用填充, 所以是77. 因为我们实际上应该
paddingisa77
因为我们需要使用填充, 所以是77. 因为我们实际上应该
Because. we should actually use some configuration file to. save, but because we-will.
使用一些配置文件来保存, 但由于我们将使用预训练的稳定
Because. we should actually use some configuration file to. save, but because we. will
扩散模型, 大小已经为我们固定了, 但在未来, 我会重构
:be. using-with the pre-trained stable diffusion model, the size. is. already. fixed. for.
扩散模型, 大小已经为我们固定了, 但在未来, 我会重构
us, but in
扩散模型, 大小已经为我们固定了, 但在未来, 我会重构
the. future I will refactor the code to add some configuration actually. to. make. it.
代码以添加一些配置, 使其更具扩展性.
the. future I will refactor the code to add some configuration. actually. to. make. it.
代码以添加一些配置, 使其更具扩展性.
more extensible.
这是一个层列表, 我们称之为 CLIP 层.
This is a list of layers, each we call it a clip layer.
我们有12个, 这表示多头注意力的头数, 然后是嵌入
We:have. this12which indicates the number of heads of the multi-head attention and.
我们有12 个, 这表示多头注意力的头数, 然后是嵌入
then the embedding size which is 768
大小, 即768.
then the embedding size which is 768
我们有12个这样的层
Andwe have 12of these layers.
然后我们有层归一化, 层范数, 我们告诉它有多少特征
Then we have the layer normalization, layer norm
然后我们有层归一化, 层范数, 我们告诉它有多少特征
所以是768, 然后我们定义前向方法, 这是张量, 这个返回
and :we. tell. him how many features, so768 and then we define. the. forward. method. this
所以是768, 然后我们定义前向方法, 这是张量, 这个返回
istensor and this one returns float tensor..
浮点张量, 为什么是长张量?
is tensor and this one returns float tensor.
浮点张量, 为什么是长张量?
因为输入 ID 通常是表示词汇表中每个词位置的数字.
Because. the input IDs are usually numbers that indicate the-position of. each. token.
因为输入 ID 通常是表示词汇表中每个词位置的数字.
inside of the vocabulary.
另外, 如果这个概念不清楚, 请去看我之前关于 Transformer 的
Also this. concept, please if it's not clear, go watch my previous. video about. the
视频, 因为当我们处理文本模型时, 那里解释得非常清楚
Also this. concept, please if it's not clear, go watch my previous. video about. the
好的, 首先我们将每个词转换为嵌入.
Okay, first we convert each token into embeddings.
然后, 大小是多少? 这里我们从一个批次大小序列长度转换
and. then so what is the size here we are going from batch. size sequence length. into
然后我们逐层应用编码器的所有层, 就像在 Transformer 模型中
batch. size sequence length and dimension where the dimension is. 768 then. we apply. one
然后我们逐层应用编码器的所有层, 就像在 Transformer 模型中
after one
然后我们逐层应用编码器的所有层, 就像在 Transformer 模型中
ec after-another all the layers of this encoder just like in the transformer model.
然后我们逐层应用编码器的所有层, 就像在 Transformer 模型中
一样, 最后我们应用层归一化, 最终返回输出, 输出当然
是序列到序列模型, 就像 Transformer 一样, 所以输入的形状应该
:oc where. the output is of course it's a sequence to sequence mode I just like the
与输出的形状匹配.
::oc transformer so the shape of the input should match the shape of the output..
所以我们总是通过模型获得序列长度.
So we always obtain sequence length by the model.
好的, 现在让我们定义这两个块.
Okay, now let's define these two blocks.
第一个是 CLIP 嵌入.
The first one is the clip embedding.
那么让我们来看 CLIP 嵌入.
So let'sgo clip embedding.
词汇表大小是多少?
How much is the vocabulary size?
嵌入大小是多少?
What is the embedding size?
和词的数量. 好的, 所以序列长度, 基本上, 和超级
and number of token okay so the sequence length basically. and. super okay-we define.
好的, 我们使用nn. Embedding 来定义嵌入本身.
and number of token okay so the sequence length basically. and super okay-we define.
好的, 我们使用nn. Embedding 来定义嵌入本身.
the embedding itself using nn. embedding just. like. always
就像往常一样, 我们需要告诉他嵌入的数量, 即词汇表
the embedding itself using nn. embedding just. like. always
就像往常一样, 我们需要告诉他嵌入的数量, 即词汇表
就像往常一样, 我们需要告诉他嵌入的数量, 即词汇表
we need. to tell him what is the number of embeddings, so. the vocabulary. size, and
大小, 以及每个嵌入词向量的维度.
what is the dimension of each vector of the embedding token.
然后我们定义一些位置编码, 现在, 如你所记, 原始 Transformer Then we define some positional encoding, so now,
然后我们定义一些位置编码, 现在, 如你所记, 原始 Transformer
然后我们定义一些位置编码, 现在, 如你所记, 原始 Transformer As you remember the positional encoding in the original. transformer are-given. by
中的位置编码是由正弦函数给出的, 但在这里的 CLIP 中
As you remember the positional encoding in the original. transformer are-given. by.
他们实际上不使用它们, 他们使用一些学习到的参数.
sy no zoid al functions but here in Clip they actually don't. use them, they. use some.
他们实际上不使用它们, 他们使用一些学习到的参数.
learned parameters.
所以他们有这些参数, 这些参数是在训练过程中由模型学习到
So they. have these parameters that are learned by the model during training. that. tell
像这样我们应用它们, 首先我们应用嵌入, 所以我们从批量
and embeddings like this we apply them so first we apply the. embedding so. we go. from
大小序列长度到批量大小序列长度维度, 然后, 就像在原始
batch size sequence length to batch size sequence. length. dimension.
大小序列长度到批量大小序列长度维度, 然后, 就像在原始
Transformer 中一样, 我们将位置编码添加到每个词上.
Transformer 中一样, 我们将位置编码添加到每个词上
And. then, just like in the original transformer, we add the. positional. encodings. to.
Transformer 中一样, 我们将位置编码添加到每个词上.
each token.
但在这个例子中, 如我所说, 位置嵌入不是固定的, 不像
But in. this case, as I told you, the positional embeddings are. not fixed, like not
但在这个例子中, 如我所说, 位置嵌入不是固定的, 不像
sinusoidal functions, but they are learned by. the. model.
所以它们是学习到的, 然后稍后当我们加载模型时, 我们会
So. they are learned and then later we will load these parameters. when we load. the
加载这些参数.
model.
然后我们返回这个x. 然后我们有 CLi P 层, 它就像 Transformer And. then. we return this x. Then we have the clip layer, which. is. just like. the layer.
模型的层
Transformer 模型的编码器.
of the transformer model, the encoder of the transformer model.
这个实际上什么都不返回
This one returns nothing, actually.
这个是错误的.
And'this one is wrong.
好的, 我们就像在工rans former 块中一样, 我们有预归一化
OK, we have, just like in the transformer block, we have the pre-norm, then we. have
然后是注意力, 然后是后归一化, 然后是前馈.
OK, we have, just like in the transformer block, we have the pre-norm, then we. have
所以 层归一化.
So, layer normalization.
然后我们有注意力, 这是一个自注意力.
Then we have the attention, which is a self-attention.
稍后我们将构建交叉注意力, 我会向你展示它是什么. 然后
later we will build the cross attention and i will show you. what. is it then. we have
然后我们有两个前馈层, 最后我们有前向方法. 最后
another layer normalization then we have two feedforward layers
然后我们有两个前馈层, 最后我们有前向方法. 最后,
这个方法接受张量并返回一个张量, 所以让我写成张量. 好
这个方法接受张量并返回一个张量, 所以让我写成张量. 好
and finally we have the forward method finally so this one takes. tensor and returns. a.
的, 就像 Transformer 模型一样.
and finally. we have the forward method finally so this one. takes. tensor and returns. a.
的, 就像 Transformer 模型一样.
tensor so. let me write it tensor okay just like the transformer. model okay let's. go.
好的, 让我们来看看. 我们有一些, 一堆残差连接, 如
tensor so. let me write it'tensor okay just like the transformer. model. okay let'sgo
size of the em 好的. 让我们来看看. 我们有一些, 一堆残差连接
size of the e 你所见. 这里有一个残差连接, 这里也有一个残差连接.
heed )w
这里我们有两个归一化, 一个在这里, 一个在这里
前馈, 就像在原始的 Transformer 中一样 我们有两个线性层
然后我们有这个多头注意力, 它实际上是一个自注意力,
head,)w
为它是相同的输入变成了查询、键和值, 所以让我们来
因为它是相同的输入变成了查询、 键和值. 所以让我们来
same. input that becomes query key and values so let's. do. it. the first. residual
做. 第一个残差连接:. 那么这个前向方法的输入是什么?
same. input that becomes query key and values so let's. do. it. the first. residual
它是一 个批次大小和嵌入的维度, 即768.
connection x so what is the input of this forward method it's a batch. size
它是 个批次大小和嵌入的维度, 即768.
它是一 个批次大小和嵌入的维度, 即768.
and. the dimension of the embedding which is 768 the first. thing we do is. we apply-the
我们做的第一件事是应用自注意力, 但在应用自注意力之前
and. the dimension of the embedding which is 768 the first. thing we do is. we apply-the
我们应用层归一化, 所以是层归一化1. 然后我们应用
self-attention but before applying the self-attention-we apply. the layer.
我们应用层归一化, 所以是层归一化1. 然后我们应用
normalization so
我们应用层归一化, 所以是层归一化1. 然后我们应用
layernorm1 then we apply the attention but with. the. causal mask
注意力 但带有因果掩码.
layer norm 1 then we apply the attention but with. the. causal mask
呢, 如你所记, 这里自注意力, 我们有因果掩码
uh as you remember here self-attention'we have the causal mask which. basically. means.
基本上意味着每个标记不能看到下一个标记, 所以不能与未来
uh as you remember here self-attention we have the causal mask which. basically. means :
的标记相关, 只能与它左边的标记相关. 这就是我们从文本
that. every. token can not watch'the next tokens so can not be. related. to future tokens
的标记相关, 只能与它左边的标记相关. 这就是我们从文本
but only
的标记相关, 只能与它左边的标记相关. 这就是我们从文本
the one. on the left of it and this is'what we want from. a. text model actually. we.
模型中想要的. 实际上, 我们不希望一个词看到它后面的
the one. on the left of it and this is what we want from a. text model actually. we
词, 而只希望看到它前面的词. 然后我们做这个残差连接.
dont. want. the one word to watch the words that come after it but. only. the. words. that
所以现在我们在等待, 现在我们在这里做这个连接. 然后
come before it then we do this residual connection. so. now. we are
所以现在我们在等待, 现在我们在这里做这个连接. 然后
f umber of head size of the ef 所以现在我们在等待, 现在我们在这里做这个连接. 然后
所以现在我们在等待, 现在我们在这里做这个连接. 然后
wait now we. are doing this connection here then we do the. feedforward layer again we
我们做前馈层. 再次我们有一个残差连接. 我们应用
have a resid u at connection we apply the normalization u
归一化.
have a residual connection we apply the normalization u
我没有写出所有的形状.
I'm not writing all the shapes.
如果你看我的在线代码, 我已经写出了所有的形状, 但主要
If. you watch my code online, 1 have written all of them, but mostly to. save. time
是为了节省时间, 因为在这里我们希望已经熟悉了 Transformer 的
If-you watch my code online, I have written all of them, but mostly to. save. time.
结构.
A:obecause here we are already familiar with the structure. of. the transformer,
所以我在这里没有重复所有的形状
So I am not repeating all the shapes here.
我们应用前馈的第一个线性层.
We apply the first linear of the feedforward.
然后, 作为激活函数, 我们使用glu 函数, 这就是我们
Then as activation function, we use the glu function.
然后, 作为激活函数, 我们使用glu 函数, 这就是我们
然后, 作为激活函数, 我们使用glu 函数, 这就是我们
and. that's. what we call the quick GLo function which is defined. like. this. x multiplied
所说的quick GLO 函数, 定义如下:x 乘以torch. sigmoid and. that's. what we call the quick GLo function which is defined. like. this. x multiplied
(1. 702乘以×). 就是这样. 所以这被称为
w. by. torch. sigmoid of 1. 702multiplied by x and that's it should. be like this so. this.
(1. 702乘以×). 就是这样. 所以这被称为
is called the quick GLO activation function also. here there is no
quick GLO 激活函数
is called the quick GLO activation function also. here there is no
这里也没有解释为什么我们应该使用这个而不是其他的. 他们
is called the quick GLO activation function also. here there is no
这里也没有解释为什么我们应该使用这个而不是其他的. 他们
这里也没有解释为什么我们应该使用这个而不是其他的. 他们
justification on why we should use this one and not another. one. they just saw. that. in.
只是发现, 在实践中, 这个对于这种应用效果更好.
justification on why we should use this one and not another. one. they just saw. that. in.
所以这就是我们在这里使用这个函数的原因. 所以现在然后
practice. this one works better for this kind of application. so. that's why. we are
所以这就是我们在这里使用这个函数的原因. 所以现在然后
using this
所以这就是我们在这里使用这个函数的原因. 所以现在然后
function. here so now and then we apply the residual connection. and finally return. x
这完全像 Transformer 的前馈层, 除了在 Transformer 中我们没有这个激活
: This is. exactly like the feed-forwardlayer of the transformer, except that in. the.
函数, 而是有re lu 函数.
transformer we don't have this activation function, but we. have. there lu function.
如果你记得在 LAM A中我们没有relu函数, 我们有zwi glu And. if you. remember in LAMA we don't have there lu. function, we. have. the. zwi glu.
但在这里我们使用quick glue 函数, 我实际上不太熟悉.
: But. here. we are using the quick glue function, which. l actually-am-not so. familiar.
但在这里我们使用quick glue 函数, 我实际上不太熟悉.
with.
但我认为它对这个模型效果很好, 所以我保留了它.
But I think that it works good for this model and. I just kept it.
所以现在我们在这里构建了我们的文本编码器, clip, 正如你
So now we. have built our text encoder here, clip, which. is very small as you can. see..
而我们接下来要构建的是我们的单元.
And our next thing to build is our unit..
所以我们已经构建了变分自编码器, 编码器部分和解码器
部分.
现在我们接下来要构建的是这个单元.
Now the next thing we hawe to build is this wnit
如你所记、单元是那个网络. 它会给定一些噪声图像和噪声
量, 我们还向网络指示了我们添加到这个图像的噪声量.
模型必须预测有多少噪声以及如何去除它.
而这个单元是一系列的卷积, 它会减少图像的大小, 如你
所见, 每一步都是如此, 但通过增加特征的数量.
And this unit is a bunch cfc on w of utions that will reduce the size of the image
所以我们减少了大小, 但我们增加了与变分自编码器编码器中
完全相同的内容.
然后我们做相反的步骤, 就像我们对变分解码器所做的那样.
所以现在我们再次将使用一些卷积, 残差块, 注意力等.
个很大的区别是我们需要告诉我们的单元不仅是有噪声的
图像, 不仅是噪声的量, 也就是这个噪声被添加的时间步
image that is a ln eady not so how what is the image with moise
还有提示.
还有提示.
also the pno mpt.
因为如你所记, 我们还需要告诉这个单元我们的提示是什么
因为我们还需要告诉他我们希望输出图像是什么样子.
Be ca
因为有很多方法可以去除初始噪声. 所以如果我们希望初始
噪声变成一只狗, 我们需要告诉他我们想要一只狗
如果我们希望初始噪声变成一只猫, 我们需要告诉他我们想要
只猫
所以单元需要知道提示是什么, 并且他还需要将这个提示与
剩余的信息联系起来, 以及如何以最好的方式结合两种不同的
东西, 例如, 一张图像和一段文本.
我们将使用所谓的交叉注意力.
We will use what is. called the cn oss-attention
交叉注意力基本上允许我们计算两个序列之间的注意力, 其中
查询是第一个序列, 而键和值来自男一个序列.
所以让我们来构建它, 看看它是如何工作的.
So let's go build it and let's see how this works
现在, 我们要做的第一件事是创建一个新类, 一个新文件
Now, the first thing we will do is create a new class, a new. file he re called
这里叫做diffusion 因为这将是我们的扩散模型.
. Now, the first thing we will do is create a new class, a new. file he re called
我想我也会从上到下构建.
And I think also here I will build from top down.
所以我们首先定义扩散类, 然后我们一个接一个地构建每个块.
So. we. first define the diffusion class and then we build each. block one-by one.
让我们从导入常用库开始.
Let'sstart by importing the usual libraries.
所以导入 Torch.
So import Torch.
从 Torch.
From Torch.
自我注意力, 但我们还需要交叉注意力, 稍后我们将构建
Av. the. self. attention but also we will need the cross attention and. later we. will build
它. 然后让我们创建类扩散. 类扩散基本上是我们的单元.
:v :the. self. attention but also we will need the cross attention and. later we. will build.
这是由时间嵌入组成的, 我们稍后会定义它一一时间嵌入
this is made of time embedding so something. that.
这是由时间嵌入组成的, 我们稍后会定义它一一 时间嵌入
这是由时间嵌入组成的, 我们稍后会定义它一一时间嵌入
we :will define it later time embedding 320 which is the size. of. the time embedding. so
所以我们不仅需要给单元噪声图像, 还需要给它添加噪声的
we :will define it later time embedding 320 which is the size of. the time. embedding. so
所以我们不仅需要给单元噪声图像, 还需要给它添加噪声的
时间步. 因此, 单元需要某种方式来理解这个时间步.
at which
这就是为什么这个时间步、作为一个数字, 将通过使用称为
时间嵌入的特定模块转换为嵌入, 销后我们会看到它.
时间嵌入的特定模块转换为嵌入, 稍后我们会看到它.
:c using. this particular module called the time embedding and later we will see. it
然后我们构建单元.
Then we build the unit.
然后是单元的输出层.
And then the output layer of the unit.
稍后我们会看到这个输出层是什么.
And later we will see what is this output layer.
输出层.
Output layer.
稍后我们会看到如何构建它
Later we will see how to build it..
正如你记得的, 单元将接收潜在变量.
As you remember, the unit will receive the latent.
所以这个 Z, 即潜在变量, 是变分自编码器的输出.
所以这个乙, 即潜在变量, 是变分自编码器的输出.
So. this Z, which is a latent, is the output of the variational. auto encoder
所以这个潜在变量, 是一个torch. tensor.
So this latent, which is a torch. tensor.
它将接收上下文.
It will receive the context.
什么是上下文?
What is the context?
它是我们的提示, 同样是一个torch. tensor.
它是我们的提示, 同样是一个torch. tensor.
It is our prompt, which is also a torch. tensor..
并且它将接收这个潜在变量被加噪的时间点.
And it will receive the time at which this latent was. no is ified.
也就是.. 我不记得了.
Which is also... I don't remember.
我想它也是一个张量.
I think it's a tensor also.
让我们尝试定义它.
Let's try to define it.
好的, 是的, 它是张量. 好的, 让我们定义尺寸.
c okay yeah it's tensor okay let'sdefine the sizes so the latent. here. is batch size
这里的潜在变量是批次大小为四, 因为四是编码器的输出
okay yeah it's tensor okay let'sdefine the sizes so the latent. here. is batch. size
四, 闭合:好的, 网格和宽度除以八, 然后我们有
four. because four is the output of the encoder if you remember. correctly-here-four..
四, 闭合:好的, 网格和宽度除以八, 然后我们有
closing okay
四, 闭合:好的, 网格和宽度除以八, 然后我们有
:grid and width divided by eight then we have the context which. is our. prompt
上下文, 即我们的提示, 我们已经在这一步使用clip grid and width divided by eight then we have the context which. is our. prompt
上下文, 即我们的提示, 我们已经在这一步使用clip
编码器进行了转换, 它将是批次大小, 序列长度, 维度
编码器进行了转换, 它将是批次大小, 序列长度, 维度
which we. already converted with the clip encoder here which will be. batch size. by
其中维度是768, 就像我们之前定义的那样, 时间将是另
sequence. length by dimension where the dimension is 768. like we. defined before. and.
其中维度是768, 就像我们之前定义的那样, 时间将是另
the time will
一个, 我们稍后会定义它是如何定义的, 如何构建的, 但
the time will
一个, 我们稍后会定义它是如何定义的, 如何构建的, 但
be another we will define it later how it's defined how. it's built-but it's each..
它是一个嵌入大小的数字, 它是一个大小为320的向量.
ocbe another we will define it later how it's defined how it's built-but it's each.
我们做的第一件事是将这个时间转换为嵌入.
The first thing we do is we convert this time into an embedding
实际上, 我们稍后会看到, 这其实就像transformer 模型的位置
and actually this time we will see later that it's actually just. like. the positional
它实际上是一个乘以正弦和余弦的数字.
encoding of the transformer model it's actually a number. that is. multiplied by. sines.
它实际上是一个乘以正弦和余弦的数字.
and
它实际上是一个乘以正弦和余弦的数字.
cosines just like in the transformer because they. saw. that it works. for the..
就像在transformer 中一样, 因为他们发现这对transformer 有效, 所以
cosines just like in the transformer because they saw. that it works. for the
我们可以使用相同的位置编码来传达时间信息, 这实际上是
transformer. so we can also use the same positional encoding-to convey. the-information.
我们可以使用相同的位置编码来传达时间信息, 这实际上是
of the time which
我们可以使用相同的位置编码来传达时间信息, 这实际上是
v is actually. kind of an information about position so it tells. the. model at which. step..
一种位置信息. 它告诉模型我们在去噪过程中的哪个步骤.
v is actually. kind of an information about position so it tells. the. model at which. step..
一种位置信息. 它告诉模型我们在去噪过程中的哪个步骤.
we arrived in the deno is ification ua
因此, 这将把一个1, 320的张量转换为一个1, 1280. 1000
so. this. one will convert a tensor of1, 320 into a tensor of. 1, 1280, 1000 the-unit will.
的张量. 单元将把我们的潜在变量转换为另一个潜在变量
so. this. one will convert a tensor of1, 320 into a tensor of. 1, 1280, 1000 the. unit will
因此它不会改变大小. 批次为高度.
oc convert our latent into another latent so it will not change. the size batch. for.
因此它不会改变大小. 批次为高度.
height
这是变分自编码器的输出, 首先变为批次320 特征
: This is. the output of the variational auto encoder which. first becomes. batch 320
这是变分自编码器的输出, 首先变为批次320 特征.
features
那么为什么这里我们比开始时有更多的特征?
So why here we have more features than the starting?
因为让我们在这里回顾一下.
Because let'sreview hene.
如你所见, 单元的最后一层, 实际上我们需要回到这里看到
的相同数量的特征.
所以这里我们开始, 实际上这里的维度与我们即将使用的并不
匹配.
所以这是原始单元.
So this is the original unit
但稳定扩散使用的是一个修改过的单元.
所以在最后, 当我们构建解码器时, 解码器不会构建我们
需要的最终特征数量, 即4, 但我们需要一个额外的输出层
回到原始的特征大小, 这就是这个输出层的任务. 稍后我们
回到原始的特征大小, 这就是这个输出层的任务. 稍后我们
but. we. need an additional output layer to go back to the :original size of. features
会在构建这个层时看到, 输出等于self. final.
and. this. is the job of this output layer so later we will see. when we. build. this. this.
会在构建这个层时看到, 输出等于self. final.
layer so output is equal to self. final
这个将从这里的大小回到单元原始的大小, 因为单元的任务是
this one will go from this size here to back to the original. size of. the. unit.
接收潜在变量, 预测有多少噪声, 然后再次接收相同的潜在
this. one will go from this size here to back to the original. size of. the. unit.
接收潜在变量, 预测有多少噪声, 然后再次接收相同的潜在
接收潜在变量, 预测有多少噪声, 然后再次接收相同的潜在
Because. the unit, his job is to take in latent s, predict how much noise. is. ity then
接收潜在变量, 预测有多少噪声, 然后再次接收相同的潜在
变量, 预测有多少噪声, 我们去除它, 我们去除噪声.
然后我们再次提供另一个潜在变量, 我们预测有多少噪声
moise, f hen
我们去除噪声. 我们提供另一个潜在变量, 我们预测噪声
我们去除噪声, 等等, 等等, 等等.
pn edict the noise, we nemo we the noise,
所以输出维度必须与输入维度匹配.
So the output dimension must match the input. dimension.
然后我们返回输出, 即潜在变量.
And then we return the output, which is the. latent.
像这样.
Like this.
首先让我们构建时间嵌入.
Let'sbuild first the time embedding..
我认为它很容易构建.
I think it's easy to build.
所以这是编码我们所在时间步信息的某种东西.
So something that encodes information about the time step. in which. we are.
它由两个线性层组成, 所以这里没有什么花哨的东西
It is made of two linear layers, so nothing fancy. here.
线性一
Linear one.
这将通过嵌入映射到四倍.
Which will map it to four by an embedding.
然后是线性二.
And then linear two.
然后是线性二. 4xn 嵌入到4xn嵌入. 现在你明白为什么
4xnembedding into4xnembeddingand now you understand. why. it. becomes. 1280. which. is
它变成1280, 即4×320. 这个返回张量, 所以输入大小是1×320.
4xnembedding into 4xn embedding and now you understand. why. it. becomes. 1280. which. is
它变成1280, 即4×320. 这个返回张量, 所以输入大小是1x320.
4x320this one returns to the tensor so the input size. is 1x320
我们首先应用第一层, 即线性一, 然后应用 Si L U 函数
What. we do is first we apply the first layer, linear one, then. we apply. the silo
接着再次应用第二层, 即线性层, 最后返回结果.
function, then we apply again the second linear layer, and then we return. it.
这里没什么特别的.
Nothing special here.
最后, 输出维度是1×1, 280.
The last, the output dimension is one by1, 280. 280.
好的, 接下来我们需要构建的是单元. 单元需要很多块
okay. the. next thing we need to build is the unit the unit will require many. blocks. so.
所以让我们先构建单元本身, 然后再构建它所需的每个块.
let's. first build the unit itself and then we build each. of. the. blocks. that. it. will.
如你所见, 单元由一个编码器分支组成, 这就像是变分自
As you can see, the unit is made up of one ence der branch, so this is like the
编码器的编码器. 所以信息会向下流动.
所以图像变得越来越小, 越来越小, 但通道数不断增加
特征也不断增加.
nels keep in cn easing, the feat un
然后我们在这里有一个瓶颈层, 称为瓶颈一一 然后我们有
Then we have this bottleneck layer here
一个解码器部分, 所以图像会恢复到原始大小. 从非常小的
尺寸恢复到原始尺寸.
然后我们在编码器和解码器之间有这些跳跃连接, 所以编码器
每层的输出都连接到男一侧解码器的相同步骤, 你会在这里
每层的输出都连接到另一侧解码器的相同步骤, 你会在这里
and you will see this one here so we start building the left side which. is. the
看到这一点
所以我们开始构建左侧即编码器, 这是一个模块列表
and you will see this one here so we start building the. left side which is the
要构建这些编码器, 我们需要定义一个特殊的层. 基本上
encoders. which is the list of mod utes'and to build these encoders. we need. to. define-a.
要构建这些编码器, 我们需要定义一个特殊的层. 基本上
'special a
要构建这些编码器, 我们需要定义一个特殊的层. 基本上
special layer basically that'wilt appt yok let'sbuild it and then I will describe. it
基本上, 这个切换顺序, 给定一个层列表, 会逐个应用它们.
And. basically this switch sequential, given a list of layers, will. apply. them one. by
基本上, 这个切换顺序, 给定一个层列表, 会逐个应用它们
one.
所以我们可以把它看作是一个顺序的.
So'we can think of it as a sequential.
但它能识别每个层的参数, 并相应地应用
:o But it. can recognize what'are the parameters of each. of. them. and will. apply
但它能识别每个层的参数, 并相应地应用
accordingly.
所以在我定义之后, 它会变得更清晰.
So after t define it, it will be more clear.
所以首先我们像之前一 一样有一 一个卷积, 因为我们想要增加通道数.
so. first we have just like before a convolution because we want to increase the
所以, 如你所见
v number. of channels so as you can see at the beginning we. increase. the. number. of.
所以, 如你所见, 一开始我们增加了图像的通道数
所以, 如你所见 一开始我们增加了图像的通道数
a :image here it's 64butwegodirectlyto320 and then. we. have another one of. these
这里从64直接增加到320. 然后我们有了另一个这样的切换
imagehereit's64but we go directly to 320 and then we have another one. of. these.
这里从64直接增加到320. 然后我们有了另一个这样的切换
switch sequential s which is a unit residual. block
页序 这是一个单元残差块
switch sequential s which is a unit residual block
我们稍后定义它, 但它与我们为变分自编码器构建的残差块
We. define it later, but it's very similar to the residual block. that we built. already
我们稍后定义它, 但它与我们为变分自编码器构建的残差块
for the variational auto encoder.
非常相似
for the variational auto encoder.
然后我们有一个注意力块, 它与我们为变分自编码器构建的
And then. we have an attention block, which is also very similar. to. the. attention.
注意力块非常相似
block that we built for the variational auto encoder.
然后我们有了好的, 我认为最好先构建这个切换顺序
Then we have... Okay, 1think it's better to build this switch sequential, otherwise
否则我们会有太多的.. 是的.
"we have too many.. Yeah."
让我们来构建它.
" Let'sbuild'it.
非常简单
" It's'very'simple.
如你所见: 它是 一个序列, 但给定了x.
as you can see it's a sequence but given x, so which is our latent which is a
所以, 这是我们的潜在变量, 它是一个torch tensor, 我们的
as you can see it's a sequence but given x, so which is our latent which is a
上下文 也就是我们的提示, 以及时间, 这也是一个
torch. tensor our context, so our prompt and the time, which is. also a tensor wel
张量, 我们会一个接一个地应用它们, 但根据它们是什么.
torch. tensor our context, so our prompt and the time, which is. also a tensor well
张量, 我们会一个接一个地应用它们, 但根据它们是什么.
ap pty them one by one
例如, 如果层是一个单元注意力块, 它就会这样应用
例如, 如果层是一个单元注意力块, 它就会这样应用
:but. based on what they are so if the layer is a unit attention block for example. it
所以, x 和上下文的层. 为什么?
:will apply it like this'so'layer of x and context why because this attention block
因为这个注意力块基本上会计算我们的潜在变量和提示之间的
will apply it like this'so'layer of x and context why because. this attention block
因为这个注意力块基本上会计算我们的潜在变量和提示之间的
basically
这就是为什么这个残差块会计算, 会将我们的潜在变量与其
will. compute the cross attention between our latency and. the-prompt. this. is. why.
这就是为什么这个残差块会计算, 会将我们的潜在变量与其
这就是为什么这个残差块会计算, 会将我们的潜在变量与其
this. residual block will'compute will match our latent with its time step and. then. if.
时间步匹配, 然后, 如果是任何其他层, 我们就直接应用
this. residual block will'compute will match our latent with its time step and. then. if.
它, 然后返回, 但在一段时间之后.
:it's any other layer we just apply it and then we return. but after. the for. awhile
是的, 月 所以现在我们理解了这一点. 我们只需要定义这个
yeah so
是的, 所以现在我们理解了这一点. 我们只需要定义这个
Av. this. is now we understood this we just need to define this. residual block. and. this
残差块和这个注意力块.
this. is now we understood this we just need to define this residual block. and. this.
残差块和这个注意力块.
"attention block
然后我们有了另一个序列, 嗯, 这里的顺序切换.
then. we. have another sequence um sequential switch this. one. here so the. code. im
所以我写的代码实际上是基于一个仓库, 实际上我写的很多
writing actually is based on'a repository upon which actually-most of. the code. i
代码都是基于这个仓库, 而这个仓库又是基于另一个仓库
wrote is based on
代码都是基于这个仓库, 而这个仓库又是基于另一个仓库
o which is in turn based on another repository which was originally written for.
如果我没记错的话那个仓库最初是为 Tensor Flow写的.
ewhich is in turn based on another repository which was originally written for.
所以实际上, 稳定扩散的代码, 因为它是由 LMU 大学的
tensor flow if i remember correctly so actually the the code for stable. diffusion
所以实际上, 稳定扩散的代码, 因为它是由 LMU 大学的
所以实际上, 稳定扩散的代码, 因为它是由 LMU 大学的
because. it's a model that is buit by comp viz group at the. LMu. university. of course.
comp viz 小组构建的模型, 当然, 它不可能与那个代码不同.
because. it's a model that is buit by comp viz group at the. LMu. university. of course.
所以大部分代码实际上是彼此相似的.
it. can not be different from that code'so most of the code are. actually. similar. to
所以大部分代码实际上是彼此相似的.
each'other i
我的意思是, 你不能创建相同的模型并改变代码. 当然
mean-you. can not create the'same model and change the code. of. course. the. code. will. be.
代码会是相似的所以我们再次使用这个切换顺序. 所以
similar. so we again use'this one switch sequential so here-we. are. building the
这里我们正在构建编码器端, 所以我们正在缩小图像的尺寸.
encoder side so we'are reducing the size of. the image
所以我们有从320到64的残差块, 然后我们有一个从
let me. check where we are so we have the residual block of. 320 to. 64and. then. we. have
所以我们有从320到64的残差块, 然后我们有一个从
anattention block of 8to80
8到80的注意力块, 这个注意力块需要头数. 这个
anattention block of 8to80
8到80的注意力块, 这个注意力块需要头数. 这个
And. this. attention block takes the number of head, this 8. indicates. the. number. of
8 表示头数, 而这个表示嵌入大小.
head and this indicates the embedding size..
我们稍后会看到如何将这个的输出转换成一个序列,! 以便我们
We. will. see later how we transform this, the output of this into. a sequence so. that.
可以对其运行注意力机制
We. will. see later how we transform this, the output of. this. into a sequence so. that.
可以对其运行注意力机制.
we can run attention on it.
好的, 我们有这个顺序, 然后我们还有另一个
Okay, we have this sequential and then we have another one.
然后我们还有另一个卷积.
Then we have another convolution.
让我复制一
Let me just copy.
Convolution of size from 640to640 channels.
核大小为3, 步幅为2, 填充为1.
Kernel size three, stride two, padding one.
然后我们还有一个残差块, 它将再次增加特征.
Then we have another residual block that will again increase the features.
所以从640到1. 280
Sofrom640to1, 280.
然后我们有一个8个头的注意力块160是嵌入大小.
and-then we. have an attention block of 8heads and160is the embedding size. then. we
然后我们还有一个1到80的残差块, 1到80和
have another residual block of 1to80 and1to80 and8 and 160so as you can see
所以正如你所见, 就像在变分自编码器的编码器中一样
have another residual block of 1to80 and 1 to80 and8and 160 so as you. can see.
所以正如你所见, 就像在变分自编码器的编码器中一样
just like in the encoder of the variational. auto encoder
所以正如你所见, 就像在变分自编码器的编码器中一样
通过这些卷积, 我们不断减小图像的尺寸.
通过这些卷积, 我们不断减小图像的尺寸.
With these convolutions we keep decreasing the size of. the image.
所以实际上, 这里我们从潜在表示开始, 它是高度除以8
Soactually-here we started with the latent representation. which was height divided
所以实际上, 这里我们从潜在表示开始, 它是高度除以8
by8andheightdividedby8.
So let me write some shapes here
至少你需要理解尺寸的变化
At least you need to understand the size. changes.
所以批次大小为高度除以8和宽度除以8.
Sobatchsize for height divided by 8andwidth. divided. by. 8.
当我们应用这个卷积时它会变成除以16 所以它会变成
when. we apply this convolution it will become divided by 16so it will become divided
个非常小的图像. 在我们应用第二个卷积后, 它会变成
by16so. it. will become a very small image and after we apply the second. one it. will
一个非常小的图像. 在我们应用第二个卷积后, 它会变成
become divided by 32
除以
32
becomedividedby32
所以这里我们从16开始. 这里它会变成除以32. 那么这
so. here. we start from 16hereit will become divided by. 32. so. what does. it. mean.
意味着什么? 除以32? 这意味着如果初始图像大小是512
so. here. we start from16here it will become divided by. 32. so. what does. it mean
那么潜在表示的大小是64乘64.
:divide. by. 32that if the initial image was of size512 the. latent. is of size. 64. by. 64.
then it
然后它变成32乘32. 现在它变成了16乘16.
:cbecomes32by32now it has become 16by16 and. then we apply this residual
然后我们应用这些残差连接, 然后我们应用另一 一个卷积层
becomes 32by32now it has become 16by16 and then we apply this residual
然后我们应用这些残差连接, 然后我们应用另一个卷积层,
connections
它将进一步减小图像的大小从32. 这里除以32和除以
32到除以64.
further. from 32heredivide by32and divide by32todivide by. 64. every-time we
每次我们将图像的大小除以2, 特征数量是12801280. 然后我们有
further. from 32here divideby32and divide by32todivide by. 64. every-time we.
每次我们将图像的大小除以2, 特征数量是12801280. 然后我们有
divided'the we'divided the size of the image by. two.
每次我们将图像的大小除以2, 特征数量是12801280. 然后我们有
每次我们将图像的大小除以2, 特征数量是12801280. 然后我们有
and. the. number of features is12801280 then we have. a. unit this is our block so. let
个单元 一这是我们的块, 所以让我也复制这个1280 and. the. number of features is12801280 then we have. a. unit. this is our block so. let.
和1280. 然后我们有一个最后一个, 它的大小相同.
me. copy also this one of 1280and1280 and then we have a. last. one-which. is another.
和1280. 然后我们有一个最后一个, 它的大小相同.
one of the same size
所以现在我们有一个图像它是64除以64和除以64
Sonow we. have an image that is 64, divided by64 and divided. by. 64, but. with. much
但有更多的通道
more channels.
我忘了在这里更改通道数.
I forgot to change the channel numbers. here..
所以这里是1280个通道, 除以64, 除以64.
Sohere is1, 280channels and divided by 64, divided by64.
而这个保持不变.
And this one remains the same.
残差连接不会改变大小.
The residual connections don't change the size.
这里应该是1280到1, 280.
Here should be 1, 280 to1, 280...
这里应该是640到640这里应该是320到320. 所以
here should be 640to640 and here it should be320to320. so. aslsaid before we
正如我之前所说, 我们不断减小图像的大小, 但我们不断
here should be 640to640 and here it should be320. to. 320. s0. asl. said before we
增加每个像素的特征数量
keep. reducing the size of the image but we keep increasing this. number of. features. of..
增加每个像素的特征数量
each pixel
基本上, 然后我们构建瓶颈, 也就是这个单元中的这部分.
basically then webui td'the bottleneck which is this. part here of. the unit.
这是一个残差块序列. 然后我们有一个注意力块, 它会进行
this is. a sequence of residual block then we have the attention block which. will. make
交叉注意力 然后我们还有另一个残差块.
cross attention and then we have another residual block
然后我们有解码器. 所以在解码器中, 我们将做与编码器
then. we have the decoder so in the decoder we will do the. opposite. of what we. did in
然后我们有解码器. 所以在解码器中, 我们将做与编码器
相反的操作, 我们将减少特征数量, 但再次增加图像大小.
then we hame the decoder so in the decoder we will do theo
相反的操作, 我们将减少特征数量, 但再次增加图像大小.
the. encoder so we will reduce the number of features but. increase the image size.
所以我们有2. 560到1, 280.
Sowe have 2, 560 to1, 280.
为什么这里是2. 560即使在瓶颈之后我们有1, 280?
Why hereis2, 560even if after the bottleneck we have 1, 280?
所以我们在这里讨论这部分.
So we are talking about this
所以我们在这里讨论这部分.
所以在解码器的输入之后, 这个单元这一侧的输出是瓶颈的
So after the input of the decoder, so this side hene of the unit, is the output of
输出.
the bottleneck
但瓶颈输出1280个特征而编码器期望2560个, 是双倍
But. the bottleneck is outputting 1280 features while the encoder. is expecting 2560,
但瓶颈输出
1280个特征, 而编码器期望2560个, 是双倍
sodouble the amount.
的数量
o doub
因为我们必须考虑到这里有一个跳跃连接.
所以这个跳跃连接会在每一层这里使数量翻倍.
So. this skip connection will double the amount at. each layer here.
这就是为什么我们在这里期望的输入是前一层输出大小的两倍.
and. this. is. why the input we expect here is double the size. of what is the. output. of.
这就是为什么我们在这里期望的输入是前一层输出大小的两倍.
"the previous layer.
让我在这里也写一些形状所以批量大小:2560, 图像非常小.
Let me write some shapes also here, sobatchsize, 2560, the image is very small, so
所以高度乘以宽度除以64, 它将变成.
height by and width divided by 64, anditwiltbecome
然后我们应用另一个相同大小的switch sequential. 然后我们再应用一个
n then. we r apply another switch sequential of the same size. then we apply. another one
带有上采样的, 就像我们在变分自编码器中所做的那样.
then we apply another switch sequentia t of the same size. then we apply. another one
所以如果你记得, 在变分自编码器中, 为了增加图像的
a with. an. up sample just like we did in the variational auto encoder. so if you remember.
所以如果你记得, 在变分自编码器中, 为了增加图像的
in'the
所以如果你记得, 在变分自编码器中, 为了增加图像的
A variational. auto encoder to increase the size of the image. we. do. up sampling and. this
大小, 我们进行上采样, 这正是我们在这里所做的. 我们
n variational. auto encoder to increase the size of the image we. do. up sampling and. this..
进行上采样, 但这不是那种上采样.
is what we. do exactly here we do up sample but this is not the up sample. that we. did.
我们完全一样地做了, 但概念是相似的, 我们稍后也会定义
is what we. do exactly here we do up sample but this is not. the up sample. that. we did
我们完全一样地做了, 但概念是相似的, 我们稍后也会定义
exactly the same but'the concept is similar..
所以我们有另一个带有注意力的残差, 所以我们有2000 个
and. we will define it later also this one so we have another. residual. with attention.
残差, 然后我们有一个8乘160的注意力块. 然后
sowe have a residual of 2oo0 and then we have. an-attention block
残差, 然后我们有一个8乘160的注意力块. 然后
我们再次有这个. 然后我们还有一个带有上采样的, 所以
8by. 160 then we have again this one then we have another. one with upsampling-so we
我们有90, 20然后我们有一个上采样.
have9020 and then we have an up sample
这个很小. 所以我知道我没有写所有的形状, 否则这真的是
this one. is small so I know that I'm not writing all the. shapes. but otherwise it's
所以只要记住我们不断增加图像的大小但我们减少特征的
really. tiring job and very long so just remember that we are. keep increasing. the. size
所以只要记住我们不断增加图像的大小但我们减少特征的
of the
所以只要记住我们不断增加图像的大小但我们减少特征的
image but we we will decrease the number of features later we will see. that this.
数量. 稍后我们会看到, 这里的这个数字会变得非常小,
image but we we will decrease the number of features later we will see. that this.
而图像的大小将接近正常
number here will become very small and the size of the image will become. nearly-to
而图像的大小将接近正常.
the normal then we have another one
然后我们还有一个带有注意力的, 所以, 正如你所见
the normal then we have another one
然后我们还有一个带有注意力的, 所以, 正如你所见
with. attention so as you'can see we are decreasing the. features. here. then we have. 8.
我们在这里减少特征. 然后我们有8乘80, 我们也在
by 80. and we are increasing also here the size and. then. we have another. one
增加这里的大小
:oby 80. and we are increasing also here e the size and. then-we have another one
然后我们还有另一个880.
然后我们还有一个带有上采样的, 所以我们增加图像的大小
and s8o. then. we have another one with upsampling so we increase the size. of. the. image.
上采样有640个特征. 然后我们还有一个带有注意力的残差块.
so. 960. 640. 8. heads with the dimensions bedding size of 80 and. the. upsampling with. 640
上采样有640 个特征. 然后我们还有一个带有注意力的残差块
上采样有640 个特征. 然后我们还有一个带有注意力的残差块.
features and then we have another residual block with. attention.
然后我们还有一 个640, 320. 840最后-个是640乘320
then. we have another one which is a 640320840 and finally. the. last one. we have. 640.
然后我们还有 个640, 320. 840, 最后一个是640乘320
and
和8乘40. 这里的维度与将由单元输出的维度相同
和8乘40. 这里的维度与将由单元输出的维度相同
by. 320and 8and40 this dimension here is the same that will. be. applied. by. the. is
这个在这里, 然后我们会把它交给最后一层来构建原始的潜在
the. output of the unit as you can see here this one here and. then we will-give it. to
这个在这里, 然后我们会把它交给最后一层来构建原始的潜在
the final
这个在这里, 然后我们会把它交给最后一层来构建原始的潜在
layer. to. build the original latent size okay let'sbuild all these. blocks that we
大小. 好的, 让我们构建所有这些我们之前没有构建的块
layer. to. build the original latent size okay let'sbuild all these. blocks that we
大小. 好的, 让我们构建所有这些我们之前没有构建的块.
didn't build before so first let'sbuild the. up sample
所以首先让我们构建上采样, 让我们在这里构建它, 它和
didn't build before so first let'sbuild the. up sample
所以首先让我们构建上采样, 让我们在这里构建它, 它和
两个 完全一样.
Let'sbuild it here, which is'exactly'the same as the two, okay?
不改变特征的数量, 这实际上也不改变图像的大小.
without. changing the number of features and this is also. doesn't. change. the size. of
所以我们从批量通道或特征 我们称之为特征, 高度
the rimage actually so we will go from batch channels. or. features let's. call it..
所以我们从批量通道或特征 我们称之为特征, 高度
features height width to
宽度 到批量大小. 特征:高度乘以2, 宽度乘以
features height width to
宽度一 到批量大小. 特征:高度乘以2, 宽度乘以
batch. size features height multiplied by 2andwidth. multiplied by2why?.
• la per (x )
因为我们即将使用上采样, 我们现在要做的这种插值. 插值
because. we are going to use the upsampling this interpolation. that we will do. now.
×比例因子等于模式等于最近, 与我们在这里做的操作
interpolate x scale factor equal to mode is equal to nearest is. the same operation.
相同. 这里的操作相同.
that we did here the same operation. here.
它基本上会使大小翻倍, 然后我们应用一个卷积.
it will double the size basically and then we apply a convolution now we. have. to
现在我们必须定义最后的块, 我们还必须定义输出层, 以及
define the. final block and we also have to define the output layer and we also. have
注意力块和残差块. 所以让我们先构建这个输出层. 它更
define the. final block and we also have to define the output. layer and we also have
注意力块和残差块. 所以让我们先构建这个输出层. 它更
to define the
注意力块和残差块. 所以让我们先构建这个输出层. 它更
attention block and the residual block so let'sbuild first this. output layer it's
容易构建.
easier to build
这个也有一个组归一化, 同样组大小为32.
This one also has a group normalization, again with the size. of the group 32.
It also has a convolution. 它也有一个卷积.
和一 个1的填充.
Andapaddingof1.
Ok okay
In g(tiae )
最后 层需要将这个形状转换成这个形状
The final layer needs to convert this shape into. this. shape.
所以320个特征转换成4个.
So320featuresinto4.
我们有一个输入, 批量大小为320 个特征.
We have an input which is batch size of 320. features.
高度除以8, 宽度除以8.
The heightisdividedby 8and the width is divided. by. 8.
我们首先应用一个组归一化.
We first apply a group normalization.
然后我们应用 C-loop.
Then we apply the C-loop.
然后我们应用卷积, 然后返回. 这基本上就是卷积. 让我
then. we-apply the convolution and then we return this will basically the. convolution
也写下为什么我们要减少大小.
let. me-write also why we are reducing the size this convolution will. change the
这个卷积会改变通道的数量, 从输入到输出, 当我们声明它
let. me. write also why we are reducing the size this convolution will. change. the
这个卷积会改变通道的数量, 从输入到输出, 当我们声明它
number of
这个卷积会改变通道的数量, 从输入到输出, 当我们声明它
channels. from in to out and when we will declare it we say-that. we-want. to convert
时: 我们说我们想在这里从320转换到4.
channels. from in to out and when we will declare it we say-that. we-want. to convert
所以这个将是形状批量大小:4, 高度除以4, 高度除以4
from320to4here sothis one will be of shape batch size 4. height. divided. by. 4
所以这个将是形状批量大小:4, 高度除以4, 高度除以4
所以这个将是形状批量大小:4, 高度除以4, 高度除以4
high. divided by4 and width divided by 8 then we need to go build. this. residual. block.
宽度除以8. 然后我们需要去构建这个残差块和这里的
high. divided by4 and width divided by 8 then we need to go build. this. residual. block
让我们从残差块开始, 它与我们为变分自编码器构建的残差块
can d this. attention block here so let'sbuild it here let'sstart. with. the. residual
让我们从残差块开始, 它与我们为变分自编码器构建的残差块
block
让我们从残差块开始,
, 它与我们为变分自编码器构建的残差块
which. is very similar'to the residual block that we built for. the variational
非常相似, 所以是单元块.
auto encoder so unit block
这是时间步的嵌入. 如你所记, 通过时间嵌入我们转换成
this is. the. embedding of the time step as you remember with. the. time embedding-we
大小为1280 的嵌入我们有这个组归一化. 它总是这个组
transform. into an embedding of size1280 we have this group normalization. it's always
大小为1280 的嵌入手 我们有这个组归一化. 它总是这个组
'this group norm
然后我们有一个卷积, 还有一个用于时间嵌入的线性层.
then we have a convolution and we have a linear for. the time. embedding
然后我们还有另一个组归一化.
Then we have another group normalization.
我们稍后会看到这是如何合并的
We wit l see tater what is this merged
还有另 个卷积.
And another convolution.
核大小3, 填充为1.
Kine L size3, that being 1.
同样, 就像之前 样如果输入通道数等于输出通道数
Again, just like before, we have if the in-channels is equal. to. the. out-channels, we.
我们可以直接通过残差连接将它们连接起来.
Again, just like before, we have if the in-channels is equal. to the. out-channels, we
否则, 我们创建一个卷积来连接它们, 将输入的大小转换为
Otherwise, we create a convo tution'to connect them, to convert. the size. of. the. input
否则 我们不能将这两个张量相加
Otherwise, we can not add the two tensors..
所以它以这个特征张量为输入, 实际上是潜在的批量大小和
so. it takes as input this feature tensor which is'actually. the. latent
所以它以这个特征张量为输入, 实际上是潜在的批量大小和
所以它以这个特征张量为输入, 实际上是潜在的批量大小和
batch size in. channels then'we have height and width and then. also. the time-embedding
通道数. 然后我们有高度和宽度, 还有时间嵌入, 它是
batch size in. channels then'we have height and width and then. also. the time-embedding
1乘1280, 就像这里一样, 我们首先构建一个残差连接.
which. is. 1by1280just like here and we build first of all a residual connection
I
然后我们应用组归一化.
Then we do apply the group normalization.
所以通常残差连接, 残差块或多或少总是相同的.
So usually. the residual connection, the residual blocks are more. or less. always. the.
所以通常残差连接, 残差块或多或少总是相同的
same.
所以有一个归一化和激活函数
So there is a normalization and activation. function.
然后我们可以有一些跳跃连接等等.
Then we can have some skip connection, etc., etc.
然后我们有时间
Then we have the time.
这里我们将潜在特征与时间嵌入合并, 但时间嵌入没有批次和
Here. we. are merging the latency with the time embedding, but. the time embedding
通道维度, 所以我们在这里用*un squeeze 添加它.
doesnt have. the batch and the channels dimension, so. we add it. here with. un squeeze
我们将它们合并.
And we merge them.
然后我们归一化这个合并的连接.
Then we normalize this merged connection.
这就是为什么它被称为合并.
This is why it's called merged..
我们应用激活函数.
We apply the activation function.
我们应用这个卷积:最后, 我们应用残差连接. 那么
we apply. this convolution and finally we apply the residual connection so. why are we
嗯, 想法是这里我们有三个输入:我们有时间嵌入, 我们有
doing-this. well the idea is that here we have three input we. have. the time. embedding
嗯, 想法是这里我们有三个输入:我们有时间嵌入, 我们有
潜在特征, 我们有提示.
we ha we
我们需要找到一种方法将这三部分信息结合起来, 以便单元
需要学会使用特定提示作为条件, 在特定时间步检测噪声图像
中的噪声, 这意味着模型需要识别这个时间嵌入, 并且需要
将这个时间嵌入与潜在特征关联起来.
no. del needs to he cognize this this time embedding
而这正是我们在残差块中所做的.
而这正是我们在残差块中所做的
this. time. embedding with the latency and this is exactly what. we are. doing in. this.
而这正是我们在残差块中所做的.
residual
而这正是我们在残差块中所做的.
block here we are relating the latent with the time embedding
这里我们将潜在特征与时间嵌入关联起来, 使得输出依赖于
block here we are relating the latent with the time embedding
这里我们将潜在特征与时间嵌入关联起来, 使得输出依赖于
这里我们将潜在特征与时间嵌入关联起来, 使得输出依赖于
so that the. output will depend on the combination of both. not on. the single noise. or
两者的组合, 而不仅仅是单一的噪声或单一 的时间步
so that the. output will depend on the combination of both. not on. the single noise. or
这也将通过上下文来完成, 使用我们即将构建的注意力块中的
in. the. single time step and this will also be done with the. context using cross
这也将通过上下文来完成, 使用我们即将构建的注意力块中的
attention in
所以单元注意力块168, 好的.
the attention block'that we will build now so unit attention block
所以单元注意力块168, 好的
好的, 我将定义一些目前可能不太有意义的层, 但稍后当
Okay, I will. define some layers'that for now will not make much. sense, but later they.
我们实现前向方法时, 它们就会有意义了.
will make sense when we make the forward method.
_ 所以我的猫在要食物. 我想他已经有了食物, 但也许他今天
so. my cat. is asking for food i think he already have food but. maybe. he wants to. eat
所以让我先完成这个注意力块和单元, 然后我再处理他的
something special today so let'me finish this attention block and the unit. and. then.
所以让我先完成这个注意力块和单元, 然后我再处理他的
i'm all his
事情. 为什么每个人都想要注意力、自 自注意力和头通道呢?
事情. 为什么每个人都想要注意力、自注意力和头通道呢?
why everyone wants attention self-attention and head channels here we don't have. any
这里我们没有任何偏置
这里我们没有任何偏置 就像在原始的-Transformer -中一样. 所以
"we don't
这里我们没有任何偏置就像在原始的-Transformer 中一样. 所以
have any bias just like in the van ilta transformer so we. have. this attention then. we.
然后我们有一 个层归
have at ayer'normalization self. layer norm2.
然后我们又有另一个注意力. 稍后我们会看到为什么我们需要
然后我们又有另一个注意力. 稍后我们会看到为什么我们需要
which. is along the same number of features then we have another. attention-we will. see.
但这不是自注意力, 而是交叉注意力, 稍后我们会看到它是
date r. why we need all this attention but this is not a self attention it's. a cross
但这不是自注意力, 而是交叉注意力, 稍后我们会看到它是
attention
但这不是自注意力, 而是交叉注意力, 稍后我们会看到它是
and we will see later how it works then we have the. layer. norm tree
如何工作的. 然后我们有层归一化树.
and we will see later how it works then we have the. layer. norm tree
这是因为我们使用了一 个叫做 J GLU 激活函数的函数
A This. is because we are using a function that is called the J GLU. activation. function.
[cassa is, cham eis, kersel_sisec3, sedting-5]
所以我们需要这些矩阵
So we need these matrices here.
[cassa is, cham eis, kersel_sisec), sedting-5]
好的, 现在我们可以构建前向方法了所以我们的×是
okay. now. we can build the forward method so our x is our latency. so we have a batch.
潜在特征, 因此我们有一个批次大小我们有特征数量
okay. now. we can build the forward method so our x is our. latency. so we have a batch
然后我们有我们的上下文, 即我们的提示, 它是一个批次
size we have a features we have height we have width then we. have. our context. which.
然后我们有我们的上下文, 即我们的提示, 它是一个批次
is oui
然后我们有我们的上下文, 即我们的提示, 它是一个批次
prompt which is a batch size sequence length dimension the dimension is. size 768. as.
大小, 序列长度一维度. 维度是768, 正如我们之前看到的.
prompt which is a batch size sequence length dimension the. dimension is. size 768. as
大小, 序列长度, 维度. 维度是768, 正如我们之前看到的.
we sawbefore
所以我们要做的第一件事是进行归一化.
ocso the. first thing we will do is we will do the normalization so. just. like. in. the..
所以, 就像在 Transformer 中一样, 我们会取输入, 即我们的
so the. first thing we will do is we will do the normalization so. just. like. in. the.
潜在特征, 并应用归一化和卷积. 实际上, 在 Transformer 中
transformer we will take the input so our latency and we apply the normalization and
潜在特征, 并应用归一化和卷积. 实际上, 在 Transformer 中
the
潜在特征, 并应用归一化和卷积. 实际上, 在 Transformer 中
oc convolution actually in the transformer there is no convolution. but only. the.
没有卷积, 只有归一化. 所以这被称为长残差, 因为它会
convolution actually in th'etransformer there is no convolution. but only. the.
在最后应用
normalization so this is called'the long residual because. it will. be applied at. the
在最后应用
end
好的, 月 所以我们正在应用归一化, 这不会改变张量的大小.
Okay, so. we are applying the norm at ization, which doesnt. change the. size of. the
好的, 月 所以我们正在应用归一化, 这不会改变张量的大小.
tensor.
然后我们有卷积.
" Then we have convolution.
x 等于self. comb 输入的x这也不会改变张量的大小.
xis. equal. to self. comb input'of x, which also doesn't change. the size of. the tensor.
然后我们取形状, 即批次大小、特征数量、高度和宽度.
Then. we. take the shape, which is the batch size, the number of. features, the. height.
然后我们取形状, 即批次大小、特征数量、高度和宽度.
"and the width.
我们进行转置, 因为我们想要应用交叉注意力. 首先我们
A we transpose because we want'to apply cross attention first we-apply. self-attention
所以我们进行归一化加上自注意力, 目 自注意力加上跳跃连接
then. we apply cross attention'so so we do normalization plus self-attention..
所以我们进行归一化加上自注意力, 自注意力加上跳跃连接.
"self-attention with
所以我们进行归一化加上自注意力, 目 自注意力加上跳跃连接
skip. connection sox is x dot'transpose of minus one minus-two. so we are going. from
所以我们从这里 等等, 我忘了一些东西
所以我们从这里一一响, 等等, 我忘了一些东西
v. this. uh. wait i forgot something here first of all we need to. do x. is equal to xd otu
首先我们需要做的是:x等于x, 点, u和c, h
:and ch multiplied by w so we are going from this to bet. size features and then. h.
乘以 W. 所以我们从这里到批次大小特征, 然后h
multiplied by wso this one multiplied. by. this..
然后我们转置这两个维度. 所以现在我们从这里到这里
then we. transpose these two dimensions so now we get. from. here to here so. the
现在我们应用这个归一 化加上自注意力, 所以我们有一个首先
features. become the last one now we apply this normalization plus self-attention so
现在我们应用这个归一化加上自注意力, 所以我们有一个首先
and so we have a
的短残差连接, 它会在注意力之后立即应用
的短残差连接它会在注意力之后立即应用.
A :first short residual connection that will apply right after the attention so we say
所以我们说:x等于层归一化一, 即x. 然后我们应用
first short residual connection that will apply right after. the attention so we say
所以我们说等于层归一化一 即x. 然后我们应用
that x is equal to
然后我们应用残差连接, 所以加上等于残差短, 短
layer. norm. one sox then we apply the attention so self dot attention one and. then we
第一个残差连接. 然后我们说残差短再次等于六, 因为
apply. the residual connection sox is plus equal to residual short short. the first
第一个残差连接. 然后我们说残差短再次等于六, 因为
residual
第一个残差连接. 然后我们说残差短再次等于六, 因为
A connection then we say that'the residual short is again equal to. six because we-are
所以现在我们应用归一化加上带有跳跃连接的交叉注意力.
doing-to apply now the cross attention so now we apply. the. normalization
所以现在我们应用归一化加上带有跳跃连接的交叉注意力.
所以现在我们应用归一化加上带有跳跃连接的交叉注意力.
plus. the. cross attention'with'skip connection so what we did here. is what. we do. in.
所以我们在这里做的是我们在任何 Transformer 中都会做的事情.
plus. the. cross. attention'with'skip connection so what we did here. is what. we do. in.
所以让我在这里展示我们在任何 Transformer 中做的事情.
any. transformer so let me show you here what we do in any. transformer. so. we apply
所以让我在这里展示我们在任何 Transformer 中做的事情.
所以让我在这里展示我们在任何 Transformer 中做的事情.
any. transformer so let me show you here what we do in. any-transformer. so. we. apply
我在这里展示我
size of the 4 所以我们应用一些归一化, 我们计算注意力, 然后我们将其
size of the emoe
smber of headt 与跳跃连接结合. 在这里, 我们现在将, 葡不是计算自
head)w
与跳跃连接结合. 在这里, 我们现在将, 而不是计算自
connection here and now we will instead of calculating a self-attention we will do. a
size of the eml 与跳跃连接结合. 在这里, 我们现在将, 葡不是计算自
注意力 我们将进行交叉注意力, 我们还没有定义它
connection here and now we will instead of calculating a self-attention we will. do. a
所以, 呢, 短. 然后首先我们计算, 我们应用归一化
cross attention which we stil i didn't define we will define it later. so
然后是潜在特征和提示之间的交叉注意力.
:uh. short and then first we calculate we apply the normalization. then. the cross
这是交叉注意力, 所以这是交叉注意力, 我们会看到如何
v attention. between the latent s and the prompt this is the cross. attention so. this. is.
这是交叉注意力, 所以这是交叉注意力, 我们会看到如何
cross attention
这是交叉注意力, 月 所以这是交叉注意力, 我们会看到如何
and we will see how and x plus or equal to residual. residual short.
然后加上或等于残差, 残差短, 好的, 然后我们
and we will see how and x plus or equal to residual. residual short
然后加上或等于残差, 残差短, 好的, 然后我们
再次做 X. 最后, 就像在注意力 Transformer 中一样, 我们有
再次做 X. 最后, 就像在注意力 Transformer 中一样, 我们有
Okay, and then again, we'll do X. Finally, just like with the attention transformer,
= number of heads 个带有 J GLU 激活函数的前
at (bead,
head )w
个带有 J GLU 激活函数的前馈层
we have a feed forwardlayer with the J GLU activation. functio. n.
而且这实际上,-如果你看过稳定扩散的原始实现, 它就是像
And. this is actually, if you watch the original implementation of the stable
而且这实际上, 一如果你看过稳定扩散的原始实现, 它就是像
diffusion, it's implemented exactly like. this...
diffusion, it's implemented exactly like this. 这样实现的.
所以基本上后来我们进行逐元素乘法
So it's basically later we do element-wise multiplication.
所以这些都是涉及很多参数的特殊激活函数.
So. these are special activation functions that involve a. lot. of parameters.
但为什么我们使用这个而不是另一个. 我告诉过你, 就像
but. why we use one and not the other I told you just like. before. they just. saw. that.
以前一样. 他们只是发现这个在这个应用中效果更好. 没有
but. why we use one and not the other I told you just like. before. they just. saw. that.
然后我们应用跳跃连接, 所以我们应用交叉注意力. 然后
:this. one works better for this kind of application there is no other. then. we apply
然后我们应用跳跃连接, 所以我们应用交叉注意力. 然后
the skip
然后我们应用跳跃连接, 所以我们应用交叉注意力. 然后
connection so we apply the cross attention then we define another one. here so. this
所以这个基本上是归一化加上带有j glue 的前馈层和跳跃连接
connection so we apply the cross attention then we define another one. here so. this.
所以这个基本上是归一化加上带有j glue 的前馈层和跳跃连接
one is basically normalization.
所以这个基本上是归一化加上带有j glue 的前馈层和跳跃连接
-plus flip forward layer with j glue and script connection. in which the. script
其中跳跃连接在这里定义. 所以在最后我们总是应用跳跃
plus flip forward layer with j glue and script connection. in which the script
连接.
A ::connection is defined here so at the end we always apply. the. script. connection
最后, 我们将张量变回不再是像素序列. 所以我们反转之前
finally we change back
最后, 我们将张量变回不再是像素序列. 所以我们反转之前
to. our. tensor to not be a sequence of pixels anymore so-we. reverse the previous
的转置, 转置到
transposition transpose to
x + resi ba_shert
所以基本上我们从批次大小, 宽度, 宽度乘以高度, 高度
so basically. we go from batch size width width multiplied. by. height. height. multiplied.
宽度.
A:oby width and features into batch size features height multiplied by width...
然后我们移除这个乘法, 所以我们反转这个乘法和chw.
then we. remove this multiplication so we reverse this multiplication and. chw. finally.
最后我们将应用我们在开头定义的长跳跃连接.
A we will apply the long skip connection that we defined here at the beginning so. only.
所以只有在尺寸匹配的情况下.
if the
如果尺寸不匹配, 我们应用这里的这个, 我们这里有返回
size. match if the sizes don't match we apply the here this. one we have here return
如果尺寸不匹配我们应用这里的这个, 我们这里有返回
self. com output
self com 输出, 这就是我们整个单元.
我们已经定义了所有东西, 我想, 除了交叉注意力, 它
We. have. defined everything, I think, except for the cross. attention, which is very
(1. 12) 非常快.
fast
所以我们转到我们之前定义的注意力, 并把它放在错误的
So we go. to the attention that we defined before and we put it in. the wrong folder.
它应该是.
It should be
跳跃变化.
Skip changes
让我检查一下我是否放对了.
tetmecheckif1 put it correctly...
是的, 我们只需要在这里定义这个交叉注意力.
Yeah, we only need to define this cross attention. here.
好的, 注意力. 所以让我们去定义这个交叉注意力
:okay attention'so let'sgo and let'sdefine this cross attention. so. class it. will. be
所以类, 它将非常类似于一一不, 实际上, 与自
okay attention so let'sgo and let's define this cross attention. so. class it. will. be
注意力相同, 除了键来自一侧, 查询一 一抱歉, 查询
very. similar to the not very similar actually same as the self attention except. that
注意力相同, 除了键来自一侧, 查询一 一抱歉, 查询
the keys
注意力相同, 除了键来自一侧, 查询一 抱歉, 查询
come. from. one side and the query and sorry the query come from. one side and. the. key
来自一侧, 而键和值来自另一侧.
and the values from another side
这是键和值嵌入的维度.
This is the dimension of the embedding of the keys. and. the values.
这是一个查询.
This is one of the queries
这是 WQ 矩阵.
This is the w Q matrix.
在这种情况下, 我们将定义, 而不是一个由 WQ 、 WK
nthis case we will define, instead of one big matrix made. of. three, w Q, WK and. wy,
和 WV 组成的大矩阵, 我们将定义三个不同的矩阵
n this case we will define, instead of one big matrix made. of. three, W Q, w Kand. wv,
和 WV 组成的大矩阵, 我们将定义三个不同的矩阵
we will-define'three different matrices.
两种系统都可以.
Both systems are fine.
你可以定义为一个大的矩阵, 或者三个单独的矩阵.
You can define it as one big matrix or three. separately.
实际上, 这不会改变任何东西
It doesn't change anything actually..
所以交叉来自键和值.
So the cross is from the keys and the values.
哎呀, 线性.
Oops, linear.
然后我们保存这个交叉注意力的头数, 以及每个头将看到的
Then. we. save the number of heads of this cross attention. and also the dimension. of.
然后我们保存这个交叉注意力的头数, 以及每个头将看到的
how much information each head will. see.
信息维度.
how much information each head will. see.
头等于嵌入除以头数. 让我们定义前向方法:x 是我们的
and. the. head is equal to the embed divided by the number of heads. let's. define. the
所以我们正在关联×, 这是我们的延迟, 其大小为批次
forward method x is our query andy is our key and values. so. we are. relating x which
所以我们正在关联×, 这是我们的延迟, 其大小为批次
is our latency
大小. 它将有一个序列长度, 自己的序列长度, q
大小. 它将有一个序列长度, 自己的序列长度, 9一
Awhich is. of size batch size it will have a sequence length its. own. sequence. length. q
提示, 这将是批次大小.
. let's call it q and its own dimension and they which is the. context or the-prompt
提示, 这将是批次大小.
which will be batch size
猕猴桃的序列长度, 因为提示将成为键和值, 每个都有自己
sequence length of the kiwi because the prompt will become. the. key. and the values. and.
我们可以说这将是77 的批次大小, 因为我们的提示序列
:each. of. them will have its own embedding size dimension. of. kiwi we. can already say.
我们可以说这将是77 的批次大小, 因为我们的提示序列
that this
我们可以说这将是77的批次大小, 因为我们的提示序列
Awill. bea. batchsizeof77because our sequence length of. the-prompt is77 and. each
长度是77, 每个嵌入的大小是768. 所以让我们构建这个.
will. be a. batch size of 77 because our sequence length of. the prompt is77. and. each
长度是77, 每个嵌入的大小是768. 所以让我们构建这个.
embedding isofsize768solet'sbuild this one.
这个输入形状等于x shape. 好的, 然后我们有中间形状, 和
this. is input shape is equal to x. shape okay then we have. the interim shape like. the
之前一样.
same as before so
所以这是序列长度, 然后是n, 头数, 以及每个头将看到
same as before so
所以这是序列长度, 然后是n, 头数, 以及每个头将看到
this is. the sequence length then the nnumber of heads and. how much. information. each.
的信息量:头.
head will see the head the first thing we do is multiply queries by. wq metrics. so
我们做的第一 一件事是将查询乘以wq metrics, 所以查询等于.
head will see the head the first thing we do is multiply queries by wq metrics. so
我们做的第一 一件事是将查询乘以wq metrics, 所以查询等于.
query is
我们做的第一件事是将查询乘以wq metrics, 所以查询等于.
equal. to. then we do the same for the keys and the values. but. by using the other.
然后我们对键和值做同样的事情, 但使用其他矩阵, 正如我
equal. to. then we do the same for the keys and the values. but. by using the other.
然后我们对键和值做同样的事情, 但使用其他矩阵, 正如我
matrices
之前告诉你的;键和值是 Y而不是 X. 我们再次将它们
之前告诉你的;键和值是 Y而不是 X. 我们再次将它们
:and as. l. told you before the key and the values are the Y-and not. the X again. we
分成 H个头所以 H个头数. 然后我们转置
:and as. l. told you before the key and the values are the Y and not the X again. we
我不会写形状, 因为它们匹配我们在这里做的相同变换
split. them. into H heads so Hnumber of heads then we transpose. I will not. write. the.
我们再次计算权重, 即注意力, 作为查询乘以键的转置
again. we calculate the weight which is the attention as. query. multiplied by. the
然后我们将其除以每个头的维度, 取平方根. 然后我们做soft max.
transpose of the keys and then we divide it by the dimension of each head by. the.
然后我们将其除以每个头的维度, 取平方根. 然后我们做soft max.
square root then we do the soft max.
在这种情况下, 我们没有任何因果掩码, 所以我们不需要像
n. this case we don't have any causal mask, so we don't need. to apply. the. mask. like
以前那样应用掩码, 因为在这里我们试图关联to ken, 所以提示
n. this case we don't have any causal mask, so we don't need. to. apply. the. mask. like
与像素.
:before because here we are trying to relate the tokens, so. the. prompt with. the
与像素.
pixels.
所以每个像素可以看任何token 的词, 基本上任何token 也
So. each. pixel can watch any word of the token and any token. can. watch. any pixel.
可以看任何像素.
basically.
所以我们不需要任何掩码.
So we don't need any mask.
所以为了得到输出, 我们将其乘以头矩阵, 然后输出再次
So. to. obtain the output we multiply it by the bead matrix and then. the output again
转置, 就像之前一样.
is transposed just like before.
所以现在我们正在做完全相同的事情, 就像我们在这里做的
So now we are doing exactly the same things that we. did. here
那样.
So now we are doing exactly the same things. that. we. did. here.
所以转置, 重塑等.
然后返回输出. 这就结束了我们的构建. 现在让我给你展示
and then return output and this ends our building of the let. me show. you. now. we. have.
size of the 然后返回输出. 这就结束了我们的构建. 现在让我给你展示
Multi Head ( Q. K, V) = Concat (head,
我们已经构建了稳定扩散的所有构建块, 所以现在我们可以
最终将它们组合在一起.
所以我们接下来要做的是创建一个系统, 该系统将接收噪声
接收文本, 接收时间嵌入并运行.
例如, 如果我们想做文本到图像, 我们会根据计划多次通过
单元运行这个噪声. 所以我们将构建调度器, 这意味着,
因为单元被训练来预测有多少噪声、但我们然后需要去除这个
所以要从一个噪声版本转到, 获得一个较少噪声的版本,
noise so to
我们需要去除单元预测的噪声, 这个工作由调度器完成.
权重, 然后将所有这些组合在一起.
model
我们实际上构建了所谓的管道.
And we actually build what i s. ealled the pipeline
所以文本到图像, 图像到图像等的管道.
So the pipeline of text to image, image to image.
我们开始吧.
A ad let'sgo.
现在我们已经构建了单元的所有结构, 我们构建了变分自
Now that. we have built all the structure of the unit, we have built the variational
现在是时候将它们全部组合在一起了.
Now it's time to combine it all together.
所以我首先请您做的是实际下载稳定扩散的预训练权重, 因为
So. the first thing I kindly ask you to do is to actually download the pre-trained
我们稍后需要推理它.
weights of the stable diffusion because we need to inference it later.
所以如果你去这个仓库, 我分享了这个, Py Torch 稳定扩散.
So if you go to the repository, I shared this one, Py Torch stable diffusion.
所以如果你去这个仓
ested fine 个仓库, 我分享了这个, Py Torch -稳定扩散.
Tested fine-tuned models :
你可以直接从 Hugging Face. 的网站下载 Stable Diffusion. 1. 5的预训练
Tested fine-tuned models :
You can download che pre-trainedweights of the Stable Diffusion L. 5 directly from
Tested fine-tuned models :
and sve it in the t foldr 权重.
-the-website of Hugging Face.
所以你下载这里的这个文件, 这是 EMA, 意思是指数移动
平均, 这意味着它是一个已经被训练过的模型, 但他们没有
在每次送代中改变权重, 而是使用了一个指数移动平均计划.
safe tenser s'
ago
ago 所以这对推理很有好处.
So this is good for inferencing.
ago 这意味着权重更稳定.
It means that the weights are more stable.
dt lag "sat eten secs'
v1-5-pr 但如果你想稍后微调模型, 你需要下载这个.
But if you want to fine-tune later the model, you need to download this one.
saf v tense rs'
我们还需要下载tokenizer 的文件, 国为当然, 我们会给模型
些提示来生成图像.
will give some prompt to the model to generate an image
提示需要由tokenizer 进行分词, tokenizer 将单词转换为标记, 并将
And the prompt needs to be tole nized by a tokenizer, which will convert the words
ago 标记转换为数字.
into tokens and the tokens into numbers.
然后这些数字将被我们的 CLIP 嵌入映射为嵌入.
然后这些数字将被我们的 CLIP 嵌入映射为嵌入.
:: The numbers will then be mapped into embeddings by our clip embedding here.
所以我们需要为tokenizer 下载两个文件.
So we need to download two files for the tol ken izer.
ago 首先, 这个文件的权重.
So first of all, the weights of this one file here.
v1-5-pr 然后在tokenizer文件夹中, 我们找到merges. txt和vocab js on.
The n on the tokenizer folder, we find the merges, tut and the ve cab. json.
如果我们看一下我已经下载的vocab. json 文件, 它基本上是
lf we look at the vocab. json file, which I already-downloaded,
如果我们看一下我已经下载的vocab. json 文件, 它基本上是
个词汇表, 所以每个标记都映射到一个数字, 就是这样
it's basically a vocabulary so each token mapped to a number that's it just like what.
就像tokenizer 所做的那样.
it's basically a vocabulary'so each token mapped to a number that's it just like what.
就像tokenizer 所做的那样.
the. tokenizer does and then i also prepared the picture of a dog that i will be using
然后我还准备了一张狗的图片, 我将用它来进行图像到图像的
At he. tokenizer does and then i also prepared the picture of a dog that i will be using
然后我还准备了一张狗的图片, 我将用它来进行图像到图像的
for
然后我还准备了一张狗的图片, 我将用它来进行图像到图像的
image. to. image but you can use any image you don't have to use the one i am. using of
转换. 但你可以使用任何图片, 当然不必使用我正在使用的
image. to. image but you can use any image you don't have to use the one i am usingof
所以现在让我们首先构建管道, 看看我们如何推理这个稳定
course so now let'sfirst build the pipeline so how we will inference this stable
所以现在让我们首先构建管道, 看看我们如何推理这个稳定
diffusion
扩散模型, 然后在构建管道的过程中, 我也会向你解释调度
diffusion
扩散模型, 然后在构建管道的过程中, 我也会向你解释调度
model and. then while building the pipeline i will also explain you how the scheduler
器是如何工作的.
will work
我们稍后会构建调度器. 我会解释所有的公式, 背后的所有
and. we will build the scheduler later I will explain all the formulas, all the
数学原理. 所以让我们开始吧.
mathematics behind it so let'sstart let'screate a new file, let'scall it
让我们创建一个新文件, 命名为pipeline. py, 然后我们导入通常
mathematics behind it so let'sstart let'screate a new file, let'scall it
让我们创建一个新文件, 命名为pipeline. py, 然后我们导入通常
pipeline. py and we
让我们创建一个新文件, 命名为pipeline. py, 然后我们导入通常
import the usual stuff numpy numpy as empty we will also use t qdm..
的东西一一numpy, numpy 一 作为np.
import the usual stuff numpy numpy as empty we will also use t qdm..
我们还将使用t qdm 来显示进度条.
import the usual stuff numpy numpy as empty we. will also use t qdm.
我们还将使用t qdm 来显示进度条.
稍后我们将构建这个采样器, DD PM 采样器, 我们稍后会构建它.
And. later we will build this sampler, the DD PM sampler, and we will build. it later.
我还会解释这个采样器在做什么, 它是如何工作的, 等等
And l will also explain what is this sampler doing and how. it works, et. ceterayet
等等.
cetera.
所以首先, 让我们定义一些常量.
So first of all, let's define some constants.
稳定扩散只能接受、生成512×512大小的图像
The. stable diffusion can only accept, produce images. of size512by-512.
所以高度是512x512.
So height is 512by512.
潜在尺寸.
The latent size,
潜在维度是变分自编码器的潜在张量的大小.
latency dimension is the the size of the latency tensor of. the variational.
而且, 正如我们之前看到的, 如果我们检查大小, 变分自
auto encoder and as we saw before if we go check the size the. encoder. of. the.
编码器的编码器会将512×512的图像转换为512除以8的
variational auto encoder will ta
编码器的编码器会将512×512的图像转换为512除以8的
convert something that is 512by512 into something that is 512. divided. by8. so. the.
convert something that is 512
is. 512divided. by8. so. the
所以潜在维度是512除以8, 高度也是如此, 512除以
convert something that is 512by512 into something that is. 512. divided. by8. so. the
13. 我们也可以称之为宽度除以8和高度除以8. 然后
14. 我们也可以称之为宽度除以8和高度除以8. 然后
:512. dividedby 8we can also call it width divided by 8andheightdivided. by. 8. then.
这将是我们进行文本到图像和图像到图像转换的主要函数, 它
we. create-a function called the generator this will be the main. function. that will. be
这将是我们进行文本到图像和图像到图像转换的主要函数, 它
allow us
这将是我们进行文本到图像和图像到图像转换的主要函数, 它
to do. text to image and also image to image which accepts a prompt which. is a string
接受一个提示, 这是一个字符串, 一个无条件提示.
to do. text to image and also image to image which accepts a prompt which. is a string
接受一个提示, 这是一个字符串, 一个无条件提示.
接受一个提示, 这是一个字符串, 一个无条件提示.
w an. unconditional prompt so unconditional prompt this is also. called. the-negative
如果你曾经使用过稳定扩散, 例如通过 Hugging Face 库, 你会
prompt. if-you ever used stable diffusion for example with. the. hugging face. library
如果你曾经使用过稳定扩散, 例如通过 Hugging Face 库, 你会
you will know
知道有一个.
you will know
你也可以指定一个负面提示, 告诉你想要什么, 例如, 你
that. there. is a you can also specify a negative prompt which. tells that you. want. for.
你也可以指定一个负面提示, 告诉你想要什么, 例如, 你
example you want a picture of a cat but you don't want the. cat
想要一张猫的图片, 但你不想猫在沙发上. 所以, 例如
example you want a picture of a cat but you don't want the. cat.
想要一张猫的图片, 但你不想猫在沙发上. 所以, 例如
想要一张猫的图片, 但你不想猫在沙发上. 所以, 例如
to be. on. the sofa so for example you can put the word sofa in. the. negative. prompt. so.
图像时, 它会尽量避免沙发的概念. 类似这样.
it will. try. to go away from the concept of sofa when generating. the image something.
图像时, 它会尽量避免沙发的概念. 类似这样.
like this
这与我们之前看到的无分类器引导有关. 所以, 但别担心
and. this. is connected with the classifier free guidance that we saw. before so. but
在我们构建它的时候, 我会重复所有的概念. 所以这也是
w:and. this is connected with the classifier free guidance that we saw. before so. but.
一个字符串.
:don't worry I will repeat all the concept while we are building. it so this. is also. a
个字符串.
string we can
如果我们正在构建图像到图像的转换, 我们可以有一个输入
string we can
如果我们正在构建图像到图像的转换, 我们可以有一个输入
have an input image in case we are building an. image. to. image
如果我们正在构建图像到图像的转换, 我们可以有一个输入
图像, 然后我们有一个强度参数.
图像, 然后我们有一个强度参数.
And then we have the strength..
强度, 我稍后会向你展示它是什么, 但它与我们是否有一个
Strength, I will show you later what is it, but it's related to if. we have an input
输入图像以及我们从一张图像生成另一张图像时, 希望对初始
image and. how much if we start from an image to generate :another. image y. how. much.
输入图像以及我们从一张图像生成另一张图像时, 希望对初始
attention we want to pay to the initial starting image.
图像给予多少关注有关.
attention we want to pay to the initial starting image.
我们还可以有一个参数叫做do Cfg, 意思是执行无分类器引导.
And we. can also have a parameter called do Cfg, which means. do classifier-free
我们还可以有一个参数叫做do Cfg, 意思是执行无分类器引导.
guidance.
我们将其设置为"是"
We set it to yes.
cfg Scale, 这是我们希望模型对提示给予多少关注的权重.
A cfg Scale, which is the weight of how much we want the model to. pay attention. to. our
cfg Scale, 这是我们希望模型对提示给予多少关注的权重.
prompt.
这是一个从1到14的值.
It's avalue that goes from 1to14.
我们从7. 5开始.
We start with 7. 5.
采样器名称, 我们只会实现一个, 叫做ed pm.
The sampler name, we will only implement one, so. its called ed pm..
我们想要进行多少次推理步骤, 我们将进行50 次. 我
how many. inference steps we want to do and we will do50. ithink. it's quite. common. to
认为进行50 步相当常见, 这实际上能产生不错的结果.
how many. inference steps we want to do and we will do5o. i. think. it's quite. common. to
模型是预训练的模型.
do. 50 steps which produces actually not bad results the models. are the. pre-trained
模型是预训练的模型
models the
种子是我们如何初始化随机数生成器. 让我换行, 否则我们
models the
读起来会发疯.
otherwise we become crazy reading this okay. new line
好的, 换行, 所以种子. 然后我们有一个设备, 我们
otherwise we become crazy reading this okay. new line u
好的, 换行, 所以种子. 然后我们有一个设备, 我们
好的, 换行, 所以种子. 然后我们有一个设备, 我们
So seed, then we have the device where we want to create. our tensor.
CUDA 上, 然后我们不再需要这个模型, 我们就把它移到 CPU We have idle. device, which means basically if we load some. model. on. cu DA. and then. we.
CUDA 上, 然后我们不再需要这个模型, 我们就把它移到 CPU don't need the model, we move it to the. CPu..
don't need the model, we move it to the. c Pu.
然后是我们稍后会加载的tokenizer.
And then the tokenizer that we will load later.
这是我们的方法, 这是我们的主流程, 给定所有这些信息
e this is. our method, this is our main pipeline that given all. this. information will.
将生成一张图片, 因此它会关注提示, 如果有输入图像
generate. one picture so it will pay attention to the prompt, it will pay attention. to
将生成一张图片, 因此它会关注提示, 如果有输入图像
the input
它会根据我们指定的权重(如强度和 CFG 比例)来关注
image if. there is according to the weights that we have specified. so the strength. and.
输入图像. 我会重复所有这些概念.
the CFG. scale I will repeat all this concept, don't worry later. l. will explain. them
别担心, 稍后我会实际解释它们是如何工作的, 包括在代码
the CFG. scale I will repeat all this concept, don't worry later. I will explain. them
别担心, 稍后我会实际解释它们是如何工作的, 包括在代码
actually
别担心, 稍后我会实际解释它们是如何工作的, 包括在代码
how. they work also on the code level so let'sstart the first. thing we. do. is
层面. 所以让我们开始吧.
how. they work also on the code level so let'sstart the. first. thing we. do. is
我们做的第一件事是禁用, 好的, torch nog red, 因为我们正在
how. they work also on the code level so let'sstart the first. thing we. do. is
我们做的第一件事是禁用, 好的, torch nog red, 因为我们正在
我们做的第一件事是禁用, 好的, torch nog red, 因为我们正在
We. disable, okay, torch. nog red because we are inferencing the. model..
推理模型.
We. disable, okay, torch. nog red because we are inferencing the. model.
我们首先要确保的是强度应该在0到1之间.
o The first thing we make sure is the strength should be. between zero and one.
所以如果.. 那么我们就抛出一个错误.
So if... Then we raise an error.. x
抛出值错误.
Raise value error.
如果我们要将东西移到 CPU, 必须在0和1之间
mustbe. between 0and1 with idle device if we want to move things to. the CPu-we
如果我们要将东西移到 CPU, 必须在0和1之间
create this lambda function otherwise
否则我们创建这个lambda 函数. 然后我们创建将使用的随机数
create this lambda function otherwise.
生成器.
Then we create the random number generator that we will use.
我想我在这方面有些混乱.
I think I made some mess with this.
所以这个应该像这样.
So this one should be like here.
还有生成器.
And the generator.
是一个我们将用来生成噪声的随机数生成器, 如果我们想用
is a :random. number generator that we will use to generate. the noise and. if. we want. to
否则我们手动指定一个.
start. it with the seed so if seed then we generate with the random seed otherwise-we
否则我们手动指定一个.
specify one manually
让我修复这个格式, 因为我不懂格式化文档.
o Let me. fix this formatting because I don't know... format. document.. okay..
现在至少.. 然后我们定义clip.
Now at least the... Then we define clip.
clip 是我们从预训练模型中提取的模型
. The clip is a model that we take from the pre-trained. models.
所以它里面会有clip 模型.
So it will have the clip model inside
所以这个模型, 基本上.
So this model here, basically.
这个在这里.
This one here.
我们将其移到我们的设备上.
We move it to our device.
好的, 如你所记, 使用无分类器指导, 让我回到我的
okay-as you. remember with the classifier free guidance so let. me-go back. to my-slides
好的, 如你所记, 使用无分类器指导, 让我回到我的
幻灯片 我们推理摸型
幻灯片, 当我们进行无分类器指导时, 我们推理模型两次.
首先通过指定条件, 即提示, 然后通过不指定条件, 即
没有提示.
然后我们用一个权重线性组合模型的输出, 这个权重 W
就是这里的 CFG 比例.
就是这里的 CFG 比例.
this weight here, CFG scale.
它表示我们希望在多大程度上关注有条件的输出相对于无条件的
Jt indicates how much we want to pay attention to the conditioned. output. with respect.
输出, 这也意味着我们希望模型在多大程度上关注我们指定的
. to. the. unconditioned output, which also means that how-much. we. want the. model. to-pay
输出, 这也意味着我们希望模型在多大程度上关注我们指定的
attention to the condition that we have specified.
条件.
attention to the condition that we have specified.
条件是什么?
What is the condition?..
提示, 我们写下的文本提示
The prompt, the textual prompt that we have-written.
而无条件的实际上也会使用负面提示.
所以在稳定扩散中使用的负面提示、也就是这里的这个参数
所以在稳定扩散中使用的负面提示, 也就是这里的这个参数
prompt that you use in stable diffusion which is this parameter here so. unconditioned.
无条件提示, 这就是无条件的输出.
prompt
无条件提示, 这就是无条件的输出.
this. is the. unconditional output so we we will sample. the we-will inference from. the..
所以我们会对模型进行两次推理, 一次有提示, 一次没有
this. is the-unconditional output so we we will sample. the we. will inference from. the.
有提示的一次, 另一次是无条件提示, 通常是空文本
model. twice. one with the prompt one without with the one. with the prompt one with. the
有提示的一次, 另一次是无条件提示, 通常是空文本
unconditioned prompt which is usually an empty. text. an. empty string
空字符串, 然后我们通过这个组合两者, 这将告诉模型.
unconditioned prompt which is usually an empty. text. an. empty string
空字符串, 然后我们通过这个组合两者, 这将告诉模型
通过使用这个权重, 我们将以这样的方式组合输出, 以便
and. then. we combine the two by this and this will tell. the model. by. using. this weight..
我们可以决定我们希望模型在多大程度上关注提示. 所以让
attention to the prompt so let'sdo it if we want to. do. classifier free guidance
我们来做吧. 如果我们想做无分类器指导, 首先使用tokenizer attention to the prompt so let'sdo it if we want to do classifier free guidance
将提示转换为标记.
first convert the prompt into tokens using the. tokenizer.
我们还没有指定什么是tokenizer, 但稍后我们会定义它. 所以条件
o we didn't specify what is the tokenizer yet but later we will. define it. so the.
标记, tokenizer, 批量编码加.
conditional. tokens tokenizer batch encode plus we want to. encode. the prompt we. want.
我们想要编码提示, 我们想要填充到最大长度, 这意味着
conditional tokens tokenizer batch encode plus we want to. encode. the prompt we. want.
我们想要编码提示, 我们想要填充到最大长度, 这意味着
to append the padding up to the maximum. length
如果提示太短, 它将用填充物填充, 最大长度, 如你
to append the padding up to the maximum. length.
如果提示太短, 它将用填充物填充, 最大长度, 如你
如果提示太短, 它将用填充物填充, 最大长度, 如你
:c which. means that the prompt if it's too short it will fill up. it with-paddings and
所记, 是77, 因为我们在这里也定义了它.
:cwhich. means that the prompt if it's too short it will fill up. it with. paddings and
序列长度是77, 我们取这个tokenizer 的输入 ID.
the max. length as you remember is77 because we have. also. defined it. here the
然后我们将这些标记(输入 ID)转换为张量, 其大小
77and we take the input ids of this tokenizer then we convert. these tokens. which-are
然后我们将这些标记(输入 ID )转换为张量, 其大小
input ids into a tensor
然后我们将这些标记(输入 ID )转换为张量, 其大小
which. will. be of size batch size and sequence length so conditional tokens and we-put
为批次大小和序列长度, 所以是条件标记. 现在我们将其
which. will. be of size batch size and sequence length so conditional tokens. and we-put
放在正确的设备上. 我们通过 CLIP 运行它, 因此它将转换
it. in the. right device now we run it through clip so it will convert. batch. size and
批次大小和序列长度
it. in the. right device now we run it through clip so it will. convert. batch. size and
所以这些输入 ID将被转换为大小为768的嵌入, 每个
sequence length so these input ids will be converted into. embeddings
所以这些输入 ID将被转换为大小为768的嵌入, 每个
所以这些输入 ID将被转换为大小为768的嵌入, 每个
of. size768each vector of size768solet's call it dimand. what we. do is
向量大小为768, 所以我们称之为dim, 我们所做的是条件
of. size768each vector of size768solet'scallit dimand. what we. do is
所以我们正在获取这些标记并通过 CLIP 运行它们, 所以这里
conditional context is equal to clip of conditional tokens. so we are taking. these.
所以我们正在获取这些标记并通过 CLIP 运行它们, 所以这里
tokens and we are
所以我们正在获取这些标记并通过 CLIP 运行它们, 所以这里
running them through clips so this forward method here which. will return. batch. size
的这个前向方法将返回批次大小, 序列长度, 维度, 这
running them through clips so this forward method here. which. will return. batch. size
正是我在这里写的内容.
sequence length dimension and this is exactly what i have. written. here.
我们对无条件标记做同样的事情, 所以负面提示, 如果你
We do the. same for the unconditioned tokens, so the negative prompt, which. if you.
不想指定, 我们将使用空字符串, 这意味着模型的无条件
don't want to specify, we will use the empty string, which means. the unconditional
不想指定, 我们将使用空字符串, 这意味着模型的无条件
output of the model. cx au
输出.
output of the model.
所以模型, 在没有条件的情况下会生成什么?
So. the model, what would the model produce without any condition?..
所以如果我们从随机噪声开始并要求模型生成图像, 它将生成
So if. we. start with random noise and we ask the model to produce. an. image y it will.
所以如果我们从随机噪声开始并要求模型生成图像, 它将生成
produce an image, but without any condition.
所以模型将根据初始噪声输出它想要的任何内容
So the model will output anything that it wants based on the initial. noise..
我们将其转换为张量然后通过 CLIP 传递, 就像条件标记
c we. convert it into tensor then we pass it through clips just like. the conditional
样. 所以它们会.
:owe. convert it into tensor then we pass it through clips just. like. the conditional
它将成为标记, 是的:所以它也将成为大小为批次大小
tokens so. they will it will become tokens yes so it will also. become a tensor of. size.
它将成为标记, 是的, 所以它也将成为大小为批次大小
batch size
它将成为标记, 是的, 所以它也将成为大小为批次大小
sequence length dimension where the sequence length is. actually-always. 77-and. also. in.
序列长度, 维度的张量, 其中序列长度实际上总是77, 在
sequence length dimension where the sequence length. is actually. always. 77-and. also. in.
序列长度, 维度的张量, 其中序列长度实际上总是77, 在
thiscaseit was always77because it's the. max. lengthhere
序列长度, 维度的张量, 其中序列长度实际上总是77, 在
这个例子中也是77, 因为这里是最大长度, 但我忘了写代码
这个例子中也是77, 因为这里是最大长度, 但我忘了写代码
buti. forgot to write the code to convert it into so unconditional tokens is. equal. to
将其转换为. 所以无条件标记等于tokenizer 批量加. 所以
but i forgot to write the code to convert it into so unconditional tokens is. equal. to
无条件提示, 也是负面提示, 填充与之前相同.
tokenizer. batch plus so the unconditional prompt so also. the negative prompt. the
无条件提示, 也是负面提示, 填充与之前相同.
padding is the same as before so max. length.
所以最大长度和max Length 定义为77, 我们从这里获取输入 ID
paddingisthe same asbefore somax. length..
所以最大长度和max Length 定义为77, 我们从这里获取输入 ID.
所以最大长度和max Length 定义为77, 我们从这里获取输入 ID.
and the max Length is defined as 77 and we take the input IDs. from. here so. now we. have.
所以现在我们有了这两个提示, 我们所做的是将它们
and the max Length is defined as 77 and we take the input IDs. from. here so now we have.
连接起来. 它们将成为我们输入单元的批次.
these. two. prompts what we do is we concatenate them they. will become the. batch. of. our
连接起来. 它们将成为我们输入单元的批次.
input to the unit
好的, 所以基本上我们所做的是我们获取条件和无条件的输入
_ Okay, so basically what we are doing is we are taking. the conditional. and
并将它们组合成一个单一的张量
unconditional input and we are combining them into. one. single. tensor
所以它们将成为一个批次大小为2的张量, 即2, 序列
So. they. will become a tensor of batch size 2, so2, sequence. length and. dimension
其中序列长度实际上. 我们已经可以写成2乘77乘
where sequence length is actually, we can already write it. will become. 2. by77. by768.
15. 因为77是序列长度, 维度是768.
because 77 is the sequence length and the dimension is 768 if we don't. want. to. do
如果我们不想做条件分类器, 自由引导, 我们只需要使用
because 77 is the sequence length and the dimension is 768 if we don't. want. to. do
如果我们不想做条件分类器, 自由引导, 我们只需要使用
conditional
如果我们不想做条件分类器, 自由引导, 我们只需要使用
classifier. free guidance we only need to use the prompt and. that's it so. we do. only
所以我们只通过单元进行一步, 并且只使用提示, 不将
one step through the unit
所以我们只通过单元进行一步, 并且只使用提示, 不将
:c and only with the prompt without combining the unconditional input with. the
无条件输入与条件输入结合起来
:and only with the prompt without combining the unconditional input with. the
无条件输入与条件输入结合起来
conditional input but in this case we can not decide how much. the model pays attention.
但在这种情况下, 我们无法决定模型对提示的关注程度
conditional input but in this case we can not decide how much. the. model pays. attention.
但在这种情况下, 我们无法决定模型对提示的关注程度
to the the prompt
但在这种情况下, 我们无法决定模型对提示的关注程度
because we don't have anything to combine it with so again we. take the just. the
所以再次, 我们只取提示, 就像之前一样, 我们可以取
because. we don't have anything to combine it with so again we. take the just. the
所以再次, 我们只取提示, 就像之前一样, 我们可以取
prompt just like before
所以再次, 我们只取提示, 就像之前一样, 我们可以取
vo we can. take it, let's call it just tokens and then we transform. this into a. tensor
然后我们将这个转换为张量, 张量长, 我们将其放在正确的
we can. take it, let'scall it just tokens and then we transform. this into a. tensor
然后我们将这个转换为张量, 张量长, 我们将其放在正确的
tensor long we put it in the right device
设备上, 我们计算上下文, 这是一个大张量, 我们通过
tensor long we put it in the right device
设备上, 我们计算上下文, 这是一个大张量, 我们通过
设备上, 我们计算上下文, 这是一个大张量, 我们通过
we. calculate the context which is a one big tensor we pass. it through clip. but. this
CLIP 传递它, 但在这种情况下它将只有一个, 只有一个.
we. calculate the context which is a one big tensor we pass. it through clip. but. this
所以批次大小将是一, 所以批次维度, 序列, 再次是77 case it will be only one only one so the batch size will be one so the batch
所以批次大小将是一, 所以批次维度, 序列, 再次是77 dimension the
所以批次大小将是一, 所以批次维度, 序列, 再次是77
sequence. is. again77and the dimension is 768so here we are. combining. two prompt..
维度是768. 所以这里我们结合了两个提示, 这里我们结合
here we are combining one.
here we are combining one
因为我们将在模型中运行两个提示, 一个无条件, 一个有
Because we will run through the model two prompts, one. unconditioned. and. one..
所以一个带有我们想要的提示, 一个带有空字符串, 模型将
conditioned, so one with the prompt that we want, one with. the. empty string, and. the
所以一个带有我们想要的提示, 一个带有空字符串, 模型将
model will
产生两个输出, 因为模型处理批次大小. 这就是我们为什么
model will
产生两个输出, 因为模型处理批次大小. 这就是我们为什么
produce. two outputs, because the model takes care of. the batch. size, that's why we
既然我们已经完成了使用 CLIP, 我们可以将其移动到空闲设备.
Since we have finished using the clip we can move it. to the. idle. device.
这实际上非常有用, 如果你有一个非常有限的 GPU 并且你想
This. is very useful actually if you have a very limited GPU. and you want to offload
在使用模型后卸载它们, 你可以通过再次将它们移动到 CPU the models after using them, you can offload them back to. the. CPu by moving. them. to
来卸载它们.
the CPU again.
然后我们加载采样器.
Then we load the sampler.
目前我们还没有定义采样器, 但我们使用它, 稍后我们构建
For. now we didn't define the sampler, but we we use it and. later. we build. it because
它, 因为如果你在知道如何使用它之后构建它, 我认为更
lt'sbetter to build it after you know how it is used if we. build. it. before. l think
它, 因为如果你在知道如何使用它之后构建它, 我认为更
it's easy to get lost and what is it happening?.
容易理解发生了什么.
it's easy to get lost and what is it happening?
实际上发生了什么?
What's happening actually so if the sampler name. is DD PM
所以如果采样器名称是 DD PM, ddpm, 那么我们构建采样器ddpm
What's happening actually so if the sampler name. is DD PM.
所以如果采样器名称是 DD PM, ddpm, 那么我们构建采样器ddpm
所以如果采样器名称是 DDPM, ddpm, 那么我们构建采样器ddpm
ddpm. then. we build the sampler dd pm sampler we pass it to. the noise-generator. and. we
希望在推理中进行多少步, 我稍后会向你展示为什么. 如果
tell the sampler how many steps we want to do for the inferencing and. i will show-you
采样器不是dd pm, 那么我们抛出一个错误, 因为我们没有实现
later
采样器不是dd pm, 那么我们抛出一个错误, 因为我们没有实现
what. why if the sampler is not dd pm then we erase an error. because we didn't
任何其他d d, 任何其他采样器, 所以已知采样器.
:owhat. why if the sampler is not dd pm then we erase an error. because we. didnt
让我做f 采样器名称好吧.
implement any other dd any other sampler so known. sampler. let me-dof
让我做f 采样器名称好吧.
为什么我们需要告诉他多少步?
Why we need to tell him how many steps?
因为正如你记得的, 让我们回到这里, 这个调度器需要进行
Because as you remember, let'sgo. here
因为正如你记得的 让我们回到这里,
Because
因为正如你记得的, 让我们回到这里, 这个调度器需要进行
很多步
我们告诉他我们想要进行的确切步数.
所以在这种情况下, 噪声化步骤将是50.
So in this case, f he noise-ification steps will be 50.
即使训练期间我们有最多1000 步, 在推理期间我们不需要
进行1000步
dontneed to do 1o00 steps
我们可以做更少.
Wle can do less.
当然, 通常你做的步骤越多, 质量越好.
Msu ally them one steps you do, fhe better the
因为你能去除的噪声越多.
Because the more noise you can ne mom
但不同的采样器工作方式不同. 而对于ddpm, 通常50步就
是以得到一个不错的结果.
对于某些其他采样器, 例如ddim, 你可以做更少的步骤.
对于一些基于微分方程工作的采样器, 你可以做得更少. 这
对于一些基于微分方程工作的采样器, 你可以做得更少. 这
even less. depends on which sampler you use and how. how. lucky you are. with. the.
取决于你使用的采样器以及你如何使用, 以及你在特定提示下
even less. depends on which sampler you use and how. how. lucky you are. with. the
的运气如何.
particular prompt actually also
实际上还有, 这是将通过单元的潜在变量, 如你所知, 它
particular prompt actually also x
实际上还有, 这是将通过单元的潜在变量, 如你所知, 它
This. is. the La tents that will run through the unit and as. you. know it's. of size..
的尺寸是潜在高度和潜在宽度, 我们之前已经定义过了
的尺寸是潜在高度和潜在宽度, 我们之前已经定义过了.
This. is. the La tents that will run through the unit and as you. know it's. of size.
所以它是512除以8乘以512除以8, 即64乘以
Latentsheight and Latent swidth which we defined before so. it's. 512divided. by. 8. by.
所以它是512除以8乘以512除以8, 即64乘以
512divided by
所以它是512除以8乘以512除以8, 即64乘以
:8so 64. by64and now let's do what happens if the user specifies an input image
16. 现在让我们看看如果用户指定了一个输入图像会发生
8so64. by64and nowlet's dowhat happens if the user specifies an input image
所以如果我们有一个提示我们可以通过运行无分类器引导来
So. if we have a prompt, we can take care of the prompt. by. either running
处理提示, 这意味着根据这里的比例, 将模型的输出与有
classifier-free guidance, which means combining the output. of. the. model. with. the.
提示和无提示的输出结合起来.
prompt and without the prompt, according to. this scale here.
或者我们可以直接要求模型仅使用提示输出一张图像, 但这样
or we. can. directly just ask the model to output only one image only. using. the prompt.
我们就不能将两个输出按这个比例结合.
but then we. can not combine the two output with this scale what. happens. however. if. we.
然而, 如果我们不想做文本到图像, 而是想做图像到图像,
会发生什么?
如果我们像之前看到的那样做图像到图像, 我们从一张图像
开始, 用编码器对其进行编码, 然后对其添加噪声.
然后我们要求调度器去除噪声, 噪声, 噪声.
但由于单元也会受到文本提示的调节, 我们希望在单元去噪
这张图像时, 它会朝着这个提示的方向移动.
所以这就是我们要做的.
So this is what we will do
首先, 我们加载图像, 对其进行编码, 并添加噪声
所以如果指定了输入图像, 我们加载编码器, 将其移动到
:so. if an. input image is specified we load the encoder we. move. it to. the device. in.
设备上一一例如, 如果我们使用 CUDA 一 一然后我们加载
so. if an input image is specified we load the encoder we move. it to. the. device. in.
图像的张量, 调整其大小, 确保它是512x512 的高度, 然后
case we are. using Cu DA for example then we load the tensor. of the image were size. it,
图像的张量, 调整其大小, 确保它是512x512 的高度, 然后
we make sure that it's512x512.
将其转换为numpy 数组, 再转换为张量.
将其转换为numpy 数组, 再转换为张量.
:with height and then we transform it into a numpy array-and. then into. a. tensor.
那么这里的尺寸会是多少?
So what will be the size here?
它将是高度乘以宽度乘以通道数
It will be height by width by channel.
而通道数将是3.
And the channel will be3.
接下来我们要做的是重新缩放这张图像.
The next thing we do is were scale this. image.
这意味着什么?
What does it mean?
这意味着该单元的输入应在-1到+1之间进行
That. the. input of this unit should be normalized between, should. be, sorry, rescaled
这意味着该单元的输入应在-1到+1之间进行
归一化, 抱歉, 是重新缩放.
因为如果我们加载图像, 它将有三个通道
每个通道将在0到255之间.
Each channel will be between zero and 255.
所以每个像素有三个通道, RGB, 每个数值在0到255
Soeachpielhame fhreechannels, RGt, and each m umber is between r zero
之间.
但这并不是单元想要的输入.
But this is mot what the unit wants. as input
单元希望每个通道、每个像素都在-1到+1之间
单元希望每个通道、每个像素都在-1到+1之间
the unit wants every channel, every pixel to be between minus one and plus one. so we
所以我们需要这样做
o will. do this we will build later this function, it's called rescale to transform
我们稍后会构建这个函数 它叫做rescale- 将0到
anything that
我们稍后会构建这个函数一 它叫做rescale一 一将0到
is. between0and255into something that is between minus. one and plus. one
255之间的任何值转换为-1到+1之间的值
255之间的任何值转换为-1到+1之间的值
And this will not change the size of the. tensor.
并且这不会改变张量的大小
And this will not change the size of the. tensor.
我们添加批次维度.
We add the batch dimension.
解压.
Un squeeze.
这添加了批次维度.
This adds the batch dimension..
批次大小.
Batch size.
Ok.
Okay.
然后我们改变维度的顺序
And then we change the order of the dimensions.
即0, 3, 1, 2.
which is 0, 3, 1, 2.
why?
Why?
因为, 如你所知, 变分自编码器的编码器需要批次大小
Because, as you know, the encoder of the Variational Auto encoder wants. batch. size,
宽度、 通道
channel, height and width, while we have batch size, height, width, channel.
所以我们对它们进行排列, 以获得编码器的正确输入.
So we permute them to obtain the correct input. for. the encoder.
这个进入通道、 高度和宽度, 然后这部分我们可以删除.
This one, go into channel,
这个进入通道、高度和宽度, 然后这部分我们可以删除.
v :and. height and width and then this part we can delete okay. this is. the input. then
然后我们要做的是采样一些噪声, 因为, 如你所记, 为了
and. height and width and then this part we can delete okay. this is. the input. then
运行编码器, 我们需要一些噪声, 然后它将从我们之前定义
:what. we do. is we sample some noise because as you remember the encoder. to. run-the
运行编码器, 我们需要一 一些噪声, 然后它将从我们之前定义
encoder we need
运行编码器, 我们需要一些噪声, 然后它将从我们之前定义
some. noise-and then he will sample from this particular-gaussian. that we. have defined
的特定高斯分布中采样. 所以, 编码器噪声, 我们从
before so encoder noise we sample it from our-generator
所以我们定义了这个生成器, 以便我们只需定义一个种子
So we. have defined this generator so that we can define only one seed and we can. also
这就是我们使用生成器的原因.
And this is why we use the generator.
潜在形状.
Latent s shape.
Ok.
Okay.
现在让我们通过解码器运行它.
And now let'srun it through the decoder.
通过 VA运行图像. 这将产生潜在变量, 所以输入图像
Runthe image through
通过 VA 运行图像. 这将产生潜在变量, 所以输入图像
of. the v A. this will produce latency so input image tensor. and then we give it some
张量, 然后我们给它一些噪声. 现在我们就在这里. 我们
of. the VA. this will produce latency so input image tensor. and then we give it some
生成了这个, 这是我们的潜在变量.
noise now we are exactly here we produced this this is our. latency so we-give. the.
生成了这个, 这是我们的潜在变量.
ne ewactly here we produced this this is our laten
所以我们把图像和一些噪声一起输入编码器. 它将生成这个
所以我们把图像和一些噪声一起输入编码器. 它将生成这个
encoder. along with some noise it will produce a latent representation of. this image.
现在我们需要告诉我们的, 如你所见, 我们需要给这个潜在
now. we. need to tell our um as you can see here we need. to. add. some. noise. to. this.
现在我们需要告诉我们的, 如你所见, 我们需要给这个潜在
变量添加一些噪声. 我们如何添加噪声? 我们使用调度器.
强度基本上告诉我们.
noise we wse our scheduler the str
我们在这里定义的强度参数告诉我们, 在生成输出图像时
the strength parameter that we defined here tells us how much. we want. the model. to..
我们在这里定义的强度参数告诉我们, 在生成输出图像时
我们希望模型对输入图像的关注程度.
强度越大, 我们添加的噪声就越多. 所以, 强度越大
噪声就越强.
因此, 模型将更具创造性, 国为模型有更多的噪声需要
去除, 并且可以创造出不同的图像,
但如果我们给这个初始图像添加较少的噪声, 模型就不能很有
创造性, 国为图像的大部分已经定义好了, 所以没有多少
噪声可以去除. 因此, 我们预计输出会更接近输入
所以这里的强度基本上意味着, 你添加多少噪声? 我们添加
所以这里的强度基本上意味着, 你添加多少噪声? 我们添加
uh strength. here basically means the more noise you how. much. noise to add the. more
所以这里的强度基本上意味着, 你添加多少噪声? 我们添加
的噪声越多, 输出与输入的相似度就越低.
我们添加的噪声越少, 输出与输入的相似度就越高, 因为
调度器, 抱歉, 有更少的, 改变图像的可能性, 因为
噪声较少
噪声较少
um possibility of changin is less. noise
所以让我们来做吧. 首先我们告诉采样器我们定义的强度
so. let's. do it first we tell the sampler what is the strength. that. we. have-defined.
稍后我们会看到这个方法在做什么. 但现在我们只是写下来.
so. let's. do it first we tell the sampler what is the strength. that. we. have. defined
然后我们要求采样器根据我们定义的强度, 给我们的潜在变量
and. later. we will see what is this method doing but for now. we just write. it and. then..
然后我们要求采样器根据我们定义的强度, 给我们的潜在变量
we ask the
然后我们要求采样器根据我们定义的强度, 给我们的潜在变量
sampler. to add noise to our latency here according to the strength that-we. have
添加噪声:添加噪声
:sampler. to add noise to our latency here according to the strength that we. have
添加噪声:添加噪声
defined add noise
基本上, 通过设置强度, 采样器将创建一个时间步长计划.
basically. the sampler by setting the strength will create a. time. step schedule later..
稍后我们会看到, 通过定义这个时间步长计划, 我们将开始
we will. see it and by defining this time step schedule we. will. start what. is. the
确定初始噪声水平. 因为如果我们设置噪声水平, 例如
we will. see it and by defining this time step schedule we. will. start what. is. the
确定初始噪声水平. 因为如果我们设置噪声水平, 例如
initial noise
确定初始噪声水平. 因为如果我们设置噪声水平, 例如
:o level. we will start with because if we set the noise level. to. be. for example. the
强度为1, 我们将从最大噪声水平开始.
iocstrength to be1 we will start with the maximum noise. level. but. if we. set. the.
但如果我们将强度设置为0. 5, 我们将从一半的噪声开始, 而
strengthtobe1we will start with the maximum noise. level. but. if we. set. the.
不是完全的噪声. 稍后当我们实际构建采样器时, 这一点会
strength to be 0. 5we will start with half noise not all. completely noise
不是完全的噪声. 稍后当我们实际构建采样器时, 这一点会
不是完全的噪声. 稍后当我们实际构建采样器时, 这一点会
and later. this will be more clear when we actually build. the sampler so. now just
更加清晰. 所以现在只需记住我们就在这里. 所以我们有了
remember that we are exactly here so we have the image we. transformed the. compressed
更加清断. 所以现在只需记住我们就在这里. 所以我们有了
图像.
我们将图像, 通过编码器压缩, 变成了潜在变量, 根据
强度水平添加了一些噪声, 然后我们需要将其传递给扩散
强度水平添加了一些噪声, 然后我们需要将其传递给扩散
then. we need to pass it to the model to the diffusion model so. now we don't need. the.
模型. 月 所以现在我们不再需要编码器了, 可以将其设置为
encoder anymore we can set it to the idle device
闲置设备.
encoder anymore we can set it to the idle. device
如果用户没有指定任何图像, 那么我们如何开始去噪呢?
if the. user didn't specify any image then how can we start. the denoising?.
如果用户没有指定任何图像, 那么我们如何开始去噪呢?
这意味着我们想要进行文本到图像的转换、所以我们从
这意味着我们想要进行文本到图像的转换, 所以我们从
it means. that we want to do text to image so we start with. random. noise. so we. start.
随机噪声开始. 所以我们从随机噪声开始. 让我们采样一些
it means. that we want to do text to image so we start with. random. noise ·so we. start
随机噪声, 然后生成器和设备是设备
with random. noise. let'ssample some random noise then generator. and. device is. device
所以让我写一些注释. 如果我们正在进行文本到图像的转换
so let. me. write some comments if we are doing text to image start with random. noise.
从随机噪声开始, 随机噪声定义为n01或n0i实际上.
:so let. me write some comments if we are doing text to image. start with random. noise.
然后我们最终加载扩散模型, 这是我们的单位扩散模型.
random noise defined as no1orn0i actually we then finally load. the diffusion model.
然后我们最终加载扩散模型, 这是我们的单位扩散模型.
which is our unit diffusion it's models
稍后我们会看到这个模型是什么以及如何加载它.
Later we will see what is this model and how. to load. it.
我们将其加载到我们工作的设备上, 例如 CUDA.
We. take it to our device where we are working, so. for example. cu DA
时间步长基本上意味着, 如你所记, 训练模型时我们有最多
and. then. our sampler will define some time steps time steps. basically. means. that as..
时间步长基本上意味着, 如你所记, 训练模型时我们有最多
1000个时间步长, 但在推理时我们不需要做1001步. 在
我们的例子中, 如果最大强度水平是1000, 我们将进行例如
don'tneed todo1001 steps in our case we will be doing-for example 50. steps. of.
我们的例子中, 如果最大强度水平是1000, 我们将进行例如
inferencing if the maximum strength. level is 1000.
50 步的推理.
inferencing if the maximum strength. level. is 1000...
例如 如果最大水平是1000, 最小水平将是1.
For example, if the maximum level is 1000, the minimum. level. will. be 1.
或者如果最大水平是999, 最小水平将是0.
Orif the maximum level is 999, theminimum. will. be. 0.
这是线性时间步长.
And this is linear time steps.
如果我们只做50步, 这意味着我们需要从1000开始
lf:we do. only50, it means that we need to do, for example, we. start with. 1000. and
然后每20步进行一次.
then we do every 20.
所以980, 然后960:940, 920, 900, 然后800.. 什么? 880
5o980, the n960, 940, 920, 900, then 800.., What?. 880...
直到我们到达第0 级.
Until we arrive to the Oth level.
基本上, 每个时间步长表示一个噪声水平.
:basically. each of these time steps indicates a noise level. so. when with the noise
所以在有噪声的情况下, 当我们对图像或初始噪声进行去噪时
basically. each of these time steps indicates a noise. level. so. when with the noise
所以在有噪声的情况下, 当我们对图像或初始噪声进行去噪时
when we de noise fhe the image or the initial r
(在我们进行文本到图像转换的情况下), 我们可以告诉
调度器根据特定的时间步长去除噪声, 这些时间步长由我们
调度器根据特定的时间步长去除噪声, 这些时间步长由我们
tell the scheduler to remove noise according to particular. time. steps which are
想要的推理步数定义. 这正是我们现在要做的.
defined by-how many inference steps we want and this is. exactly. what we. are going. to
想要的推理步数定义. 这正是我们现在要做的.
想要的推理步数定义. 这正是我们现在要做的.
defined by. how many inference steps we want and this. is. exactly. what we. are going. to.
想要的推理步数定义. 这正是我们现在要做的.
do now
当我们初始化采样器时, 我们告诉它我们想要进行多少步
when. we. initialize the sampler we tell him how many steps we want. to. do. and he. will
它将创建这个时间步长计划. 所以, 根据我们想要的步数
create this. time step schedule so according to how many we. want and now. we just go
现在我们只需按计划进行.
through it so
所以我们告诉时间步长, 我们创建了t qdm, 这是一个进度条
through it so
所以我们告诉时间步长, 我们创建了t qdm, 这是一个进度条
we tell. the. time steps we create t qdm which is a progress. bar. we. take. the. time steps
我们获取时间步长并对每个时间步长对图像进行去噪.
and for each of these time step swede noise-the image
所以我们有1300. 这是我们的. 我们需要告诉单位, 如
so. we. have1300 this is our we need to tell the unit as you. remember. diffusion
单位输入的是时间嵌入. 那么时间步长是什么?
the unit has. as input the time embedding so what is the. time step we. want. to. de noise
我们想要对上下文进行去噪, 这可以是提示, 或者在我们
w. the. context which is the prompt or in case we are doing a. classifier. free guidance
进行无分类器引导的情况下, 还包括无条件提示和潜在状态的
also the
进行无分类器引导的情况下, 还包括无条件提示和潜在状态的
unconditional prompt and the latent the current state of the. latent. because we will.
当前状态, 因为我们将从一些潜在状态开始, 然后不断对其
unconditional prompt and the latent the current state of the. latent. because we. will.
当前状态, 因为我们将从一些潜在状态开始, 然后不断对其
进行去噪. 你根据时间嵌入, 即时间步长, 不断对其进行
进行去噪. 你根据时间嵌入, 即时间步长, 不断对其进行
denoising it according to the time embedding. to. the. time. step
我们首先计算时间嵌入, 这是当前时间步长的嵌入, 我们将
we. calculate first the time embedding which is an embedding of. the current time. step.
从该函数中获取它. 稍后我们将定义它.
:and we will obtain it from this function later we define. it. this. function basically.
这个函数基本上会将一个数字(即时间步长)转换为一个
and we will obtain it from this function later we define. it. this. function basically
这个函数基本上会将一个数字(即时间步长)转换为一个
will
向量, 这个向量的大小为320, 描述了这个特定的时间步长
convert. a. number so the time step into a vector one. of. size. 320. that describes. this
向量, 这个向量的大小为320, 描述了这个特定的时间步长
particular time step o
正如你稍后将看到的, 它基本上就等于我们为transformer 模型所
所以在transformer 模型中, 我们使用正弦和余弦来定义位置. 在
we did. for. the transformer model so in the transformer model we. use the. sines :and. the
所以在transformer 模型中, 我们使用正弦和余弦来定义位置. 在
cosines
所以在transformer 模型中, 我们使用正弦和余弦来定义位置. 在
to. define. the position here we use the sines and cosine to. define. the time. step. and..
让我们构建模型输入, 即潜在状态, 其形状为批次大小4
let'sbuild the model input which is the latency
让我们构建模型输入, 即潜在状态, 其形状为批次大小4
让我们构建模型输入, 即潜在状态, 其形状为批次大小4 which. is of shape batch size 4, because it's the input. of. the. encoder, of the
因为它是变分自编码器的输入, 大小为4, 抱款, 它有
which. is of shape batch size 4, because it's the input. of. the. encoder, of the
4个通道, 然后是潜在高度和潜在宽度, 即64乘
:variational auto encoder, which is of size4, sorry, whichhas. 4channels, and. then. it
4个通道, 然后是潜在高度和潜在宽度, 即64乘
has latency height, and the latency width, which is. 64by 64,
4个通道, 然后是潜在高度和潜在宽度, 即64乘
17. 现在如果我们做这个, 我们需要发送, 基本上我们发送
18. 现在如果我们做这个, 我们需要发送, 基本上我们发送
now. if. we do if wedo this one we need to send basically. we. are sending. the
的是条件. 它在哪儿?
o now. if. we do if we do this one we need to send basically. we. are sending. the.
的是条件它在哪儿?
conditioned where is it here we send the conditional input but also the unconditional
需要发送无条件输入, 这意味着我们需要发送带有提示和不带
input if we do
需要发送无条件输入, 这意味着我们需要发送带有提示和不带
the. classifier free guidance which means that we need to. send the same latent. with
提示的相同潜在状态, 所以我们可以做的是, 如果我们做无
the. classifier free guidance which means that we need to. send. the same latent. with.
分类器引导, 可以将这个潜在状态重复两次.
the prompt. and without the prompt and so what we can do. is. we. can. repeat. this. latent.
分类器引导, 可以将这个潜在状态重复两次.
twice if we are doing the classifier free guidance
我将在1上放置重复.
I'm going toput by repeat on 1.
这将基本上转换批次大小为4.
This will basically transform batch size. 4.
我将在1上放置重复.
I'm going toput by repeat on1.
这将基本上转换批次大小为4.
This will basically transform batch size. 4.
我将在1上放置重复.
I'm going toput by repeat on 1.
这将基本上转换批次大小为4.
This will basically transform batch size. 4.
所以这将实际上是初始批次大小的两倍, 即1, 并且有4
so. thisis umgoing to be um twice the size of the initial. batch size. which. is one
个通道和潜在高度及潜在宽度.
actually. and four channels and latency height and latency width. so. basically. we are.
基本上, 我们将这个维度重复两次. 我们制作了两个
repeating
基本上, 我们将这个维度重复两次. 我们制作了两个
this. dimension twice we are making two copies of the latency one will be-use
潜在状态的副本. 一个将用于带有提示的情况, 一个用于
the prompt one without the prompt..
不带提示的情况.
the prompt one without the prompt.
所以现在我们检查模型输出
So now we check the model output.
模型输出是什么?
What is the model output?
它是单元预测的噪声.
It is the predicted noise by the unit.
所以模型输出是单元预测的噪声
So the model output is the predicted noise. by. the. unit.
我们进行扩散, 模型输入, 上下文和时间嵌入.
We do diffusion, model input, context and time embedding
如果我们进行无分类器引导, 我们需要结合条件输出和无条件
And if. we. do classifier-free guidance, we need to combine. the. conditional. out pu and
如果我们进行无分类器引导, 我们需要结合条件输出和无条件
the unconditional output.
输出.
the unconditional output.
因为我们传递了模型输入, 如果我们进行无分类器引导
Because we are passing the input of the model, if we are. doing classifier-f
我们给的是批次大小为2, 模型将产生一个批次大小为
guidance, we are giving a batch size of 2, the model will produce an output tha b
的输出.
batch size of 2
然后我们可以将其分成两个不同的张量
So we can then split it into two different. tensors.
个是条件性的, 另一个是无条件性的.
One will be the conditional and one will be the unconditional.
所以输出条件性和输出无条件性是通过这种方式使用chunk 分割的.
So the output conditional and the output unconditional are splitted. in. this. way w sing
所以输出条件性和输出无条件性是通过这种方式使用chunk 分 割的
chunk.
维度是沿着第0维度的, 月 所以默认是第0维度
A:the. dimension is along the Oth dimension so by default it's. the. oth. dimensio
然后我们根据这里的公式将它们结合起来, 其中
then we. combine them according to this formula here where is. the.. lll mis
然后我们根据这里的公式将它们结合起来, 其中.
Input 我将根据这里的公式省略. 所以无条件输出减去一一抱款
le ll miss out
一条件输出, 减去无条件输出, 乘以我们定义的
尺度, 再加上无条件输出.
所以模型输出将是条件尺度乘以条件输出减去无条件输出, 再
defined plus the unconditioned output so the model. output.
所以模型输出将是条件尺度乘以条件输出减去无条件输出, 再
所以模型输出将是条件尺度乘以条件输出减去无条件输出, 再
will be. conditional scale multiplied by the output conditioned. minus the. output
所以模型输出将是条件尺度乘以条件输出减去无条件输出, 再
unconditioned plus the output unconditioned and. then
然后我们基本上完成了. 现在来说, 可以说是关键部分.
unconditioned plus the output unconditioned and. then
然后我们基本上完成了. 现在来说, 可以说是关键部分.
然后我们基本上完成了. 现在来说, 可以说是关键部分.
what. we do. is basically okay now comes the let'ssay the clue part so we have a. model
所以我们有一个模型能够预测当前潜在状态中的噪声
what. we do. is basically okay now comes the let's say the clue part so we have a. model
所以我们开始, 例如, 想象我们在做文本到图像. 所以让
:o that is able to predict the noise in the current latency so we. start. for. example
所以我们开始, 例如, 想象我们在做文本到图像. 所以让
imagine we
所以我们开始, 例如, 想象我们在做文本到图像. 所以让
are doing text to image so let'me go back here we are going um text to. images
所以我们开始, 例如, 想象我们在做文本到图像. 所以让
我回到这里.
我们在这里进行文本到图像的转换. 所以我们从一些随机噪声
开始, 并将其转换为潜在状态. 然后, 根据某种调度器
根据某个时间步, 我们不断去噪. 现在我们的单元将预测
潜在状态中的噪声.
one dict the noise in the latency
但我们如何从图像中去除这种噪声以获得更少噪声的图像呢?
这是由调度器完成的.
所以在每一步、我们问单元图像中有多少噪声, 我们去除
它, 然后我们再次将其交给单元, 问还有多少噪声, 并
去除它.
然后我再次问还有多少噪声, 然后我们去除它, 然后还有
多少噪声, 当我们完成所有这些时间步后, 我们取
潜在状态, 交给解码器, 解码器将构建我们的图像. 这
time. steps we
正是我们在这里所做的. 所以想象一下我们没有输入图像,
正是我们在这里所做的. 所以想象一下我们没有输入图像
take. the latent give it to the decoder which wilt build our image. and this. is exactly
所以我们有一些随机噪声
what we. are. doing here so imagine we don't have an input image so. we have some. random
所以我们有一 一些随机噪声
noise
我们在这个采样器上定义一些时间步, 基于我们想要执行多少
we define some time steps on this sampler based on how many inference steps we want.
推理步骤.
o do
我们完成所有这些时间步, 我们将潜在状态交给单元. 单元
we. do. all this time step, we give the la tents to the unit, the. unit will. tell. us. how
噪声.
much is the predicted noise, but then we need to. remove. this noise..
所以让我们来做, 让我们去除这种噪声.
So let'sdo it, so let'sremove this noise.
所以潜在状态等于采样器. step, B 时间步
o So. the. latent s are equal to sampler. step, t me step, latent s, model, output
这基本上意味着, 从更噪声的版本中获取图像.
This basically means, take the image,
这基本上意味着, 从更噪声的版本中获取图像.
好的, 让我写得更清楚一些
Okay, let me write it better
去除单元预测的噪声
Remove noise predicted by the unit.
单元, 好的.
Unit, okay.
这就是我们的去噪循环.
And this is our loop of denoising.
然后我们可以进行空闲扩散
Then we can do to idle diffusion...
现在我们有了去噪后的图像, 因为我们已经进行了多次步骤
now we. have. our de noise d imagebecause we have done it for. many. steps now what we. do
接下来我们要做的是加载解码器, 即模型的解码器, 然后
now we. have. our de noise d imagebecause we have done it for. many. steps now what we. do
我们的图像通过解码器运行. 所以我们让潜在状态通过解码器
is we. load. the decoder which is models decoder and then our images is run. our. image.
我们的图像通过解码器运行. 月 所以我们让潜在状态通过解码器
is run through the decoder
运行.
So we run the latency through the decoder.
运行.
所以我们在这一步这样做.
So we do this ste p hene.
所以我们让潜在状态通过解码器运行.
So we run this latency through the decoder.
这将生成图像.
This will give the image.
实际上只会生成一张图像, 因为我们只指定了一张图像
It. actually will be only one image because we only. specify. one image
然后我们让图像等于, 因为图像最初, 如你所记得的, 在
then. we. do. images is equal to because the image was initially. as you remember. here. it
这里被重新缩放. 所以从0到255在新尺度上, 即在
then. we. do. images is equal to because the image was initially. as you remember. here. it
0到-1和+1之间.
was. rescaledso from 0 to255in in a new scale that is between. 0. minus1. and plus. 1
现在我们做相反的步骤所以再次从-1到1缩放到
nowwe
现在我们做相反的步骤, 所以再次从-1到1缩放到
do. the. opposite step sore scale again from minus1 to1. into0. to255with clamp.
0到255, 使用clamp等于true, 稍后我们会看到这个
do. the. opposite step sore scale again from minus1to1. into 0. to255with clamp
0到255, 使用clamp等于true, 稍后我们会看到这个
equal true
函数. 它非常简单, 只是一个缩放函数. 我们进行排列
equal true
函数. 它非常简单, 只是一个缩放函数. 我们进行排列
:o later we will see this function it's very easy it's just a. rescaling function we
因为为了在 CPU 上保存图像, 我们希望通道维度成为最后
later we will see this function it's very easy it's just a. rescaling function we -.
permute because to save the image on the CPu we want the channel dimension to. be. the
last one
排列0, 2, 3, 1, 所以这个基本上, 会将批次
permute. 023. 1 so this one basically will take the batch size. channel. height width
大小、通道、高度、宽度转换为批次大小、高度
into
宽度、通道. 然后我们将图像移动到 CPU, 然后将其转换为
batch size. oops height width channel and then we move the. image. to the. cpu and. then
宽度、通道. 然后我们将图像移动到 CPU, 然后将其转换为
we convert it into a numpy array and then we return. the. image
numpy 数组, 然后返回图像.
we convert it into a numpy array and then we return. the. image
瞧, 让我们构建这个重新缩放方法. 那么旧尺度、旧范围
voila let'sbuild this rescale method so what is the old scale old range what. is. the
是什么, 新范围和clamp 是什么? 所以让我们定义旧的
voila let'sbuild this rescale method so what is the old scale. old range what is. the
最小值. 旧的最大值是旧范围, 新的最小值和最大值.
new :range. and the clamp so let's define the old minimum old. maximum. is. the. old. range
最小值. 旧的最大值是旧范围, 新的最小值和最大值.
new minimum and new maximum.
新范围, 减去等于旧均值. 乘以等于新最大值减去新均值.
new. range minus equal to old mean multiply equal to new max. minus new mean. divided. by e
除以旧最大值减去旧最小值.×加上等于新最小值. 我们
除以旧最大值减去旧最小值.×加上等于新最小值. 我们
:old max. minus old min xplus equal to new min we are just. rescaling so. convert
只是在重新缩放. 所以将某个在这个范围内的东西转换到这个
old max. minus old minx plus equal to new min we are just rescaling so. convert.
范围内, 如果是clamp, 那么×等于x clamp.
something that is within this range into this range and if. it's. clamp. the n. x. is. equal.
范围内, 如果是clamp, 那么x等于xclamp.
to x. clamp new min new max and then we. return x
新最小值, 新最大值, 然后我们返回×, 然后我们有时间
to x. clamp new min new max and then we. return x
新最小值, 新最大值, 然后我们返回×, 然后我们有时间
嵌入
Then we have the time embeddings.
我们在这里没有定义的方法, 这个获取时间嵌入的方法
The method. that we didn't define here, this get time embedding, this means basically.
基本上意味着取时间步, 它是一个数字, 所以是一个整数
The method. that we didn't define here, this get time embedding, this. means basically.
并将其转换为一个大小为320 的向量.
take. the time step, which is a number, so which is an. integer, and. convert it into a
并将其转换为一个大小为320的向量.
vectorofsize320.
这将会使用与我们用于转换或位置嵌入的相同系统来完成.
And. this. will be done exactly using the same system that we. use. for the transform. or..
这将会使用与我们用于转换或位置嵌入的相同系统来完成.
for the positional embeddings.
所以我们首先使用与transformer 相同的公式定义我们的余弦和正弦的
So-we. first define the frequencies of our cosines and the sines. exactly. using. the
所以我们首先使用与transformer 相同的公式定义我们的余弦和正弦的
same formula of the transformer.
频率.
same formula of the transformer.
所以如果你记得公式等于10. 001除以10. 000的某个
So. if you. remember the formula is equal to the 10, 001over. 10, 000 to the-power. of
所以如果你记得公式等于10, 001除以10, 000的某个
something, of l, Iremember correctly
所以它是10, 000的次方, 减去torch. arrange.
So it's power of 10, 000 and minus torch dotarrange.
所以我在这里提到这个公式, 以防你忘记.
So T am referring to this formula just in case you forgot.
= size of the e mo+dding vect 所以我在这里提到这个公式, 以防你意记
ncat (head, head )wo = humber of heads
Multi Head t( Q, K, V) = Concat(head, heod_)w
= number of heads • d/ h
size of the er 让我用幻灯片找到它. 我在这里谈论这个公式, 所以定义
让我用幻灯片找到它. 我在这里谈论这个公式, 所以定义
let me. find it using the slides I am talking about this formula here so. the. formula.
让我用幻灯片找到它
让我用幻灯片找到它. 我在这里谈论这个公式, 所以定义
that. defines the positional encodings here here we just use a. different dimension-of.
位置编码的公式. 这里我们只是使用了嵌入的不同维度.
the embedding
这个会产生160 个数字, 然后我们用时间步乘以它, 所以
this one will produce something that is 160 numbers and then we multiply. it with. the.
我们创建一个大小为1的形状. 所以x等于torch. tensor,
this one will produce something that is 160 numbers and then we multiply. it with. the
它是 一个单一时间步的t 类型
time. step. so we create a shape of size one sox is equal to. torch. dot tensor. which-is
它是一个单一时间步的t 类型
a single time step of ttype
取所有内容:我们增加一个维度, 所 所以我们增加一个维度.
take everything we add one dimension so we add one dimension this is. like doing.
这就像做un squeeze : 乘以频率, 然后我们用正弦和余弦乘以它
un squeeze. multiply by the frequencies and then we multiply. this. by the sines and. the
就像我们在原始transformer 中所做的那样.
un squeeze. multiply by the frequencies and then we multiply. this. by the sines and. the.
就像我们在原始transformer 中所做的那样.
cosine just like we did in the original transformer
这个会返回一个大小为100乘62的张量, 所以是320
this. one will return a tensor of size100by62sowhichis. 320. because. we are.
这个会返回 个大小为100乘62的张量, 所以是320
concatenating two tensors
因为我们连接了两个张量, 不是余弦而是×的正弦.
因为我们连接了两个张量, 不 不是余弦而是×的正弦.
:o Not. cosine but sine of x. And then I concatenated along the. last dimension.
这就是我们的时间嵌入.
And this is our time embedding
所以现在让我们回顾一下我们在这里构建的内容.
So now let'sreview what we have. built. here.
我们基本上构建了一个系统一种方法它接受提示
we built basically a system a method that takes the prompt. the unconditional prompt.
无条件提示, 也称为负提示, 提示或空字符串, 因为如果
also. called. the negative prompt the prompt or empty string because if. we. don't want
我们不想使用任何负提示, 输入图像, 所以我们想从哪张
to use any
我们不想使用任何负提示, 输入图像, 所以我们想从哪张
negative prompt the input image so what is the image we want to. start. from in. case. we
强度是我们希望在去噪图像时对输入图像给予多少关注, 或者
want. to do. an image to imi age the strength is how much attention we want. to pay-to.
强度是我们希望在去噪图像时对输入图像给予多少关注, 或者
this input image when we de noise. the image.
强度是我们希望在去噪图像时对输入图像给予多少关注, 或者
我们希望向其添加多少噪声
基本上, 我们添加的噪声越多, 输出与输入图像的相似度就
or. how. much noise we want'to add it to it basically and. the. more noise. we add. the
越低, 如果我们想做无分类器引导, 这意味着如果我们希望
less. the. output will resemble the input image the with if. we want. to. do. classifier.
越低, 如果我们想做无分类器引导, 这意味着如果我们希望
free guidance
越低, 如果我们想做无分类器引导, 这意味着如果我们希望
which means that if we want the model to output to output. one is. the.
模型输出, 一个是有提示的输出, 一个是没有提示的输出
which means that if we want the model to output to output. one is. the.
模型输出 个是有提示的输出, 一个是没有提示的输出
模型输出, 一个是有提示的输出, 一个是没有提示的输出
output with. the prompt and one without the prompt, and then we. can adjust. how. much. we
然后我们可以根据这个比例调整我们希望对提示给予多少
want'to pay attention to the prompt according to. this scale
关注
want to pay attention
然后我们定义调度器, 只有一个, DPM, 我们现在就定义它.
And. then we define the scheduler, which is only one, the. DPM, and we will. define. it.
然后我们定义调度器, 只有一个, DPM, 我们现在就定义它.
now.
以及我们想要做多少步
And how many steps we want. to. do..
cet et ])
我们做的第一件事是创建一个生成器, 它只是一个随机数
: The first thing we do is we create a generator, which. is just a. random number
stets_ Npe * (1, 4, LAr Nm_ E5 T, LAG2)
然后我们做的第二件事是, 如果我们想做无分类器引导
Then, the second thing we do is, if we want to do classifier-free guidance, as we
正如我们需要做的, 基本上, 我们需要通过单元两次:一次
need. to do, basically we need to go through the units twice, one. with. the. prompt, one
有提示, 次没有提示. 我们做的是实际上我们创建一个
without the
有提示, 一次没有提示. 我们做的是实际上我们创建一个
prompt, the thing we do is that actually we create a batch. size. of. two, one. with. the
无条件提示或负提示.
prompt and one without the prompt, or using the unconditioned prompt or. the negative
无条件提示或负提示.
prompt.
如果我们不做无分类器引导, 我们只构建一个只包含提示的
In case we. don't do the classifier-free guidance, we only build one. tensor. that only.
如果我们不做无分类器引导, 我们只构建一个只包含提示的
includes the prompt.
张量.
includes the prompt
我们做的第二件事是, 如果有输入图像, 我们加载它
The second thing we do is we load, if there is an input. image, we load it.
因此, 我们不是从随机噪声开始, 而是从一个我们根据定义
So instead. of starting from random noise, we start from an image. to which. we add. the
因此, 我们不是从随机噪声开始, 而是从一个我们根据定义
noise according'to the strength we have defined.
然后, 对于由采样器定义的步数, 实际上是由我们在这里
Then. for. the number of steps defined by the sampler, which. are actually. defined. by
循环
:o the number of inference steps we have defined here, we. do a. loop, a. for loop.
Let ets )
对于每个for 循环单元将预测一些噪声, 调度器将移除
that. for each for loop the unit will predict some noise and. the scheduler. will. remove
对于每个for 循环, 单元将预测一些噪声, 调度器将移除
这些噪声并给出一个新的潜在值.
that for each for loop the unit will pn edict some noise and the scheduler wil
然后这个新的潜在值再次被输入到单元中, 该单元将预测一些
噪声, 我们根据调度器移除这些噪声
Then f his new latent is fed again to the unit which will pn edicts
然后我们再次预测一些噪声并移除一些噪声.
我们唯一需要理解的是现在我们如何从图像中移除噪声
因为我们知道单元被训练来预测噪声, 但我们实际上如何移除
它? 这是调度器的工作. 所以现在我们需要在这里构建这个
它? 这是调度器的工作. 所以现在我们需要在这里构建这个
c remove it and this is the job of the scheduler so now we. need to go build this
scheduler
所以让我们去构建它. 让我们开始构建我们的 DD PM 调度器.
so. let'sgo build it let's start building our um ddpm. scheduler. so ddpm. py oops. i
所以dd pm py. 哦, 我忘了把它放在文件夹里. 让我回顾一件事.
:so. let's go build it let's start building our um ddpm scheduler. sodd pm. py oops. i
是的, 这是错误的. 好吧.
所以导入torch, 导入min p y, 让我们创建类tdpm-sampler.
So. import torch, import min py, and let'screate the class. td pm-sampler
好的, 我没有称之为调度器, 因为我不想让你与我们将稍后
Okay l didn't call it scheduler because I don't want you to. be-confused. with. the
好的, 我没有称之为调度器, 因为我不想让你与我们将稍后
beta schedule, which we will define later.
所以在这里我称之为调度器.
So I call it scheduler here.
哦, 为什么我打开了
Oops :
哦, 为什么我打开了这一 个?
Oops, why I opened this one?
我在这里称之为调度器, 但实际上我指的是采样器, 因为
现在我们将定义beta 调度. beta 调度是什么?
它表示每个时间步的噪声量. 然后有一个被称为调度器或
采样器的东西. 从现在开始我将称之为采样器, 所以这里的
调度器实际上指的是采样器.
对于混淆我感到抱歉. 当视频发布时, 我将更新幻灯片.
sampler I'm sorry
对于混淆我感到抱歉. 当视频发布时, 我将更新幻灯片.
for the confusion I will update the slides when the video is out
那么训练步数是多少?
So how much were the training steps?
即1000.
Whichis1000
beta 是, 好的现在我定义了两个常数, 稍后我会定义它们
The. beta is, okay now I define two constants and later I define. them, what are. they.
它们是什么以及它们来自哪里.
and where they come from.
0, 85和betaend是浮点数0. 0120.
0, 85 and beta end is a floating point of. 0. 0120.
好的, 参数beta start 和beta end, 基本上如果你去看论文
Okay, the parameter beta start and beta end, basically. if. you go. to the-paper.
(12 )
Now we discuss our choices in po(x-1x)= N(x1:μ(x. t).∑g(x,)) for 1 <1≤ T. First,
Now we discuss our choices in po(x-1x)= N(x1:μ(x. t).∑a(x,)) for 1 <t ≤ T. First,
If we look at the forward process y we can see that the-forward process isi the process
pbstetiorg ha s o 3earnahle parare ters. sd / Is a constant dunn g train it ig and can be ie none d 如果我们看正向过程, 我们可以看到正向过程是使图像更加
B[ log p(xo)] ≤ B.
-log
L (3)
The forward process varige
t≥1
33] or held constant as
B[logp(xo)] ≤ B,
q(x|x1)]
L(3)
The forward process variances 3; can be leamed by reparameterization [33] or held constant as
张噪声较少的图像,
B[ logp(xo)] ≤ B
log
Po( Xo:r) 像?
q(x[x1)]
L (3)
r≥1
B[ logp(xo)] ≤ B
log
q(x1:r|xo)]
Po( Xo:r)
L (3)
The forward process variances 8 can be leamed by reparameterization [33] or held constant as
r≥1
根据这
Po( Xo:r) 系列高斯分布
p( Xt1|x)
ssian
B[—logpw(xo)] ≤ E
p( Xo:r)
logp[x)
L(3
B[logp(xo)] ≤ B.
-log
po( Xo. r)]
q(x[x1) ]
L(3)
The forward process variances 3 can be learmed by reparameterization [33] or held constant as
而在潜在空间中
Po( Xo: T) 个'beta start
= L (3)
and as
而在潜在空间中, 在稳定扩散中他们使用了一个beta start.
and as. in. the latent in the stable diffusion they use a. beta. start so. the first value.
所以beta的第一个值是0. 0085, 最后一个方差, 也就是这个
and as. in. the latent in the stable diffusion they use a beta start so. the first value
将图像完全转化为噪声的beta, 等于0. 0120.
:of. beta. is0. oo85 and the last variance so this the beta. that will turn the. image
log
Po( Xo:r)
-logp(xr)-
= L (3)
将图像完全转化为噪声的beta, 等于0. 0120.
:complete noise is equal to 0. 0120 it'sa choice made. by. the authors and-and the
这是作者做出的选择, 它是线性的. 我们将使用线性调度,
complete noise is equal to 0. 0120it's a choice made. by. the authors and and. the.
这是作者做出的选择, 它是线性的. 我们将使用线性调度.
这是作者做出的选择, 它是线性的. 我们将使用线性调度,
A it's. a linear we will use a linear schedule actually there are. other schedules. which.
实际上还有其他调度, 例如余弦调度等.
it's. a linear we will use a linear schedule actually there are. other schedules which.
但我们将使用线性调度, 我们需要定义这个beta 调度, 它
are. for example the cosine schedule etc but we will be using. the linear one and. we
但我们将使用线性调度, 我们需要定义这个beta 调度, 它
need to
但我们将使用线性调度, 我们需要定义这个beta 调度, 它
A define. this. beta schedule which is actually 10o0 numbers between. beta start and. beta
实际上是beta start 和betaend之间的10o0个数字. 所以
define. this. beta schedule which is actually 1oo0 numbers between. beta start and. beta
实际上是beta start 和betaend之间的1000个数字. 所以
end so let'sdo it
所以这是使用线性空间定义的, 其中起始数字是beta start,
so. this. is. defined using the linear space where the starting-number is. beta start
所以beta start 的平方根, 因为在稳定扩散中他们是这样定义的
:actually. to the square root of beta start so square root of beta start because this.
所以beta start 的平方根, 因为在稳定扩散中他们是这样定义的.
is how they
所以beta start 的平方根, 因为在稳定扩散中他们是这样定义的.
:define. it. in the stable diffusion if you check the official repository. they will also
如果你查看它方仓库, 他们也会有这些数字并以完全相同的
define. it. in the stable diffusion if you check the official repository they will also
方式定义:0. 5, 然后是训练步数. 所以我们想把这个线性空间
方式定义:0. 5, 然后是训练步数. 所以我们想把这个线性空间
19. 5. then. the number of training steps so in how many pieces we. want. to. divide. this
分成多少份?
20. 5. then. the number of training steps so in how many pieces. we. want. to. divide. this.
分成多少份?
:linear space beta end and then the type is to rch. float32 Ithink and. then to. the.
betaend, 然后类型是torchfloat, 32我想, 然后是平方, 因为
linear. space beta end and then the type is to rch. float32. I. think and. then to. the.
beta end, 然后类型是torchfloat, 32我想, 然后是平方, 因为
powerof2
他们把它分成10oo, 然后是平方, 这是在hugging face 的扩散器
他们把它分成10o0, 然后是平方, 这是在hugging face 的扩散器
because they do they divide it into 1000 and then. to. the. powerof. 2..
他们把它分成10oo, 然后是平方, 这是在hugging face 的扩散器
他们把它分成10o0, 然后是平方, 这是在huggingface的扩散器
w:and this. is in the diffusers libraries from hugging phase i think. this is. called. the
库中. 我认为这被称为缩放线性调度.
and this. is in the diffusers libraries from hugging phase i think. this is. called the
现在我们需要定义其他常量, 这些常量是我们的前向和后向
:scaled. linear schedule now we need to define other constants. that are needed. for-our
现在我们需要定义其他常量, 这些常量是我们的前向和后向
(3 )
arbitrary time step in closed fom : using the notation a
1 3 and
Ia, we have 过程所需的. 所以我们的前向过程依赖于这个beta 调度
and our b acls ward process se our forward process depends on this beta schedule but
但实味上这只是单步的
(4 )
actually this is only for the single ste
所以如果你想从原始图像通过一步或多步添加更多噪声, 我们
actually this is only for the single step so if you want to go from for ew ample the
arbitrary time step t in closed form : using the notation o
1 β and
= I α, we have
(4)
byone step forward of moreno is
q(x|xo) = N(x;√axo,(1
á) I)
(4)
arbitrary time step t in closed fom : using the notation og
1
β; and a := II, we have
图像到任何时间步在0到1000之间的噪声图像, 使用
the original image to any no is ified version of the image at any time step between 0
这里的这个, 它依赖于你在这里看到的alpha bar. 所以alpha the original image to any no is ified version of the image at any time step between 0
bar 的平方根和方差也依赖于这个alpha bar. 什么是alpha bar?
using this one here which depends on alpha bar that you can see here so the square
alpha bar是从1到t的alpha的乘积.
bar alpha bar is the product of alpha
所以如果我们想从时间步0 (即无噪声的图像)到时间步
going from one up to t so if we are for cw ample we want to go from the time step zero
10(即带有一些噪声的图像), 记住时间步1000
which is the image without any noise to the time step 10 which is the image with some
意味着它完全是噪声
And remember that time step 1oo0 means that it's only noise.
所以我们想达到时间步10, 这意味着我们需要计算 AS1
So we want to go to time step 10, which means that we need to calculate this AS of 1
AS2, AS3. AS4. 直到 AS10
然后把它们相乘.
And we multiply them together.
这就是乘积.
This is the product or y.
而这个 A, 这个alpha 是什么?
And this A, what is this alpha?
这个alpha 实际上是1减去beta.
This alpha actually is 1 minus beta.
所以让我们先计算这些alpha.
So let's calculate these alphas first.
所以alpha 实际上是1减去beta.
So alpha is actually1 minus beta
beta, 自betas, 所以它变成浮点数. 然后我们需要计算从1
beta. self. betasso it becomes floating and then we need to. calculate the-product of..
beta, 自betas, 所以它变成浮点数. 然后我们需要计算从1
到t 的这些alpha 的乘积, 这很容易用py torch 完成.
到t 的这些alpha 的乘积, 这很容易用pytorch完成.
v:this alphasfrom 1tot and this is easily done with py torch we. precompute. them
我们预先计算它们. 基本上这也是com prod self alphas.
this alphas from 1 totand this is easily done with py torch. we. precompute. them.
我们预先计算它们. 基本上这也是com prod self alphas.
basically this is also com prod self. alphas
这将基本上创建一个数组, 其中第一个元素是第一个alpha.
o This. will create basically an array where the first element is. the first. alpha.
所以alpha, 例如, 0.
So alpha, for example, 0.
第二个元素是alpha 0乘以alpha1.
The second element is alpha 0multipliedby. alpha 1
第三个元素是alpha 0乘以alpha1乘以alpha2, 等等.
The third element is alpha 0multiplied by alpha 1multiplied. byalpha 2, etc
所以我们说这是一个累积乘积.
So it's a cumulative product, we say
然后我们创建一个表示数字1 的张量, 稍后我们会用到它.
: Then. we. create one tensor that represents the number 1 and. later we will use. it.
然后1. 0. 好的, 我们保存生成器, 保存训练步数, 然后
and so1. 0. okaywe save the generator we save the number of. training steps and. then
时间步基本上因为我们想要逆转噪声, 我们想要去除噪声.
we :create. the time step schedule the time step basically because. we want. to. reverse.
时间步基本上因为我们想要逆转噪声, 我们想要去除噪声
我们将从更多噪声到更少 操声, 所以 我们将人
1000 到
最初. 所以假设时间步等于torch from. 我们逆转这个, 所以
initially. so let's say time steps is equal to torch from we reverse this. so. this. is
这是从0到1000.
:oinitially. so let'ssay time steps is equal to torch from we reverse this. so. this. is
这是从0到1000.
from 0to 1000but actually we want 1000. to. 0
但实际上我们想要1000到0, 这是我们的初始计划.
from 0to 1000 but actually we want 1000. to. 0.
但实际上我们想要1000到0, 这是我们的初始计划.
如果我们想做1000 步, 但后来一 一因为这里我们实际上
and. this. is our initial schedule in case we want to do1000. step. but. later. because.
指定了我们想要做多少推理步骤一一 我们会改变这里的时间
here. we actually specify how many inference steps we want. to do we. will. change. this.
指定了我们想要做多少推理步骤一 我们会改变这里的时间
time steps
指定了我们想要做多少推理步骤一一我们会改变这里的时间
here. so if the user later specifies less than 1000 we will. change it so, let's. do
步. 所以如果用户后来指定少于1000步, 我们会改变它.
let's. create the method that will change this time steps based. on how many actual.
所以让我们创建一个方法, 根据我们实际想要执行的步骤
let's. create the method that will change this time steps based. on how many actual.
如我之前所说, 我们通常执行50步, 这实际上也是他们
as Isaid before we usually perform 50 which is also actually. the one. they use
通常在hugging face 库中使用的步数. 让我们保存这个值
easl said before we usually perform 50 which is also actually-the one. they use
因为稍后我们会用到它.
w :normally. for. for. example in hugging face library let's save. this. value because. we.
因为稍后我们会用到它.
will need it later
现在, 如果我们有一个数字, 例如, 我们从1000开始
Now, if we. have a number, for example, we go from 10o0, actually this is. not. from. 0
实际上这不是从0到1000, 而是从0到1000减1,
Now, if we. have a number, for example, we go from 10o0, actually this is. not. from. 0
实际上这不是从0到1000, 而是从0到1000减1,
to. 1000, but it's from 0to1000minus1, because this is excluded, so it will be
因为这是排除的. 所以它将是999, 998, 997, 996, 等等
to. 1000, but it's from 0 to1000 minus1, because this is excluded, so it will be
因为这是排除的. 所以它将是999, 998, 997, 996, 等等
from 999, 998, 997, 996, etc., up. to. 0.
直到0.
from 999, 998, 997, 996, etc., up to. 0.
所以我们有1000个数字, 但我们不想要1000个, 我们
: Sowe. have. 1000 numbers but we don't want 1000 numbers we. want. less we-want. 50. of
开始, 然后是999减20, 然后是999减40, 等等
them. so what we do is basically we space them every 20so. we. start. with. 999. then. 999.
开始, 然后是999减20, 然后是999减40, 等等
minus20
开始, 然后是999减20, 然后是999减40, 等等
then999. minus40etc etc until we arrive to 0 but in total. here. will be 1000 steps.
直到我们到达0. 但在这里总共会有1000步, 而在这里会
then999. minus40etc etc until we arrive to 0 but in total. here. will be 1000 steps
直到我们到达0. 但在这里总共会有1000步, 而在这里会
andherewillbe50steps
有50步.
and here will be50steps
为什么是减20?
Why minus20?
因为20是1000除以50, 如果我没记错的话.
Because 20is1000 divided by50, if1'mnot mistaken.
所以这正是我们要做的.
So this is exactly what we are going to. do.
所以我们计算步长比率, 即self dot num training step 除以我们
So. we calculate the step ratio, which is self dot num training step. divided by. how
实际想要的步数.
So. we calculate the step ratio, which is self dot num training step. divided by how
然后我们根据我们实际想要执行的步数重新定义时间步
A And. we. redefine the time steps according to how many we. actually. want. to. make..
零步推理步骤乘以这个步长比率并四舍五入. 我们像之前一样
zero num inference steps multiplied by this step ratio and round it we. reverse. it
反转它, 因为这是从零开始的. 所以这实际上意味着0
zero num inference steps multiplied by this step ratio and. round. it we. reverse. it.
然后是20, 然后是40, 然后是60, 等等, 直到我们达到999.
just like. before because this is from zero so this is actually. means. zero. the n. 20
然后我们反转它并复制为类型mpdot64, 所以是一个长
then40then60etcuntil we reach 999 then we reverse. it and. copy
然后我们反转它并复制为类型mpdot64, 所以是一个长
然后我们反转它并复制为类型mpdot64, 月 所以是一个长
as typemp-dot 64so a longone and then we define as tensor. now. the code. looks. very
整型, 然后我们定义为张量. 现在代码看起来彼此非常
as typemp. dot 64so a longone and then we define as tensor now. the code. looks. very
不同, 因为实际上我一直在从多个来源复制代码.
different from each other because actually i have been copying the code from multiple
不同, 因为实际上我一直在从多个来源复制代码.
sources
也许其中一个, 我想, 我是从hugging face 库复制的, 所以
may be one of them i think i copied from the hugging face. library. so i didn't change.
我没改它. 我保留了原始版本.
w may be one of them i think i copied from the hugging face. library so i didn't change.
我没改它. 我保留了原始版本.
it ii kept it to the original one
好的, 但这个想法是我之前展示给你的.
Ok, but the idea is the one I showed you. before.
现在我们设置了我们想要的确切时间步数, 并且我们像这样
Now we. set. the exact number of time steps we want and were define this time steps. array
重新定义了这个时间步数组
like this.
让我们定义下一个方法, 它基本上告诉我们:让我们定义如何
let's. define the next method which basically tells us let'sdefine. the method on. how
给某物添加噪声的方法. 所以想象一下我们有一张图片.
let's. define the next method which basically tells us let's define. the method on. how.
给某物添加噪声的方法. 所以想象一下我们有一张图片.
如你所记, 要做图像到图像的转换 我们需要
to add noise to something so imagine
如你所记, 要做图像到图像的转换, 我们需要给这个潜在
变量添加噪声. 我们如何给某物添加噪声? 嗯, 我们需要
应用论文中定义的公式. 所以让我们来看看论文这里.
应用论文中定义的公式. 所以让我们来看看论文这里.
apply the formula as defined in the paper so let's go in the paper here we need to
这意味着, 给定这张图片, 你需要, 我想得到这张图片在
formula here and that's it this is it means that given this image
时间步t 的噪声化版本, 这意味着我需要从中取样
you need i want to go to the noise no is ified version of this image at time step t
我们需要从这个高斯分布中取样, 但我们没有.
嗯, 好吧, 让我们构建它, 我们将应用与变分自编码器
don't uh
相同的技巧.
okay let's build it and we will apply the same trick that we did for the variational
如你所记, 在变分自编码器中, 我实际上已经展示了我们
auto encoder. as you remember in the variational auto encoder i actually. already showed
如你所记, 在变分自编码器中, 我实际上已经展示了我们
how we
如你所记, 在变分自编码器中, 我实际上已经展示了我们
sample. from. a distribution of which we know the mean and. the variance here we. will. do.
这里我们将做同样的事情, 但当然, 我们需要构建均值和
sample. from. a distribution of which we know the mean and. the. variance here we. will. do.
这里我们将做同样的事情, 但当然, 我们需要构建均值和
the same here
方差.
But of course we need to build the mean and. the variance.
这个分布的均值是什么?
What is the mean of this. distribution?
所以我们需要构建均值和方差, 然后从中采样.
So we need to build the mean and the variance and then we sample from this
所以我们需要构建均值和方差, 然后从中采样.
So we
所以让我们来做吧.
So let's do it.
DD PM.
DDPM.
所以我们取原始样本.
So we take the original samples.
这是一个浮点张量, 然后是时间步. 所以这实际上是时间
:o which is a float tensor and then the time steps so this is. actually time step. not
它指示我们想在哪个时间步添加噪声, 因为你可以在时间步
time steps. it. indicates at what time step we want to add the noise. because-you. can. add.
1、2、3、4, 一直到1000添加噪声, 并且随着
the noise at
1、2、3、4, 一直到1000添加噪声, 并且随着
:otime step 1, 2, 3, 4up to1000 and with each level the noise. increases. so. the
每个时间步, 噪声增加. 所以在时间步1的噪声化版本
timestep1, 2, 3, 4upto1000 and with each level the noise. increases. so. the
不会那么嘈杂, 但在时间步1000将会是完全的噪声
:noisifiedversion at times tep1 will be not so noisy but at. times tep 1000. will. be
不会那么嘈杂, 但在时间步1000 将会是完全的噪声
complete noise
这返回一个浮点张量.
This returns a float tensor.
好的, 我们先计算一下, 让我看看我们需要先计算什么.
Okay, let'scalculate first, let me check what we need. to calculate first
好的, 我们先计算一下, 让我看看我们需要先计算什么.
好的, 我们先计算一下, 让我看看我们需要先计算什么.
Okay, let's calculate first, let me check what we need to calculate first.
我们可以先计算均值, 然后再计算方羞.
We can calculate first the mean and chen the variance.
所以为了计算均值, 我们需要这个alpha 累积乘积, 也就是
So. to. calculate the mean we need this alpha cum prod, so. the. cumulative product. of
所以为了计算均值, 我们需要这个alpha 累积乘积, 也就是
the alpha, which stands for alpha bar.
alpha bar.
the alpha, which stands for alpha bar.
所以alpha bar, 如你所见, 是所有alpha 的累积乘积, 其中
每个alpha 是1减去beta
So the alpha bar, as you can seg is the cumulative product of all the alphas, which
所以我们取这个alpha bar, 我们称之为alpha 累积乘积, 它
So we take. this alpha bar, which we will call alpha cum prod, so. it's already defined
已经在这里定义了.
here.
Alpha 累积乘积是self. one. two. device.
Alpha cum prod is self dot one dot two. device.
我们将其移动到相同的设备, 因为我们稍后需要将其与它
we move. it. to. the. same device because we need to later combine. it with. it. and. of. the
结合, 并且类型相同. 这也是一个我们移动到其他张量相同
we :move. it. to the same device because we need to later combine. it. with. it. and. of. the.
设备的张量.
same type. this is a tensor that we also move to the same device. of. the other tensor.
现在我们需要计算alpha bar 的平方根. 所以让我们来做吧.
now. we. need to calculate the square root of alpha bar so. let's. do it square root of.
现在我们需要计算alpha bar 的平方根. 所以让我们来做吧.
现在我们需要计算alpha bar 的平方根. 所以让我们来做吧.
now. we. need to calculate the square root of alpha bar so. let's. do it square root of
alpha 累积乘积或alpha 乘积的平方根是alpha 累积乘积在时间步
now. we. need to calculate the square root of alpha bar so. let's. do it. square root. of
t的0. 5次方. 为什么是0. 5次方?
alpha comprod oralphaprodis alpha com prod at. the. time step. t
t的0. 5次方. 为什么是0. 5次方?
t的0. 5次方. 为什么是0. 5次方?
to the power of 0. 5whyto thepowerof. 0. 5.
因为一个数的0. 5次方意味着对这个数取平方根, 因为它是
because. having a number to the power of o. 5means doing the square root. of. the. number
1/2次方, 变成了平方根. 然后我们展平这个数组
because it's the square root of one half which becomes the square sorry to. the power.
1/2次方, 变成了平方根. 然后我们展平这个数组,
of one
1/2次方, 变成了平方根. 然后我们展平这个数组
halfwhich becomes the square root and then we flatten this array
然后;基本上, 因为我们需要结合这个alpha 累积乘积
And then basically, because we need to combine this alpha com prod, which. doesn't. have
它没有维度, 它只有一个维度, 就是它本身. 但我们需要
And then basically, because we need to combine this alpha. com prod, which. doesnt. have
将其与潜在变量结合, 我们需要添加一些维度.
dimensions, it only has one dimension, which is the. number. itself, but we need. to
所以一个技巧是不断用un squeeze 添加维度, 直到你拥有相同数量
So one. trick. is to just keep adding dimensions with un squeeze until you have the. same
所以一个技巧是不断用un squeeze 添加维度, 直到你拥有相同数量
number of dimensions.
的维度.
number of dimensions.
所以直到n 的平方, 形状小于, 大部分代码我取自 Hugging So until then of s squared, the shape is less. than
所以直到n 的平方, 形状小于, 大部分代码我取自 Hugging
所以直到n 的平方, 形状小于, 大部分代码我取自 Hugging Most. of this code I have taken from the Hugging Face. libraries samplers
Face 库的采样器.
Most. of this code I have taken from the Hugging Face. libraries samplers.
所以我们保持维度, 直到这个张量和这个张量具有相同的
v So. we keep the dimensions until this one and this tensor and this. tensor have. the
所以我们保持维度, 直到这个张量和这个张量具有相同的
same dimensions.
这是因为否则我们在将它们相乘时无法进行广播.
This :is. because otherwise we can not do broadcasting when. we multiply. them. together
我们需要计算这个公式的另一部分是这里:1减去alphabar.
the. other. thing that we need to calculate this formula is. this part. here. one. minus.
所以让我们来做吧.
v :the. other. thing that we need to calculate this formula is. this part. here. one. minus
是1减去alpha累积乘积. 在时间步t的0. 5
alpha bar. so let's do it so s square to f one minus alpha prod. as. the name-implies. is
是1减去alpha累积乘积. 在时间步t的0. 5
alpha. cum. prod at the time step t to the power of 0. 5why. o. 5. because we. dont. want..
因为我们不想要方差, 我们想要标准差, 就像我们在变分自
alpha. cum. prod at the time step t to the power of 0. 5why. 0. 5. because we. dont. want
因为我们不想要方差, 我们想要标准差, 就像我们在变分自
the. variance we want the standard deviation just like. we. did with. the.
编码器的编码器中所做的那样.
the. variance we want the standard deviation just. like. we. did with. the
编码器的编码器中所做的那样.
我们想要标准差, 因为, 如你所记得的, 如果你有一个
encoder of. the variational auto encoder we want the standard deviation because. as you
n01, 你想要转换成具有给定均值和方差的n, 公式是:x
and the
n01, 你想要转换成具有给定均值和方差的n, 公式是:x
variance. the formula is x is equal to mean plus the standard. deviation multiplied by..
等于均值加上标准差乘以n01. 让我们回到正题. 所以这是
variance. the formula is x is equal to mean plus the standard deviation multiplied by
标准差, 我们也展平了这个.
the. n01. let'sgo back so this is the standard deviation deviation and. we. also. flatten
标准差, 我们也展平了这个.
this one
然后我们再次不断添加维度, 直到它们具有相同的维度
and. then again we keep adding the dimensions until they. have. the same. dimension.
否则我们无法将它们相乘或相加, 使用un squeeze.
otherwise we. can not. multiply them together or sum them together. un squeeze so. we. keep
所以我们不断添加维度. 现在, 如你所记得的, 我们的
otherwise we. can not multiply them together or sum them together. un squeeze so. we. keep
所以我们不断添加维度. 现在, 如你所记得的, 我们的
adding
方法应该给图像添加噪声. 所以我们需要添加噪声意味着我们
dimensions. now as you remember our method should add noise. to. an image. so. we. need. to
需要从 N01中采样一些噪声. 所以我们需要使用我们有的
add. noise means we need to sample some noise so we need to sample some noise. from thee
需要从 N01中采样一些噪声. 所以我们需要使用我们有的
N01using this generator that we have.
这个生成器来采样一些噪声.
N01 using this generator that we have.
我想我的猫今天对我非常生气, 因为我没有和他玩够, 所以
think my. cat is. very angry today with me because i didn't. play. with. him enough so..
如果你们不介意的话, 我稍后需要和他玩. 我想我们就要
think my. cat is very angry today with me because i didn't. play. with him enough so.
我们很快就会完成. 所以让我们用我们计算出的噪声、均值
later. if you guys excuse me i need to later play with him i think. we-will be. done-we
和方差, 根据这里的公式来获取噪声样本.
variance. that we have calculated according exactly to. this. formula here so. we do. the.
所以我们实际上做的是均值, 不, 均值是这个乘以x0.
a:variance. that we have calculated according exactly to this formula here so. we do. the
所以我们实际上做的是均值, 不, 均值是这个乘以x0.
mean
所以我们实际上做的是均值, 不, 均值是这个乘以x0.
actually no the mean is this one multiplied by x0 so the mean is this one multiplied
所以均值是这个:1乘以x0是均值.
所以我们需要取这个alpha 累积乘积的平方根乘以x0, 这将是
所以我们需要取这个alpha 累积乘积的平方根乘以x0, 这将是
this will. be the mean so the mean is square root of alpha prod. multiplied by. the.
所以输入图像或我们想要加噪的任何东西, 加上标准差, 即
original latent sox0so the input image or whatever we want. to. noi sify
这个的平方根乘以从 N01 采样的样本.
:plus. the. standard deviation which is the square root of. this one multiplied by. a
这个的平方根乘以从 N01采样的样本.
sample from the No1 so thenoise and this is how we no sify. an image this. is how. we
所以噪声, 这就是我们如何给图像加噪, 这就是我们如何给
sample from the No1 so the noise and this is how we no sify. an image this. is how. we
所以噪声, 这就是我们如何给图像加噪, 这就是我们如何给
add noise to an
图像添加噪声. 所以这个, 让我写下来.
add noise to an
所以这一切都是根据 DDM论文的方程4以及这个来的
image so. this one let me write it down so all of this is according. according. to. the.
所以这一切都是根据 DDM论文的方程4以及这个来的.
equation4ofthe DDMpaper
所以这一切都是根据 DDM论文的方程4以及这个来的.
and also. according to this okay now that we know how. to add. noise we. need. to
好的, 现在我们知道如何添加噪声, 我们需要理解如何去除
and also. according to this okay now that we know how. to. add. noise we. need. to
所以, 如你所记得的, 让我们再回顾一下这里. 想象一下
understand how to remove noise so as you remember. let's. review. again
所以, 如你所记得的, 让我们再回顾一下这里. 想象一下
我们在做文本到文本, 或文本到图像, 或图像到图像, 这
并不重要.
重点是, 如你所记得的, 我们的单元被训练来仅预测噪声
量, 给定带有噪声的潜在值, 给定提示和添加噪声的时间步
所以我们做的是, 我们从这个单元得到了预测的噪声. 我们
需要去除这个噪声.
所以单元会预测噪声, 但我们需要某种方法来去除噪声以得到
下一个潜在值. 我的意思是一一你可以看到这里的反向
The forward process variances 5; can be learmed by repararmeterization [33] or held constant as hyperparameters. and expressiveness of the reverse process is ensured in part by the choice of next latent what lime an by this is your tan see this reverse process here so the
21. 1 Forward process and L
posterior q has no learnable paraimeterx, so 7. y is a constant during traning ani can be ignored.
. Rrp 过程所以度间过程在这里定义
sf he
The forward process variances 5; can be leamed by reparameterization [33] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of We want to go from wt. sosohrethingmorerioisy. to something less noisy based on the
Ef cient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L ) as:
so the formula we should be looking at is actually here so her g here because we have
Now we discuss our choices in p(x-1x.)= N(x1:μg(x, t).∑(x. t)) for 1<t≤ T. First,
so the formula we should be doo king lat is actually here so here here because we have
second is optinal for xa deterministically set to one point. These are the rwo extr erne choices so the formula we should be looking at isactuatlyheresohiere here-be cad se we have
that optimizing a n objective resembling denoising score matching is equivalent to using variational
inference to fit the finite-time marginal of a sampling chain rese mbling Langevin dynamies. 所以我们应该看的公式实际上在这里所以这里因为我们
ewe have
所以现在让我们构建这部分, 在构建的过程中我也会告诉你我
so. let's. build this part now and while building it I will also. tell-you which. formula
在每一步引用的是哪个公式, 这样你也可以跟着论文走
so. let's. build this part now and while building it I will also. tell-you which. formula.
所以现在让我们构建方法.
I'm. referring to at each step so you can also follow the paper so. now let's. build. the.
所以现在让我们构建方法.
method
让我们调用步骤方法, 给定添加噪声的时间步长, 或者我们
let's. call step method that given the time step at which the. noise was added or. we.
我们也可以跳过一 不是, 我们认为它被添加, 但我们
think. it was added. because. when we do the reverse process we. can also. skip it's. not
我们也可以跳过 不是, 我们认为它被添加, 但我们
we think it
可以跳过一些时间步长. 所以我们需要告诉他应该在哪个时间
was. added but. we. can. skip some time steps so we need to. tell. him. what is. the. time.
可以跳过一些时间步长. 所以我们需要告诉他应该在哪个时间
step at which it should remove the noise
步长去除噪声.
step at which it should remove the noise
Latent s. 所以你知道, 单元与 Latents 一起工作, 所以与这里的
The. Latent s, so as you know, the unit works with Latent s, sowith the se Zs here, so.
Latent s. 所以你知道, 单元与 Latents 一起工作, 所以与这里的
这些 Z s一超, 所以这个 Z、它持续去噪. 所以 La tents.
这些 Z s一起, 所以这个 Z, 它持续去噪. 所以 La tents.
this Z, andit keeps denoising, so the Latent s..
然后, 什么是模型输出?
And then, what is the Model Output?
所以单元的预测噪声
So the predicted noise of the unit.
所以模型输出是预测的噪声
So the Model Output is the predicted noise.
Torch 点张量
Torch dot tensor.
这个模型输出对应于这个epsilon θ的xtt.
This model output corresponds to this epsilon theta. of xtt..
that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynami es.
om这个模型输出对应于这介inepsilon e. 的xtguy
that optimizing an objective resembling denoising score matching is c qui valent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.
To summarize we can train the reverse process mean function approxima tor μto predictor by modifying its param er zation we can train t top red t. There is also the possibilty f predic ng
that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics. 所以这是在时间步t的预测噪声这介潜伏是我们的·x,
所以这是在时间步t的预测噪声. 这个潜伏是我们的xt
So. this. is. the predicted noise at time step t this latency is. our xt and What else we
所以这是在时间步的预测噪声这介潜伏是我们的对,
inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.
else we
我们还需要什么? 我们有a lpha, 我们有beta, 我们都有.
So. this. is. the predicted noise at time step t this latency is. our xt and what else we.
that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynami es. 我们还需要件么? 我们有alpha pu 我们有beta. 我们都有.
我们还需要什么? 我们有a lpha, 我们有beta, 我们都有.
need the-alpha we have the beta we have we have everything Okay, let'sgo so. tis.
我们还需要什么? 我们有a lpha我们有beta我们都有
inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.
os otis
好的, 我们走. 所以t 等于时间步.
need the-alpha we have the beta we have we have everything Okay, let's. go. so. t. is.
好的, 我们走. 所以t 等于时间步.
equal to time step
个t 等于self 点. 获取前一 一个时间步t. 这是
w:the previous t is equal to self dot get previous time step. t. this. is a function. that
构建它, 实际上我们现在就可以构建它. 非常简单
given this time step calculates the previous one later we. will. build. it actually we
构建它 实际上我们现在就可以构建它. 非常简单.
can build it now it's very simple ok
好的, 获取前
self 时间步, 这是 个整数.
can build it now it's very simple ok
好的, 获取前个时间步self 时间步这是一个整数.
get previous time step self time step which is an integer we return another integer..
我们返回另一 一个整数前 个时间步等于时间步减去self previous. time step is equal'to the time step minus self minus. basically this quantity
减去. 基本上这里的数量:步长比率. 所以self 点num.
here step ratio so self dot num training steps
训练步数除以self num ln ference Steps. 返回前 个t
here step ratio so self dot num training steps
训练步数除以self num ln ference Steps. 返回前 个t
训练步数除以self num ln ference Steps 返回前- 个t divided by self. num inference Steps return previous.
这个基本上会返回, 例如, 给定数字999, 它会返回999
: This one will return basically, given for example the number 999, it will. return
减去20, 因为时间步, 例如初始时间步, 将是:假设它是
: This one will return basically, given for example the number 999, it will. return
1000, 我们正在进行的训练步数是1000除以我们正在进行的
number. 999minus20, because the time steps, for example the initial. time. step will
1000, 我们正在进行的训练步数是1000除以我们正在进行的
suppose
1000, 我们正在进行的训练步数是1000除以我们正在进行的
it's1000, the training steps we are doing is 1000 divided by the number. of. inference
推理步数, 是50, 所以这意味着1000减去20, 因为1000
it's1ooo, the training steps we are doing is 1000 divided by the number. of inference
除以50是20
:step, which. we will be doing is 50
22. minus20because. 1000. divided
除以50是20
by50is20
所以它会返回980. 当我们给它980作为输入时, 它会返回
soit. will. return 980whenwegivehim980 as input hewitt return :960so. what is. the
23. 所以我们在for 循环中将要做的下一步是什么, 或者
:so it. will return 980whenwegivehim980 as input hewitt return :960so. what is. the
去噪的前 一步是什么?
next step. that we will be doing
去噪的前一 步是什么?
denoising
所以我们从时间步1000的图像噪声到时间步980的图像
sowe are going from the image noise at the time step 1ooo. to an image noise at. the
噪声, 例如. 这就是前一步的意义
然后我们检索一些数据. 稍后我们会用到它. 所以alpha pod then we. retrieve some data tater we will use it so alpha pod t is. equal. to. self. dot
t 等于self 点alpha. 现在, 如果你不理解, 别担心
then we. retrieve some data later we will use it so alpha pod t is. equal. to. self. dot.
我只是收集一些我们需要计算公式的数据, 然后我会告诉你
alpha for now if you don't understand don't worry because. later i will write i will.
我只是收集一些我们需要计算公式的数据, 然后我会告诉你
ust
我只是收集一些我们需要计算公式的数据, 然后我会告诉你
collect. some data that we need to calculate a formula. and then i will. tell you
如果我们没有任何前 步 那么我们就不知道要返回哪个alpha If we. dont. have any previous step, then we don't know which alpha to return, so. we
所以我们只返回1
justreturn1.
实际上有一篇来自 Byte Dance 的论文 我认为, 抱怨这种方法不
And. actually there is a paper that came out I think from Byte Dance that was.
正确, 因为最后的时间步没有信噪比等于零. 但好吧
complaining that this method of doing is not correct. because. the. last. time step
正确, 因为最后的时间步没有信噪比等于零. 但好吧
doesn't have the signal
正确, 因为最后的时间步没有信噪比等于零. 但好吧
A to. noise. ratio equal to zero but okay this is something we dont. need. to. care about.
这是我们现在不需要关心的事情
A :to. noise. ratio equal to zero but okay this is something we. dont. need. to. care about
这是我们现在不需要关心的事情
now actually
10 W :
如果你感兴趣, 我会在评论中链接这篇论文
If you're interested I will link the paper in the comments.
Also. this. code I took it from Hugging Face diffusers library. because. l mean we. are.
意思是我们在应用公式, 所以即使我自己写, 也不会有任何
不同, 因为我们只是应用论文中的公式.
oc applying formulas so even i wrote it by myself it wouldn't. be any different.
不同, 因为我们只是应用论文中的公式.
because we are just applying formulas from. the paper.
我们需要做的第一件事是根据论文的公式15 计算原始样本.
the. first. thing we need to do is to compute the original sample according to. the
that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics. 我这么说是什么意思? 如你所见, 高在哪儿这个可
gto the
xo, but we found this to lead to worse sample quality early in our experiments.) We have shown that formula i5uftheparsrwnardoineih by this asyodcaseewnereisir this one
it is just another parameterization of p(x x), so we verify its ef fee tiveness io S ction an model'svariational bound to an. objective that rescmbles deoisingscore matching. Nonetheless.
. 3 宫在哪儿? 所以 实际上 让我在这里给你看男一
wherei sitberesoactuaulyiet me show you another formula here
Since our simplified objective ) discards the weighting in Eq. 2). it is a weighted variational
with very small moun s of noisc, so jt is. heneficiaf o dow-weight them s hat the etwork can where is it here so actually let me show you another formula here
24. 1 Forward process and L We ignore the fact that the forward processes Bare lear mable by re parameterization and instead fix them to constants (see Section f of dea is ) Thus in our mplementation the approximate pr where is it here so actually let me show i you another formula here
objective for diffusion models ( Section ). Ultimately, our model design is justified by simplicity and empirical results ( Section ). Our discussion is categorized by the terms of Eq. ).
25. 1 Forward process and LT
objective for diffusion models ( Section B ). Ultimately, our model design is justified by simplicity 前向
objective for diffusion models ( Section B ). Ultimately. our model design is justified by simplicity As your car f see g wer can real culate the previous step. so the less noisy, the forward
objective for diffusion models ( Section B ). Ultimately. our model design is justified by simplicity 像,
process sorry y the'reverse process
objective for diffusion models ( Section ). Ultimately. our model design is justified by simplicity the nie an is r defined i a this way and the variance is defined in this way.
objc ctive for diffusion models ( Scc tion ). Ultimately, our model design is justified by simplicity and empirical results ( Section ). Our discussion is categorized by the terms of Eq. ).
26. 1 Forward process and LT
objc ctive for diffusion models ( Section B ). Ultimately, our model design is justified by simplicity 3. 1 Forward process and LBu wba is Cle prediced
objective for diffusion models ( Section ). Ultimately. our model design is justified by simplicity and empirical results ( Section ). Our discussion is categorized by the terms of Eq. ).
27. 1 Forward process and LT
objective for diffusion models ( Scction3 ). Ultimately, our model design is justified by sm plicity
objective for diffusion models ( Section ). Ultimately. our model design is justified by simplicity and empirical results ( Section ). Our discussion is categorized by the terms of Eq. ).
28. 1 Forward process and L
objc ctive for diffusion models ( Scc tion ). Ultimately, our model design is justified by simplicity
29. 1 Forward process and L
Wc新以这个预测的x0c我们他可以检索它n
pustcrinrglusu Soythis predicted x0, we can also retrieve it. beign ol
We set T = 1000 for all cxperiments so that the number of neural network evaluations needed during sampling matches previous work [53. 55]. We set the forward process variances to constants increasing linearly from 310 to By = 0. 02. These constants were chosen to be small relative to data scaled to 1, 1], ensuring that reverse and forward processes have approximately the same functional form while keep ins the signal-to-noise ratio at xz as small as. possible ( Lz
to the network using the Transformer sinusoidal position embedding 60]. We use self-attention at
48] with group normalization throughout [66 ]. Parameters are shared across time, which is speeified
the 16 × 16 feature map resolution [63, 60]. Details are in Appendix B.
the same functional form while kce ping the signal-to-noise ratio at x as small as possible ( L
Dk.(g(xx) N(0. I))
10bits per dimension in our experiments ).
To represent the reverse process, we use a U-Net backbone similar to an unmasked Pixel CNN++ [52.
48] with group no malization throughout [66 ] Parameters are shared across tme. which is specified
Since our simplified objective ) discards the weighting in Eq. (2), it is a weighted variational e _data
30. 1 Sample quality 如果我没记错的话, 它在这里
C IF using'the formula nuimber15:ifiremember correctly u it's here y
使用公
byr使用公式15y如果我设记错的话, 它在这里d B
Progresive lossy compression We can probe further into the rate-dist orion behav jor of our model
sau using the formula number 15yif I remember correctly it is here yd
san ple x ~(x) using approx in a tl DL((x]p(xj bsts on averge Fixr any dist nb to ns p and which ae access to. a procedure such sminialraniomcoding9201. that can transmit a using the formula number 15, iflremember correctly, it's here
Figure 5: Unconditional CIFAR10 test set rate-distortion vs. time. Distortion is measured in root mean square d
error oe a [0, 255| seale. See Table for details.
progressive decompression from random bits. In other words, we predict the result of the reverse Progressive generation We also run a progressive unconditional generation process given by
error on a [0, 255] scale. See Table for details, 实际上在数值上是等价的, 从噪声
fed in root mean squared
I arge number of degrces of frcedom in implementation. One must choose the variance s β3 of the
unddeng
31. 1 Forward process and L 所以事实上山例如在代码中他们说从xt到x减去
So. as ar matter of fact i for example /here in the icode y they say to go from xt to xt
xo, but we found this to lead to worse sample quality early in our experiments.) We have shown that
that optimizing an objc tive resembling denoising score matching is c qui valent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynami es.
(12 )
I : repeat 但实际上它们是相同的东西, 圆为bt等于 T减泰alpha But actua alto one minus alpha tas
3
t乘以beta. alpha定义为1减去beta, 如你所记.
p( Xo: T) 在这个公式中我们
obtain in
3 Diffusion models and denoising auto encoders 在这个公式中我们
3 Diffusion models and denoising auto encoders 需要计算均值. 我们需要计算方差 根据这里的公式
in which we need to calcu i ate the mean and we need to calculate the variance
3 Diffusion models and denoising auto encoders 我们知道
3 Diffusion models and denoising auto encoders 我们不
other alphas we know because there are :parameters that depend on-beta i what we don't
Now we discuss our choices in po(x1x)= N(x:μg(x. t).∑(x. t)) for 1<t≤ T. First,
other alphas we know because there ire parameters that depend on beta what we don't
the same functional form while kce ping the signal-to-noise ratio at x as small as possible ( L =
0可以按照这里的公式15计算
知道的是x0 但x0可以按照这里的公式15计算
所以首先使用 DDTM论文的公式15计算预测的原始样本.
So. first, compute the predicted original sample using formula 15. ofthe DDTMpaper.
预测的原始样本
延迟减去, 同时所以我们做延迟减去1减去alphat
latency minus while so we do latency minus the'square root of 1. minus. alpha. twhat. is
延迟减去, 同时. 所以我们做延迟减去1减去alphat
Iminusalphatwhat is
延迟减去, 同时所以我们做延迟减去1减去alphat
latency minus while so we do ta tency minus the square root. of 1. minus. alphatwhat. is
延迟减去, 同时. 所以我们做延迟减去1减去alphat
的平方根. 1减去alphat的平方根等于beta.
的平方根. 1减去alphat的平方根等于beta.
latency minus while so we do ta tency minus the square root of 1. minus. alpha twhat. is
的平方根. 1减去alphat的平方根等于bet a.
:the squarerootof1minusalphat isequattobeta soihave herebeta. twhich. is.
的平方根:1减去alphart的平方根等于beta.
所以我这里有bet at 它已经是1减去alphat, 如你
the squarerootof1 minusatphatisequattobeta soihave herebeta. twhich. is
所以我这里有betat它已经是1减去alphat, 如你
already'l
所见, alpha bar时间步t的1减去alphabar, 因为我
minus alpha tas you can see alpha bar1 minusalphabarat the. time step. t. because. i
所见, alpha bar:时间步t的1减去alphabar, 国为我
所见, alpha bar时间步t的1减去alphabar, 因为我
minus alpha tas you can see alpha bar1 minus'alpha. barat. the. time step. t because. i
所见, alpha bar时间步t的1减去alphabar, 因为我
already retrieve it from here so
已经从这里检索了它.
32. 5
Figure 5: Unconditional CIF AH
到0. 5次方或beta的平方根. 所以我们做latents减去
uhone
到0. 5次方或beta:的平方根. 所以我们做latents减去
uh. one. minusuh'sorry beta'to the power to to the power. of one. half or. the square.
beta乘以30的05次方这基本上意味着beta的
Aroot. ofbeta so we do latent s minus beta prod at times 30. to. the. powerof. 0. 5which
beta乘以30的0:5次方这基本上意味着beta的
it means basically'square root of beta
beta乘以30的0. 5次方, 这基本上意味着beta的
平方根
33. 5
Figure S: Unconc
然后我们乘以时间步t 的潜在图像的预测噪声. 那么什么
然后我们乘以时间步的潜在图像的预测噪声. 那么什么
and. then we multiply this by the predicted noise of the image of. the latent at. time
然后我们乘以时间步t 的潜在图像的预测噪声:那么什么
然后我们乘以时间步的潜在图像的预测噪声. 那么什么
step. tso what is the predicted noise it's the model output because. our unit predicts
它是模型输出因为我们的单元预测噪声模型输出. 然后
the hoise
它是模型输出因为我们的单元预测噪声模型输出. 然后
model output and then we heed'to divide'this by let me check square root. of alpha. t
我们需要除以端让我检查一下一
alpha t 的平方根
model output and then we heed'to divide this by let me check square root of alpha. t.
我们需要除以一一让我检查一下一
alpha t 的平方根
modelo
我们有的, 我想, 这里的alpha t.
model
我们有的;我想, 这里的alpha t.
which we have i'think here alpha there
所以alpha t的平方根alphaprodt的0. 5次方.
Sothe square root of alpha't, alpha pro dt to the power of 0. 5.
这里我有一些东西
Here T have something.
这个我不需要
这个我不需要
This one
因为否则它是错误的, 对吗?
Because otherwise it's wrong, right?..
因为否则它是错误的, 对吗?
34. 5
Figure 5: Unconditio
首先这两个项之间有一个乘积, 然后有一个差.
首先这两个项之间有一个乘积, 然后有一个差
First there is a product between these two terms and then. there. is. the difference.
这就是我们计算预测;x0 的方式.
This is how we compute'the prediction, the xo..
这就是我们计算预测:x0 的方式.
relative to data scaled to 1. 1], ensuring that reverse and forward processes have approximately increasing linearly from 3=10- to 3 = 0. 02. These constants were chosen to be small the same functional form while kce ping the signal-to-noise ratio at x as small as possible ( L =
48|with groupnor Now:letisgo bac to :the formula @umben7yhichisspecicd
Oun(( L, fiocd sopc )
35. 46±0. 11
36. 67±0:13
37. 17
5 3. 75 (3. 72)
38. 70(3. 69
raining. However, we found it beneficial to sample quality (and simpler to implement. o. train. on he
Diffusion models and denoising auto encoders 好的
3 Diffusion models and denoising auto encoders 然后从这个分布中采样
so we calculate. this mean and this variance and then wc sample from this distribution
所以我们计算这个均值和这 个方差 然后从这 布中采样
so we-calculate this mean and then we sample. from. this distribution
所以计算红色原始样本和当前样本x t的系数. 这是你在
ocompute the coefficients
所以计算红色原始样本和当前样本xt 的系数. 这是你在
for red original sample and the current sample xt this is the same. comment that. you
diffusers 库中可以找到的相同注释, 基本上意味着我们需要计算
for red original sample and the current sample xt this is the. same. comment that. you.
3 Diffusion models and denoising auto encoders 基本上意味着我们需要计算
can find on the diffusers library v hich-basically means we need to compute this one
这里这个, 所以预测原始样本系数, 等于alpha prod t
这里这个, 所以预测原始样本系数, 等于alpha prod t
sopredicted original sam pte coefficient which is equal to. what alpha prod t minus.
3 Diffusion models and denoising auto encoders 这里波个 所以预测原始样本系数.
so predicted original sam pie co cf ficient which is equal to what alpha prod t minus allow
3 Diffusion models and denoising auto encoders 意味着
one so the previous alpha pio dt which is alpha prod t previous which means the alpha ne van i able mode Ts but the yllow
所以前一
-个alpha prod t, 即alpha prod "t前 意味着
one so. the. previous alpha prod t which is at pha prod t previous which. means the-alpha
alpha prodt 但在前 个时间步的平方根下 所以0. 5
prodtbut at the previous time step
Diffusion models and denoising auto encoders 所以0. 5
large number of degrees of freedom in implementation
alpha prodt但在前 所以0. 5
under the. square root so'to the power of o. 5multiplied by the current beta tso. the
次方乘以当前的beta t. 所以时间步t的beta, 即
under the. square root so'to the power of o. 5multiplied by. the. current beta tso. the
次方乘以当前的betat
3 Diffusion models and denoising auto encoders under the square foot so to the power of ou5muitiplied by the current beta tso the but the ysllow a
次方乘以当前的beta t. 所以时间步t的beta, 即
under the. square root so to the power of o. 5multiplied by the. current beta tso. the
找 百前的bet at.
bet a. at. thetimesteptsocurrentbeta'twhichiswe defineitherecurrent. beta. twe
我们可以有 好的 因为
retrieve it from alpha we cout d have a okay and then we. divide it by.
我们可以有一个.
3 Diffusion models and denoising auto encoders 好的 因为
3 Diffusion models and denoising auto encoders 然后
beta product t because one minus alpha bar is actually equal to be ta barbeta product allow e
beta product t because one minus alpha bar is actually equal. to be ta barbeta product
我们有了这里的这个系数:所以这里这
t then. we. have the this coefficient here so'this one here. so. this. is. current. sample
3 Diffusion models and denoising auto encoders 我们有了这里的这个系数 所以这里这个
所以这是当前样本系数等于当前alpha t的0. 5
:tthen. we. have the this coefficient here so this one here so. this. is. current. sample
所以这是当前样本系数等于当前alpha t的0. 5
coefficient is equal to current alpha t to the power of 0. 5which. means. the square.
次方.
3 Diffusion models and denoising auto encoders 这意味着这个时间的平方根, 这个, 这里的东西
lar root of this time'this this thing here so the square root of alpha ts
3 Diffusion models and denoising auto encoders Diffusion models might aprbeestrcted lassr ata Ble models but they allow a large ote f this. time'this this thing :here so the square i root iof alpha it -
3 Diffusion models and denoising auto encoders Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances of the
3 Diffusion models and denoising auto encoders 因为1 减前一个时间步
and then we multiply it by beta at the previous time step because-it sl-minus alpha
3 Diffusion models and denoising auto encoders at the previous time step corresponds to beta at-the previous time steps time step
的a lpha对应于前个时间步的beta 时间步乘以betaprod t at. the previous time step corresponds to beta at the previous time steps. time. step
的a lpha对应于前个时间步的beta 时间步乘以betaprod t multiplied
的a lpha对应于前个时间步的beta 时间步乘以betaprod t by beta prod t prev divide by beta at the time step. tso. beta prod. t
3 Diffusion models and denoising auto encoders by beta pro dtpreudivideibybetaatche time step tso beta prod t but they allow a
前一个除以:时间步t 的beta 所以beta prod t.
by beta prod t prev divide by beta at the time step. tso. beta prod. t
现在我们可以计算均值. 所以均值是这两个项pred prev 样本
now we. can compute the mean so the mean'is the sum of these. two terms. pred. prev
现在我们可以计算均值
3 Diffusion models and denoising auto encoders now we l can i compute the mie an i sothe mean is the sum of these two terms pred prey
现在我们可以计算均值. 所以均值是这两个项pred prev 样本
now. we. can compute the mean so the mean is the sum of these. two terms. pred prev.
的和. 所以让我在这里写一些
now. we can compute the mean
计算预测的前一个样本均值mod t等于预测原始样本系数
sampleso. let me write some here'compute the predicted previous sam pte. mean. mod. t
计算预测的前一个样本均值mod t 等于预测原始样本系数
计算预测的前一个样本均值mod t 等于预测原始样本系数
is equal to predicted origin at sam pte coefficient multiplied by what?.
is equal to predicted origin at sam p
fficient multiplied by what?.
3 Diffusion models and denoising auto encoders Diffusion models might appear to be a est nic ted ass of latent variable models but they allow a is :equal to predicted original sample coefficient multiplied by what?
乘以零 零是什么?
" By x zero, what is x zero?
是我们通过公式15得到的这个, 所以预测原始样本
: Is. this. one that we obtained by the for mutanumber15, sothe. predicted. original
所以×零加上这里的这个项, 这个项是什么?
v: Is. this. one that we obtained by the for mutanumber15, sothe predicted. original.
所以零加上这里的这个项, 这个项是什么?
sample, sox zero pt us this term here, what is this. term?
3 Diffusion models and denoising auto encoders 所以x. 零加上这里的这个项 这个项是什么?
largenu sample g sog zero plus this-term-lere ywka tis this term? of the
所以×零加上这里的这个项, 这个项是什么?
sample, sox zero pt us this term here, what is this term?
是这里的这个, 所以当前样本系数乘以 Xt, xt是什么?
ls. this. one here, so the current sample coefficient mutt ip tied :by xt, what is xt?
3 Diffusion models and denoising auto encoders 是这里的这个.
Is this one here so the current sample coefficient multiplied by xt :what is xt!
but the allow
3 Diffusion models and denoising auto encoders Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in j mplementation. One must choose the variances of the
是时间步 的延迟
现在, 我们已经计算了当前的均值.
Now, we have computed the mean for. now.
3 Diffusion models and denoising auto encoders 现在. 我们已经计算了当前的均值
large number of d Now g we have @om puted ehe meanifortiowyiances5of the
3 Diffusion models and denoising auto encoders Diffusion models might appear to be are stic ted class of latent variable models, but they allow a large number of degrees of freedom in imple me in tation. One must choose the variances, of the
我们还需要计算方差.
We need to compute also the variance.
让我们创建另一个方法来计算方差.
Let'screate another method'to compute the variance
test. get VARIANCE 好. 的 我们得到了前
test. get VARIANCE Okay, we obtained the previous time test t because we. need. to. do..
需要进行后续的四个计算
A :test. get VARIANCE Okay, we ob t
ioustime test t because we. need. to. do..
需要进行后续的四个计算
four later
再次我们计算alpha prod t. 所以我们需要计算这些特定项的
again we. calculate the alpha prod tso alt the terms that we need. to calculate these.
3 Diffusion models and denoising auto encoders again we. calculate the alpha prod tso alt the terms that we need to calculate these
3 Diffusion models and denoising auto encoders 所有项
Diffusion models might appear to be a ref f latenvariable models but they allow large number of degrees of free dopaes@g9egt choose the variance s8 of the
所有项
当前的beta. t等于1减alphapro. d. t. 是的1
and. the current bet a tis equal to one minus alpha prod t alpha prod this one what. is
3 Diffusion models and denoising auto encoders 是的
减alpha prod. t 除以alpha prod t -零
:current. bet a tis equal to one minus alpha prod t yeah. one minus. alpha prod t divided.
3 Diffusion models and denoising auto encoders current beta tis egual to one minus i alpha pro dtyeahone minusaiphaprodtdivided
减alpha prod. t 除以alpha prod t -零
by alpha prod t zero okay so the variance
好的, 所以根据公式6和7, 这里的方差给定为1
byalphaprodtzerookayso the variance
好的, 所以根据公式6和7, 这里的方差给定为1
好的, 所以根据公式6和7, 这里的方差给定为1
According. to the formula number 6and7, so this formula. here, is given as 1 minus
Diffusion models and denoising auto encoders 好的 所以根据公式6和7 这里的方差给定为1
According to the formula number 6rand-iso this formulas here gisgiven as1minus
减alpha prod T前一 所以1减alphaprod T前一个.
According. to the formula number 6and7
7, so this formula. here, is given as 1 minus.
减alpha prod T前一 个 所以_1减alphaprod T前 个
alphaprod Tprev, so1 minus alpha prod T prev.
3 Diffusion models and denoising auto encoders large number oalphasprod Tprevysorl-minus alpha prod Thpretpces5 of the
减alpha prod T前一个. 户 所以1减alphaprod T育 前 个
alphaprod Tprev, so 1 minus alpha prod Tprev.
除以1减alphaprod, 也就是1减alphaprod. 为什么是
divided by. one minus alpha prod which is one minus alpha prod why prod because. this
3 Diffusion models and denoising auto encoders 为什么是
divided by one minus alpha prod which is one minus alpha prod why prod because this
prod? 因为这是alpha ba r, 乘以当前的bet a,
betat,
bet at 是
divided by. one minus alpha prod which is one minus alpha prod why prod because this
3 Diffusion models and denoising auto encoders 乘以当前的beta beta t.
betat是
Diusionodemighapp
3 Diffusion models and denoising auto encoders is the alpha bar and mui tip lied by the current beta bet at and beta t is defined i Diffusion models might appear to be a rest ct edf latent variable models but they allow a
3 Diffusion models and denoising autoencoders 我不记得它在哪了 这是我们的方羞
isthe alphabarand muitiplied bythe current bet abetat and beta tis defined i
我不记得它在哪了. 1 减alpha, 这是我们的方差.
:remember where it's one minus alpha and this is our variance. we. clamp. it. oops
我们将其限制, 哎呀torch clamp 方差我们希望的最小值是
remember where it's one minus alpha and this is our variance. we. clamp. it. oops.
我们将其限制, 哎呀torch clamp 方差, 我们希望的最小值是
我们将其限制, 哎呀torch clamp 方差, 我们希望的最小值是
torch. clamp variance and the minimum that we want is1equal. to. minus20. tomake sure
1e-20, 以确保它不会达到0然后我们返回方差
torch. clampvariance and the minimum that we want is1equal. to. minus20. tomake sure
现在我们有了均值和方差, 这个方差也是用 让我写在
that it. doesn't reach O and then we return'the variance e and now. that we have. the. mean
现在我们有了均值和方差这个方差也是用一 一让我写在
and the
现在我们有了均值和方差这个方差也是用一 让我写在
variance-so this variance has also been computed using let me write. here. computed
这里, 用 DD PM 论文的公式七计算的. 现在我们回到我们的
variance-so this variance has at so been computed using let me write. here. computed
这里, 用 DD PM 论文的公式七计算的. 现在我们回到我们的
using formula
这里, 用 DD PM 论文的公式七计算的. 现在我们回到我们的
seven of. the DD PM paper and now we go back'to our step. function so what we. do. is
步骤函数·
:seven of. the DD PM paper and now we to our step function so what we do. is
所以我们做的是等于零一因为我们只有在不是最后一个时间
seven of. the DD PM paper and now we go back to our step function so what we. do. is
所以我们做的是等于零一因为我们只有在不是最后一个时间
equal to zero
所以我们做的是等于零, 因为我们只有在不是最后一个时间
Because we only need to add the variance if we are not at the. last. time step
步时才需要添加方差
Because we only need to add the variance if we are not at the. last. time step.
如果我们处于最后一 一个时间步, 我们没有噪声, 所以我们不
f we are at the last t me step, we have no noise, so we dont add any Wed on
添加任何. 实际上, 我们不需要添加任何噪声
lf we are. at the last time step, we have no noise, so we dont add any. Wed on.
添加任何 实际上, 我们不需要添加任何噪声.
need to add any noise, actually
因为重点是, 我们将从这个分布中采样.
Because the point is, we are going to sample from. this. distribution.
3 Diffusion models and denoising auto encoders 因为重点是 我们将从这个分布中采样
Diffusion modem ght appar to be rest cte ass or la len Van able mode is b they allow a I Because the point is :we are going to sample from. this distribution.
3 Diffusion models and denoising auto encoders 我们实际上从 N01. 中采样 然后
And just like we did before ;we actually sample from the-No ly and then we shift it
根据公式进行偏移.
And just like we did before, we actually sample from the No1, and then we shift it
3 Diffusion models and denoising auto encoders 根据公式进行信移
Diffusion models might ap per car i able models but they allow a large number of degrees of fre@l@@@r0gfoce So0mulaghoose the variances, of the
3 Diffusion models and denoising auto encoders Gaussian with a particular mean and a particular variance-is equal to the Gaussian at
3 Diffusion models and denoising auto encoders 加上均值
Diffusion models might appear to be astncte dass latent variable models but they allow a large numb multiplied by the i standard deviation plus the mean ;6 of the
所以我们采样噪声
So... We sample the noise.
实际上这是已经乘以噪声的方差. 所以实际上是标准差
okay we sample some noise compute the variance. actually this is. the
实际上这是已经乘以噪声的方差. 所以实际上是标准差
实际上这是已经乘以噪声的方差. 所以实际上是标准差
variance already multiplied by the noise so it's actually. the standard. deviation
因为我们会在时间之后看到self dot get variance.
:variance already multiplied by the noise so it's actually. the standard deviation
时间步t的0. 5次方所以这个0. 5, 所以这个变成了
because. we will see self dot get variance after the time step t. to the power. of. o. 5
时间步t的05次方所以这个0. 5, 所以这个变成了
sothis0. 5so
时间步t的05次方所以这个0. 5. 所以这个变成了
this. one becomes the standard deviation we multiply it. by. the. n01. so what we-are
所以我们所做的是基本上我们正从n01到具有特定u和
v. this. one becomes the standard deviation we multiply it by. the. n01. so what we. are
所以我们所做的是基本上我们正从n01到具有特定u和
doing is basically we are going from. n01
所以我们所做的是基本上我们正从n01到具有特定u和
所以我们所做的是基本上我们正从n01到具有特定u和
A:to. n. with a particular mu and a particular sigma using the. usual. trick. of. going from..
特定o的n使用通常的技巧从... x等于u加
A to. n. with a particular mu and a particular sigma using the. usual. trick of. going from..
特定的n;使用通常的技巧从.. 等于加
:x. is equal to the mu plus the sigma actually not yeah this is sigma squared then
是的, 这是平方, 因为这是方差乘以z,
是的 这是平方, 因为这是方差乘以z,
because this
是的, 这是平方, 因为这是方差乘以z,
is the. variance sigma multiplied by the z where z where z is. distributed according-to
其中z, 其中z根据n01分布.
then01
这是我们一 一直做的事情无论是编码器的变分, 还是添加
this. is the same thing that we always done also for the variation. of. the encoder also
3 Diffusion models and denoising auto encoders 噪声,
for adding the noise the same thing that we did before. this is d how you san ple from a Diffusion mode might appa robe resncedcssoenanibleodesb they allow a
这就是你如何从一个分布中采样, 如何实际偏移海洋分布的
distribution how you actually shift the parameter of the. ocean distribution
所以预测的前样本等于预测的前样本加上方差这里的方差项
so predicted prev sample is equal to the predicted prev sample plus. the variance. this.
已经包括了乘以然后我们返回:预测的前样本
variance. term here already includes the sigma multiplied. by z. and. then we. return
已经包括了乘以z, 然后我们返回:预测的前样本.
predicted prev sample oh okay now we have also built the
哦, 女 好的, 现在我们也构建了采样器. 让我检查一下我们
predicted prev sample oh okay now we have also built the
哦, 好的, 现在我们也构建了采样器. 让我检查一下我们
哦, 好的, 现在我们也构建了采样器. 让我检查一下我们
the. sampler let me check if we have everything no we missed still still something
是否都有了. 不, 我们还是漏了什么, 那就是set strength the. sampler let me check if we have everything no we missed still still something
方法, 记得当我们想做图像到图像的时候. 所以让我们回去
which. is the. set strength method as'you remember once we-want. when we. want. to. d..
方法, 记得当我们想做图像到图像的时候. 所以让我们回去
image to image so let'sgo back to check our slides..
检查我们的幻灯片.
image to image so let'sgo back to check our slides.
检查我们的幻灯
image to image so
检查我们的幻灯片.
如果我们想做图像到图像, 我们使用 VAE 将图像转换为潜在
空间. 然后我们需要向这个潜在空间添加噪声, 但我们可以
决定添加多少噪声
我们添加的噪声越多, 单元改变图像的自由度就越大. 我们
添加的噪声越少, 它改变图像的自由度就越小,
添加的噪声越少, 它改变图像的自由度就越小
freedom. it will have to change the image so what we do. is. basically. by setting. the.
所以我们所做的基本上是通过设置强度, 我们让采样器从一个
strength we make our sampler start from a particular. noise level.
特定的噪声水平开始, 这正是我们想要实现的方法.
特定的噪声水平开始, 这正是我们想要实现的方法.
and this is. exactly what the method we want to implement so. for. example. as soon. as. we.
所以, 例如, 一旦我们加载图像, 我们就设置强度, 这
and this is. exactly what the method we want to implement so. for. example. as. soon. as. we.
将改变我们从哪个噪声水平开始, 然后我们向我们的潜在空间
load the. image we set the strength'which will shift the noise level from which we
将改变我们从哪个噪声水平开始, 然后我们向我们的潜在空间
start'from
将改变我们从哪个噪声水平开始, 然后我们向我们的潜在空间
and. then. we add noise to our t a tent'to create the image. to image. here. so. let's-go
添加噪声以在这里创建图像到图像. 户 所以让我们来这里并创建
and. then. we add noise to our t a tent'to create the image to image. here. so. let's. go
添加噪声以在这里创建图像到图像. 所以让我们来这里并创建
添加噪声以在这里创建图像到图像. 所以让我们来这里并创建
and. then. we add noise to our t a tent'to create the image to. image. here. so. let'sgo
这个名为:set Strength. 的方法
here and we create this'method called set Strength
好的, 开始步骤 因为我们将会跳过一些步骤
a :ok the start step because we will skip some steps is equal to self. num ln ference steps
t + if._pt_prerios_tiestep(t)
这基本上意味着如果我们有50 个推理步骤并且我们把强度
This. basically means that if we have 5o inference steps and we. set the strength. to,
设置为, 比如说, 0:8, 这意味着我们将跳过20
let'ssay, 0. 8, it means that we will skip20%of. the steps.
所以当我们从图像到图像开始时, 例如, 我们不会从一个纯
So when we. will start from image'to image, for example, we will. not start. from a-pure
所以当我们从图像到图像开始时, 例如, 我们不会从一个纯
噪声图像开始, 而是从这个图像的80
moiseimage but we will start fno m So of noise in this in
所以单元仍然有改变这个图像的自由度, 但没有100
我们重新定义时间步, 因为我们正在改变时间表.
Were define the time steps because we are altering the. schedule..
所以基本上我们跳过一些时间步
So basically we skip some time'steps.
and. self. dot start step is equat'to start step'so actually what we. do here is. suppose.
所以实际上我们在这里做的是, 假设我们有80 and self dot start step is equal to start step so actually what we do here ri
所以我们从一个图像开始, 我们给它加噪, 然后我们让单元
相信他已经生成了这个具有特定噪声水平的图像, 现在他必须
继续去噪, 直到一一当然、他根据提示一一直到
我们达到没有任何噪声的干净图像.
acc on ding. of course also to the prompt 、until we
我们达到没有任何噪声的干净图像
image without'any noise.
现在我们有了可以调用的管道, 我们有dd pm 采样器, 我们
Now we have'the pipeline that we can. call
现在我们有了可以调用的管道, 我们有dd pm 采样器, 我们
现在我们有了可以调用的管道, 我们有dd pm 采样器, 我们
we have. the dd pm sampler we have the model built of course we need to. create. the
有了构建好的模型. 当然, 我们需要创建加载这个模型权重
we. have. the dd pm sampler we have the model built of course we need to. create. the
我们在这里称之为模型加载器. 模型加载器, 因为现在我们
function. to load the weights of this model so let'screate. another. file we. will. call.
我们在这里称之为模型加载器:模型加载器, 因为现在我们
it the model
我们在这里称之为模型加载器:模型加载器, 因为现在我们
:loader. here model loader because now we are nearly close. to. sampling. from. this
几乎接近于从最终的这个表格扩散中采样. 所以现在我们需要
loader. here model loader because now we are nearly close. to. sampling from. this.
创建加载预训练权重的方法, 这些权重我们之前已经下载过了.
finally. from this table diffusion so now we need to create the. method to. load. the
所以让我们创建它:导入clip 编码器-va 编码器.
so let'screate it import clip encoder va encoder then from decoder import. va. decoder
然后从解码器导入, va, 解码器, 融合, 导入扩散,
fusion import diffusion our diffusion model which is. our unit.
我们的扩散模型, 也就是我们的单元.
fusion import diffusion our diffusion model which is. our unit
现在让我先定义它. 然后我告诉你我们需要做什么. 所以从
now. let. me first define it then i tell you what we need to do. so preload models. from.
标准权重预加载模型.
standard weights okay as usual we load the weights using. torch
好的, 像往常一样, 我们使用torch 加载权重, 但我们还
standard weights okay as usual we load the weights using. torch
好的, 像往常一样, 我们使用torch 加载权重, 但我们还
好的, 像往常一样, 我们使用torch 加载权重, 但我们还
but we-will. create another function model Converter. load From Standard Weights. this. is. a.
将创建另一个函数:model Converter load From Standard Weights. 这是我们稍后将创建的方法, 用于
but we-will. create another function model Converter. load From Standard Weights. this. is. a :
加载预训练权重, 我会向你展示为什么我们需要这个方法.
method. that we will create later to load the weights the pre-trained weights and. l..
然后我们创建我们的编码器并加载状态字典.
will show you why we need this method then we create-our encoder.
然后我们创建我们的编码器并加载状态字典.
从我们的状态字典加载状态字典.
Load state dict from our state dict.
并且我们也设置严格为2.
Andwe also setstrict to2.
哎呀.
Oops.
加载严格字典
Load strict dict.
严格.
Strict.
然后我们有解码器.
Then we have the decoder.
而且它也是严格的.
And it's strict also.
所以这里的严格参数基本上是告诉你, 当你从 Py Torch 加载模型
So this. strict parameter here basically tells that when you load a model from
时, 例如, 这里的ck pet 文件, 它是一个包含许多键的
Py Torch, this for example, this ck pet file here, it is a. dictionary. that. contains
每个键对应于我们模型中的一个矩阵.
And each key corresponds to one matrix of. our model.
例如, 这个组归一化有一些参数.
So for example, this group normalization has some parameters.
那么, Torch 如何准确地将这些参数加载到这个组归一化中呢?
And. how can Torch toad these parameters exactly-in. this. group norm?..
通过使用我们在这里定义的变量名称.
By using the name of the variables that we. have. defined here
当我们从 Py Torch 加载模型时它实际上会加载一个字典.
And when we load a'model from Py Torch, he will actually load a dictionary
然后我们将这个字典加载到我们的模型中, 它会通过名称
: And. then we load this dictionary into our models and. he-will. match by-names.
现在的问题是预训练模型, 实际上它们没有使用我使用的相同
Now. the problem is the pre-trained model, actually they don't. use the same. name. that
名称, 实际上这段代码是基于我看到的另 段代码
L have used and actually'this'code is based on another. code. that I have-seen.
所以我们使用的名称实际上与预训练模型的名称不同
So actually the names'that we use are not the same as. the pre-trained. model.
此外, 因为预训练模型中的名称并不总是非常适合学习
Also because the names in the pre-trained model are not always very friendly. for.
此外, 因为预训练模型中的名称并不总是非常适合学习
learning.
这就是为什么我更改了名称其他人也更改了方法的名称.
This is. why I changed then amies and also other people changed. the names of. the.
这就是为什么我更改了名称, 其他人也更改了方法的名称.
methods.
但这同时也意味着预训练模型的名称与这里其他类中定义的名称
But this also. means that the automatic mapping between the names of. the-pre-trained
但这同时也意味着预训练模型的名称与这里其他类中定义的名称
model and the names defined in other classes here
之间的自动映射无法实现, 因为名称不匹配, 无法自动
model and the names defined in other classes here
之间的自动映射无法实现, 因为名称不匹配, 无法自动
之间的自动映射无法实现, 因为名称不匹配, 无法自动
Can not. happen because it can not happen automatically because the names. do. not. match
因此, 我在我的 Git Hub 库中创建了一个脚本, 你需要下载它
for. this reason there is a script that I have created in my git hub. library. here and
来转换这些名称.
它只是一个将一个名称映射到男一个名称的脚本.
It's just a script that maps one name into another.
s this one, map it into this
map it into this
所以这个将被命名为model converter. py.
So this will be called model converter. py.
Model converter. py.
Model converter. py.
就这样
And that's it.
它只是一个非常大的名称映射.
It's just a very big mapping of names.
我从 Gi t Hu b上的这个评论中提取它
And T take it from this comment here on Git Hub.
所以这是模型转换器. 我们需要导入这个模型转换器, 导入
so. this. is model converter so we need to import this model. converter import. model.
这个模型转换器基本上会转换名称, 然后我们可以使用load converter. this model converter basically will convert the names. and then. we. can. use
现在名称会相互映射, 这个strict 确保如果有任何一个名称
state. dict and this will actually map all the names now. the. names. will map. with. each.
没有映射, 就会抛出异常, 这正是我想要的, 因为我希望
other-and. this strict makes sure that if there is even one. name that doesn't. map. then
没有映射, 就会抛出异常, 这正是我想要的, 因为我希望
throw an
没有映射, 就会抛出异常, 这正是我想要的, 因为我希望
exception-which is what I want because I want to make sure. that all. the. names. map
所以我们定义了扩散, 并加载其状态字典:diffusion 和strict 等于
So. we define the diffusion and we load its states dict diffusion. and strict. equal. to
true. 让我检查一下.
So. we define the diffusion and we load its states dict diffusion. and strict. equal. to
true. 让我检查一下.
true and let me check
然后我们执行:clip 等于clip to Device, 月 所以我们将其移动到我们想要
then. we. do clip is equal to clip. to Device so we move it to the device where. we want..
工作的设备上, 然后我们也加载它的状态字典, 即参数
then. we. do clip is equal to clip. to Device so we move it to. the device where. we. want.
权重, 然后我们返回一个字典clip clip, 然后我们有编码器是
to work and then we load also his state dict so the parameters, the weights and. then
权重, 然后我们返回一个字典clip clip, 然后我们有编码器是
we return a dictionary clip uaa i
权重, 然后我们返回一个字典clip clip, 然后我们有编码器是
clip and. then we have the encoder is the encoder we have the. decoder is. the. decoder
编码器, 我们有解码器是解码器, 然后我们有扩散, 我们
clip and then we have the encoder is the encoder we have. the decoder is. the. decoder
编码器, 我们有解码器是解码器, 然后我们有扩散, 我们
and then we have the diffusion we have the. diffusion. etc
现在我们有了运行所需的所有要素. 终于, 可以进行推理
now. we. have all the ingredients to run finally the inference. guys so. thankyou. for
了. 非常感谢大家的耐心等待, 我们终于可以看到曙光了.
now. we. have all the ingredients to run finally the inference guys so. thankyou for.
所以让我们构建我们的笔记本, 这样我们就可以可视化我们将
being patient so much and it's really finally we have we can see. the light. coming-so
构建的图像. 好的, 让我们选择内核:stable diffusion. 我已经创建
our. notebook so. we can visualize the image that we will build okay let'sselect the
构建的图像. 好的, 让我们选择内核:stable diffusion. 我已经创建
kernel stable diffusion i already created it
好了.
kernel stable diffusion i already created it
在我的仓库中, 你还会找到运行这个程序所需安装的要求.
:in. my. repository you will also find the requirements that you need to install in
所以模型加载器, 管道剥离, 导入图像. 这是从python order. to. run this so let'simport everything we need so the model loader the pipeline
所以模型加载器, 管道剥离, 导入图像. 这是从python peel import
所以模型加载器, 管道剥离, 导入图像. 这是从python image. this is how to load the image from python so pat lib import actually this one we
加载图像的方式. 所以pat lib导入实际上是这个. 我们不
image. this is how to load the image from python so pat lib import actually this. one. we
加载图像的方式. 月 所以pat lib 导入实际上是这个. 我们不
don't need transformers
需要transformers.
don't need transformers
这是我们将使用的唯一库, 因为它是clip 的tokenizer.
o This. is. the only library that we will be using because it's the tokenizer of. the
这是我们将使用的唯一库, 因为它是clip 的tokenizer.
clip.
所以在将文本发送到clip 嵌入之前, 如何将其tokenize 为tokens.
: So. how. to tokenize the text into tokens before sending it to the clip embeddings.
否则, 我们还需要构建tokenizer, 这真的是很多工作.
Otherwise, we also need to build the tokenizer and it's really a lot of job.
我不允许 CU DA, 也不允许 MPS, 但如果你想使用 CUDA或 MPS
:ol. don tallow CUDA and I also don't allow MPs, but you can activate these two
我不允许 CU DA, 也不允许 MPS, 但如果你想使用 CUDA或 MPS
variables if you want to use CUDA or MPs.
你可以激活这两个变量
variables if you want to use CUDA or MPS.
如果可用并允许 CUDA, 那么设备当然会变成 CUDA.
Available and allow CUDA, then the device becomes CUDA, of course.
然后我们打印我们正在使用的设备. 好的, 让我们加载tokenizer.
v and. then we print the device we are using okay let'sload the tokenizer tokenizer. is
tokenizer 是clip tokenizer.
:o the clip tokenizer we need to tell him what is the vocabulary file so which. is
我们需要告诉他词汇文件是什么, 它已经保存在这里的data o the clip tokenizer we need to tell him what is the vocabulary file so which is
我们需要告诉他词汇文件是什么, 它已经保存在这里的data already saved
我们需要告诉他词汇文件是什么, 它已经保存在这里的data here in the data data vocabulary. json and then also the merges file.
data vocabulary json 中, 然后还有合并文件.
here in the data data vocabulary. json and then also the merges file.
也许有一天我会做一个关于tokenizer 如何工作的视频, 这样我们
may be. one. day i will make a video on how the tokenizer works so we can build. also. thee
也可以构建tokenizer, 但这需要很多时间, 我的意思是, 它与
oc tokenizer but this is something that requires a lot of time i mean and it's. not
也可以构建tokenizer, 但这需要很多时间, 我的意思是, 它与
really
扩散模型没有直接关系, 这就是为什么我不想构建它的原因,
related. to. the diffusion model so that's why i didn't want to build it the model file
模型文件是. 我将使用数据, 然后是这里的这个文件.
w related. to. the diffusion model so that's why i didn't want to build it the model file
然后我们加载模型. 所以模型是:model loader.
wisi will. use the data and then this file here then we load the model so the. models
然后我们加载模型. 所以模型是:model loader.
are model loader dot preload model from
将模型从模型文件预加载到我们选择的设备中. 好的, 让
are model loader dot preload model from
将模型从模型文件预加载到我们选择的设备中. 好的, 让
将模型从模型文件预加载到我们选择的设备中. 好的, 让
model file. into this device that we have selected okay let'sbuild from text. to. image
例如, 我想要一只猫坐在或伸展一 比如说伸展
w what we. need to define the prompt for example i want a cat sitting or stretching
例如, 我想要一只猫坐在或伸展一 比如说伸展
let'ssay stretching on the floor
在地板上, 高度详细.
let'ssay stretching on the floor
我们需要创建一个能生成好图像的提示, 所以我们需要添加
We need. to create a prompt that will create a good image so we need to add a. lot. of.
很多细节.
details.
超锐利, 电影感, 等等.
Ultra sharp, cinematic, etc.
8 K分辨率.
8 Kresolution.
无条件提示.
The unconditioned prompt.
我保持空白.
I keep it blank.
这个你也可以用作负面成员, 你可以用作负面提示.
This-you. can also use it as a negative member, you can use it as a negative prompt..
所以如果你不希望输出具有某些特征, 可以在负面提示中定义它.
So. if you. don't want the output to have some characteristics, you can define. it in.
所以如果你不希望输出具有某些特征, 可以在负面提示中定义它.
the negative prompt.
当然, 我喜欢做cfg, 也就是无分类器引导, 我们将其设置
Of. course, I like to do cfg, so the classifier-free guidance, which we set to. true.
cfg-scale 是一个介于1到14之间的数字, 表示我们
cfg-scale. is. a number between 1and 14, which indicates how much attention we want
希望模型对这个提示给予多少关注.
cfg-scale. is. a number between 1and 14, which indicates how much attention. we want
希望模型对这个提示给予多少关注.
the model to pay to this prompt.
14意味着非常关注, 而1意味着很少关注
39. means pay very much attention, or1 means pay very little attention
我使用7.
Iuse7.
然后我们还可以定义图像到图像的参数.
Then we can define also the parameters for image-to-image.
所以输入图像等于无. 图像路径等于. 我会用我的狗的图像
so input. image is equal to none image path is equal to iwill define it with my. image
来定义它, 我已经在这里有了, 但现在我不想加载它.
of. the dog which i already have here and but for now i don't want to load it so. if we
所以如果我们想加载它, 我们需要做. 输入.
want to
所以如果我们想加载它, 我们需要做. 输入.
load. it we need to do input image is equal to image. open image button. but for. now. i.
图像等于image open 图像按钮, 但现在我不会使用它. 所以现在
load. it we need to do input image is equal to image. open image button but for. now. i
图像等于image open 图像按钮, 但现在我不会使用它. 所以现在
will
让我们注释它, 如果我们使用它, 我们需要定义强度. 也
让我们注释它, 如果我们使用它, 我们需要定义强度. 也
d will. not. use. it. so. now. let's. comment it and if we use it we need to. define. the
我们将使用的采样器, 当然, 是我们唯一有的, 就是 D DPM.
the sampler
我们将使用的采样器, 当然, 是我们唯一有的, 就是 DDP M.
ocwe wil. be using of course is the only one we have is the D DPM the number. of
推理步骤的数量等于50, 种子等于42, 因为这是一个幸运
we will. be using of course is the only one we have is the D DPM the number. of
数字, 至少在一些书上是这么说的.
:oc inference steps is equal to 50andthe seed isequalto42because it's a lucky
数字, 至少在一些书上是这么说的.
number at least according to some books
输出图像等于pipeline 生成. 好的, 提示是我们定义的提示.
the :output image is equal to pipeline generate okay the prompt is the prompt. that. we
无条件提示是我们定义的无条件提示.
have. defined the unconditioned prompt is the unconditioned prompt. that we. have.
无条件提示是我们定义的无条件提示.
defined input
输入图像是我们已经定义的输入图像.
defined input
如果不是注释的, 当然, 图像的强度和 CFG 比例是我们
image is. the input image that we have defined if it's not commented of. course the
如果不是注释的, 当然, 图像的强度和 CFG 比例是我们
strength for the image
如果不是注释的, 当然, 图像的强度和 CFG 比例是我们
and the CEG. scale is the one we have defined the sampler name is the sampler. name we
定义的. 采样器名称是我们定义的采样器名称. 推理步骤的
and the CEG. scale is the one we have defined the sampler name is the sampler. name we
数量是推理步骤的数量. 种子模型设备.
have. defined the number of inference steps is the number of inference steps. the seed
数量是推理步骤的数量. 种子模型设备.
models device
idle Device 是我们的 CPU, 所以当我们不想使用某个东西时, 我们将
idle Device. is our CPU so when we don't want to use something we move it. to. the CPU
其移到 CPU 上, 而tokenizer 是tokenizer, 然后是image from Array 输出图像
idle Device is our CPU so when we don't want to use something we move. it. to. the CPU u
如果一切都做得很好, 如果所有代码都写得正确, 你总是
and the tokenizer is the tokenizer and then image. from Array output lm age. if everything
如果一切都做得很好一如果所有代码都写得正确, 你总是
is done
如果一切都做得很好一如果所有代码都写得正确, 你总是
well. if. all the code has been written correctly you can always go back. to. my
可以回到我的仓库并下载代码.
ec well if. all the code has been written correctly you can always go back. to. my.
可以回到我的仓库并下载代码
如果你不想自己写 代码 让我们运行代码, 看看结果是
如果你不想自己写代码, 让我们运行代码, 看看结果是
let's. run. the code and let's'see what is the result my computer will take. a while so
什么. 我的电脑需要一些时间, 月 所以需要一些时间, 让
4:oit will. take some time so let'srun it so if we run the code it will generate. an.
it will take some. time. so let'srun it so if we run the code it will generate an 我们运行它.
所以如果我们运行代码, 它将根据我的电脑中的提示生成一张
it will take some. time. so. let'srun it so if we run the code it will generate an
它花了很长时间, 所以我剪掉了视频, 实际上我已经用我从
according to our prompt in my computer it took really a long time so I cut the video
它花了很长时间, 所以我剪掉了视频, 实际上我已经用我从
git hub 上替换的代码替换了代码, 因为现在我想在不向你展示
and i actually already. replaced the code with the one from my git hub because now i
所有代码的情况下实际向你解释代码.
want to actually explain you the code without while showing you all the code together
它是如何工作的? 所以现在我们只使用提示生成了一个图像
how does
它是如何工作的? 所以现在我们只使用提示生成了一个图像
it work so now we we generated an image using only the prompt
Ln ference _it epa _ Inference _at epe,
我使用 CPU. 这就是为什么它非常慢, 因为我的 GPU 不够
I use the CPu, that's why its very slow because my GPu is not powerful enough and we
强大, 我们设置了一个无条件提示为零.
I use the CPu, that's why its very slow because my GPu is not powerful enough and we
强大, 我们设置了一个无条件提示为零.
set'an unconditional prompt to zero.
strength e strength,
_ Ln ference _step u
ce_it eps
我们正在使用无分类器引导, 比例为七.
We are using the. classifier free guidance and with a scale of seven.
77s/it
所以让我们进入管道 看看会发生什么.
So let'sgo in the pipeline and let's see what happens.
所以基本上, 因为我们正在做无分类器引导, 我们将生成
So basically, because we are doing. the classifier free guidance, we will generate two
两个条件信号 个带有提示 一个带有空文本, 即
So basically, because we. are doing. the. classifier free. guidance, we will generate two
无条件提示
"unconditional. prompt.
vice
ra rater )
这也被称为负面提示.
which is also'called the negative prompt.
input _in age
This will result in a'batch size'of two'that wilt run through the unit.
所以让我们回到
" So-let
所以让我们回到这里.
So let's go back to here.
假设我们正在做文本到图像的转换.
Suppose we are doing text to image.
所以现在我们的单元同时处理两个潜在变量, 因为我们设置了
So now our unit has two la tents that he's doing a t tlhe same time because we have the
批量大小为二.
batch size equal to two.
并且对于每一个, 它都在预测噪声水平.
And for each of them, it is predicting the noise level.
但是我们如何从初始噪声中去除预测的噪声呢? 所以, 因为
but how can we remove this noise from the predicted noise from the initial noise so
要生成图像, 我们从随机噪声和提示开始.
because to generate an image we start from random noise and the prompt Jmu ar Jami
最初我们用 VAE 对其进行编码, 所以它变成了一个潜在
initially we encode it with our WAE so it becomes a latent which is still noise and
变量, 仍然是噪声, 通过单元我们预测它根据时间表有多少
ni tially we encode it with our VAE so it becomes a latent which is still noise and
所以根据我们将要进行的50步推理, 第一步将是1000, 下
with the unit we predict l how much noise is it according to a schedule so according ta
所以根据我们将要进行的50步推理, 第一步将是1000, 下
50steps
一步将是980, 下一步将是960, 等等.
of inferencing that we will be doing at the beginning the first step will be 1ooo the
步将是980, 下一步将是960, 等等.
next step will be 9s0 the nent step will be 960 etc so this time will change
所以这个时间将根据这个时间表变化、所以在第50步时,
next step will be 9s0 the next step will be 960 etc so this time will change
我们处于时间步0
so that at the Soth step we are at the time step 0.
那么我们如何通过预测的噪声, 进入下一个潜在变量, 从而
and l how can we then with the predicted noise go to the nest latent so we remove this
好吧, 我们用采样器来做, 特别是用采样器的步骤方法
noise that was predicted by the unit well we de it with the sampler and in particular
好吧, 我们用采样器来做, 特别是用采样器的步骤方法
.. we do it "
好吧, 我们用采样器来做特别是用采样器的步骤方法
with the sample method of the sampler step method sorry of the sampler which
基本上会根据这里的公式七计算给定当前样本的前一个样本
with the sample method of. the sampler step method sorry of the sampler which
Monte Carlo estimates.
calculated in a Rao-Blackwell ized fashion with closed form expressions instead of high variance 基本止会根据这里的公武七计算给定当前样本的前一个样本
when. conditioned on xo :
basically Wiicdleblote the previous sample given the Current sample according to Com sequen dy. all KL divergences so they can be
Monte Carlo estimates.
calculated in a Rao-Blackwell ized fashion with closed form expressions instead of high variance 3所以基本上是给定当前样本和预测的x0计算较少噪声的
cheformulpupbegyeabereo whic dh basically a lau latest he pr mio us sample
Monte Carlo estimates.
calculated in a Rao-Blackwell ized fashion with closed form expressions instead of high variance 3样本gio所以这不是x0sn因为我们没有x0, 所以我们没有
We ignore the fact that the forward process variances are learnable by re parameterization and instead fix them to constants (see Section for details ). Thus. in our implementation, the approximate posterior q has no lear mable parameters, so L is a constant during training and can be ignored.
40. 2 Reverse process and L1: T-1
We ignore the fact that the forward process variances βare learnable by re parameterization and instead fix them to constants (see Section for details ). Thus. in our implementation, the approximate 3. 2 Revers Apothen way of. denoising is to de the sampling like this.
41. 2 Reverse process and Ljr Now we discuss 男种去噪的方法是像这样进行栗样g L< IF
econdisopim
Hhes
which resembles denoising score m 一种去噪的方法
(12)
is cqual to (c Another way of denoising is to do the sampling liked his ). wc sec
Exo.
(12)
is equal to (one term of ) the variational bound for the Langevin-like reverse process ). we see which resembles denoising score matching over multiple noise scales indexed by t [55]. As Eq. )
that optimizing an objective resembling denoising score matching is c qui valent to using variational
我实际上是这箱
12
最本的话
(12 )
which resembles de nosing score match ger up leo isescalesndexed By Ss]. As Eq. 2)
Exo.
(12)
is equal to (one term of ) the variational bound for the Langevin-like reverse process ). we see which resembles denoising score matching over multiple noise scales indexed by t[55]. As Eq. 2)
that optimizing an objective resembling denoising score matching is c qui valent to using variational
这就是我们如何去除噪声以获得较少噪声的服本
(12
andsthis is show we remover the noise to rgetalesgnoisyyversiop soonce. we get the which resembles de nosing score mai ching over m up le noise scales ndexedbyss As Eq.(2)
that optimizing an objective resembling denoising score matching is c qui valent to using variational
我们得到了 较少碾声的片
and s this is how we remove the nois which semble s denoising s cofe match in se to get a less noisy version once we get that optimizing an objective resembling d
我们得到了较少噪声的版本, 我们就继续这个过程, 直到
less noisy version we lkeep doing this process until there is no more noise so we are
所以我们处于时间步0, 在这个时间步我们没有更多的噪声,
less noisy version we lkeep doing this process until there is no more noise so we are
所以我们处于时间步0, 在这个时间步我们没有更多的噪声.
at the time
我们将这个潜在变量提供给解码器, 它将把它转换成图像.
step zero in which we have no more noise we give this latent to the decoder which
这就是文本到图像的工作方式, 男一边是图像到图像. 所以
will turn a it into an image this is how the text to image works the image to image on
这就是文本到图像的工作方式, 男一边是图像到图像. 所以
the other
所以要做图像到图像, 我们需要到这里, 并取消注释这里的
side so let'stry to do the image to image so to do the image to image we need to
所以要做图像到图像. 我们需要到这里 并取消注释这里的
scale, 代码.
go here and we uncomment this code here.
. sca ie,
Inference _at eps,
这使我们能够从狗开始, 然后给出一些提示.
This allows us to start with the dog and then give, for example, some prompt.
strength a strength,
例如我们想要这里的这只狗
E or example, we want this dog here.
treng tha strength
我们想说, 好吧, 我们想要一只在地板上伸展的狗, 细节
We want to say, okay, we want a dog
我们想说, 好吧, 我们想要一只在地板上伸展的狗, 细节
stretching on the floor. highly. detailed etc we can run it i will not run it because
stretching on the floor. highly. detailed etc we can run it i will not run it because 丰富, 等等. 我们可以运行它. 我不会运行它, 因为
它会再花五分钟.
it will take another five minutes and if we do this we can set a strength of let's
如果我们这样做一我们可以设置一个强度, 比如说0. 6, 这
it will take another five minutes and if we do this we can set a strength of let's
如果我们这样做, 我们可以设置一个强度, 比如说0. 6, 这
say0. 6
意味着让我们到这里. 所以我们设置了一个0. 6的强度
which means that let'sgo here so we set a strength of 0. 6sowe have this input
意味着让我们到这里 以我们设 个0 的强度
which means that let'sgo here so have this inp
ut
所以我们有这个输入图像. 0. 6的强度意味着我们将用变分自
which means that let's go here so we set a strength of 0. 6 so we have this input
编码器对其进行编码.
image strength of 0. 6 means that we will add we will encode it with the devi rational
编码器对其进行编码.
auto encoder
我们将变成一个潜在变量, 我们将添加一些噪声, 但多少
we'll become a latent we'll add some noise but how much noise not
不是所有的噪声, 所以它不会完全变成噪声, 但比那少一些
we'll become a latent well add some noise but how much noise not
噪声.
all the noise so that it becomes completely noise, but less noise than that.
所以比如说60
So let's say 60% noise is not really true because it depends on the schedule.
在我们的例子中它是线性的, 所以可以认为是60 In our case it's linear, se it can be considered 60% ef noise.
然后我们将这个图像提供给调度器, 它不会从第1000步
we then give this image to the scheduler which will start not from the loo0 step it
所以如果我们把强度设置为0. 6, 它将从第600步开始
villstart before so if wve set the strength to @. s it will start from the G00 step
然后每步移动20, 继续600, 然后580. 然后560、然后540
villstart before so if ve set the strength to @. s it will start from the G00 step
然后每步移动20, 继续600, 然后580, 然后560, 然后540,
and then
等等, 直到达到20
move by 20 will keep going 600 then 5s0 then 560 then 540 etc until it reaches 20.
所以总共它会做更少的步骤, 因为我们从一个较少噪声的例子
so in total it will do less steps because we start from a less no is y esxample but at
的自由度来改变图像, 因为我们已经有了图像, 所以它不能
the same time because we start with less noise the unit also has less freedom to
的自由度来改变图像, 国 因为我们已经有了图像, 所以它不能
alter the
的自由度来改变图像, 因为我们已经有了图像, 所以它不能
image because he already have the image so he can not change it toe much so how do you
改变太多. 那么你如何调整噪声水平呢?
image because he already have the image so he can net change it toe much so how de you
改变大多. 那么你如何调整噪声水平呢?
adjust the noise level?
这取决于你是否希望模型非常注意输入图像并且不要改变大多
depends if you want the unit to pay very much attention to the input image
那么你就添加较少的噪声.
and not change it too much, d hen you add less noise.
如果你想完全改变原始图像, 那么你可以添加所有可能的
If you want to change completely the original image, then you can add all the
噪声.
pes sible noise.
所以你把强度设置为一
Se you set the strength to one.
这就是图像到图像的工作方式.
And this is hew the image to image works.
我没有实现修复功能.
I didn't implement the in painting.
因为这里的原因是预训练模型, 我们使用的模型并没有针对
Because the reason'is that the pre-trained model here, so the model that we are using
因为这里的原因是预训练模型, 我们使用的模型并没有针对
is not fine-tuned for in-painting.
修复进行微调.
is not fine-tuned for in-painting.
修复进行微调.
所以果你去网站查看模型卡片, 他们会为修复提供男一个
So if you go on the website and you look at the model card, they have another model
具有不同权重的模型.
for in-painting which has different weights.
encoded masked image and 1 for the mask. itself ) whose weights were zero-initia e storing the non in painting checkpoint. During tg 这里, 这个.
25% mask everything,
Here, this one here.
Hardware :32 ×8x A100 GPUs
Examples : 但这个, 这个模型的结构也有点不同、回 为他们在单元中为
ht but'this the structure of this model is also a little different because they have in
Examples : 我当然会在我的仓库中直接实现它, 所以我将修改代码并实现
You cnuus ichth+thetunit-they have five additional input channels for'the mask i will ef course
Examples : 我当然会在我的仓库中直接实现它, 用 所以我将修改代码并实现
implement it
Examples : 我当然会在我的仓库中直接实现它, 所以我将修改代码并实现
Ynmh-jintmjrepository Girectlysoriwill modily the code and also implement the code for
Examples : 修复的代码, 以便我们可以支持这个模型
tuanuuho-jinmjprepository directly so rir will modify the code and also implement the code for
Examples : 但遗憾的是我现在没有时间, 因为在中国这里现在是国庆节
Yocnuhjnr painting so that we can r support this model but unfortu mately i don°t b ave the time
Example s: 但遗憾的是我现在没有时间, 因为在中国这里现在是国庆节
mith the _. Diffuses libary a y ML Gitfub. rtgesitery
365 5pacn
Examples : 但遗憾的是我现在没有时间, 因为在中国这里现在是国庆节
ou can uie this both with because in china-here is guoqjngje and i'm going to lao jia with my my wife so
Examples : 我要和我的妻子回老家, 所以我们时间有点紧张
ht-because in china-lhereisguoqjngjie and i'm going to lao jia with my my wife so
Examples : 我要和我的妻子回老家, 所以我们时间有点紧张
y MLGit Hud. ntpesitery
365 Space i
Examples : 我要和我的妻子回老家, 所以我们时间有点紧张
Ycnuhthwe realiteleshort of time but i hope that with my vide e guys you you got really
Examples : 但我希望我的视频能让你们真正深入了解稳定扩散, 并理解其
Yovcuhhwearealil eshortoftimebut i lhope that with my vide e guys you you got really
Examples : 背后的工作原理, 而不是仅仅使用 Hugging. Face库, 同时也要
Ycmw-jn to'stable diffusion rand your understood what is happening under the hood instead ef
Examples : 背后的工作原理, 而不是仅仅使用 Hugging Face库, 同时也要
eay MLGit Hub. negositey
just using 3655pace
背后的工作原理而不是仅仅使用 Hugging Face 库, 同时也要
the hugging face library and also notice that the model itself is not so um
注意到模型本身并不特别复杂.
particularly sophisticated if you check the decoder and the encoder they are just a
如果你检查解码器和编码器, 它们只是一堆卷积、上采样和
particularly sophisticated if you check the decoder and the encoder they are just a
如果你检查解码器和编码器, 它们只是一堆卷积、上采样和
bunch of
如果你检查解码器和编码器, 它们只是一堆卷积、上采样和
convolutions and up sampling and hor malization s, just like any other computer vision
归
"就像任何其他计算机视觉模型一样.
convolutions and up sampling and hor malization s, just like any other computer vision
归一化, 就像任何其他计算机视觉模型一样.
对于单元也是如此
当然, 他们在如何实现这一点上做出了非常聪明的选择, 对吧?
Of course, there are very smart choices in how they do it, okay?
但这并不是扩散的重要部分.
But that's not the important thing of the diffusion :/
实际上 如果我们研究像分数模型这样的扩散模型, 你会
And actually, if we study the diffusion models like score models, you will see that
实际上, 如果我们研究像分数模型这样的扩散模型, 你会
it doesn't even matter the structure of the model :
只要模型具有表达能力, 它实际上会以相同的方式学习分数
As long as the model is expressive, it will actually t earn the score function in the
函数
M/8, k/ 但在这段视频中:我们不讨论这个.
我会在未来的视频中讨论分数模型.
I will talk about score model in future videos.
我希望你能理解的是, 这一切机制是如何协同工作的, 我们
What I want you to understand is that how this all mechanism works together so how
如何学习一个预测噪声的模型, 然后生成图像, 让我再重申
can we just learn a model that predicts the noise and then we come up with images and
如何学习一个预测噪声的模型, 然后生成图像, 让我再重申
Let me rehearse again the idea./
如何学习 个预测噪声的模型 成图 上我算重申
Let me re again idea
十一下这个想法:
Let me rehearse again the idea.
所以我们从训练一个模型开始! 这个模型需要学习一个
So we started by training a modet and
conv1x1 所以我们从训练一个模型开始, 这个模型需要学习一个
that needs'to learn a'probability distribution as you remember p of theta here we we
概率分布, 如你所记, poftheta在这里, 我们无法
q(x|xo) = N(x; √axo,(1
a) I)
(4)
can not learn this one directly because we don°t know how to marginalize here so what
所以我们做的是为这个量找到一个下界, 并最大化这个下界.
q(x|xo) = N(xx;√a;xo,(1
a) I)
(4)
we did is
所以我们做的是为这个量找到一个下界, 并最大化这个下界.
q(x|xo) = N(xx;√a;xo,(1 a) I)
(4)
we find some lower bound for tl his quantity l here and we maximize this lower bound how
损失上运行梯度下降.
do we maximize this lower bound by training a medel by running the gradient descent
损失上运行梯度下降.
on this loss this loss produces a model
这个损失产生了一个允许我们预测噪声的模型.
on this loss this loss produces a model
那么我们如何实际使用这个预测噪声的模型来逆向时间, 去除
that allow us to predict the noise then he w do we actually use this model with the
噪声呢? 因为正向过程我们知道如何进行, 是我们定义的
predicted noise to go back in time with the noise because the forward process we know
噪声呢? 因为正向过程我们知道如何进行, 是我们定义的
how to go
如何添加噪声, 但在逆向时间, 如何去除噪声, 我们不
it's defined by us how to add noise but in back in time so lhow to remove noise we
知道我们按照我在采样器中描述的公式来做
(4 )
don°t know and we do it according to the formulas that i have described in the
(4)
sampler'so'theformuta'number seven
-所以公式7和这个公式, 实际上我们可以使用
(4)
sampler so the formula number seven
(12 )
is equal to (one term of ) the variational bound for the Langevin-like reverse process ). we see
model's variational bound to an objective that resembles denoising score matching. Nonetheless.
and the formula number also this ong actually we can use actually i will show you in ablation wrap r Data scaling. reverse process decoder, and L
实际上我们可以使用
(12 )
and the formula number also. this one r actually. wg. can use-actually i will show you in
Damp les : 实际上, 我会在我的其他视频中展示给你
on wb hy other here ft ave another repository i think it's called python dd pu in which
ddp m论文
E 但使用了这里的算法 所以如果你对这个版本
the dd pm paper but by using, this alg on it hm here so if your are interested in. this 2
mple 降噪感兴趣, 可以查看我的另一
+ Pyhes 199
version of the denoising you can check my other repository here this one dd pm
Exo.
[202a(1-a)
(12)
is equal to (one term of ) the variational bound for the Langevin-like reverse process ). we see which resembles denoising score matching over multiple noise scales indexed by t [55]. As Eq. 2)
that optimizing an objective resembling denoising score matching is c qui valent to using variational
wm ch resembles de nosing score match ng over mulu plenoisescalesndexedby 我还想向你展元 图像修复 如何进行图像到图像的
12
which resenibiesdenosng score malhvermutplehoise scales dexcabys As Eq2 因为当然
(12
Ialso want to. t bank very much many u repositories that ld have s used as al self t-studying
没有自己凭空创造这一切. 我研究了很多论文.
material, because of course I didn't make up all this by myself, I studied a lot ef
没有自己凭案创造这一切, 我研究了很多论文
papers, 1
我在过去几周里阅读了超过30篇论文, 所以这花费了我
read, I thinl, to study'tiese diffusion models I read more than 30 papers in the last
conv 1x1 我在过去几周里阅读了超过30篇论文, 所以这花费了我
few weeks, so'tttoot'mie a lot of time, but I was really passionate about this kind
很多时间独我对这类模塑非常热情, 国为它们很复杂,
un ning it through the forw of models,
What if we co because they re complicated and really like to study ching s that can generate new
What i we coud cm press our data before runing trough the lor ward 南我真的很喜欢研究能生成新事物的模型
ces s( UNet)?
我特别想感谢一些我使用过的资源.
I want to really thank in particular some resources that I lhave used.
让我看看, 这个在这里.
Let me see, this one's bhere.
所以它方代码, 这个 Div am Gupta 的家伙, 这个人的男一个
So the official code, this guy Div am Gupta, this other repository from this person
仓库, 我实际上非常依赖它作为基础, 还有 Hugging Face 的
So the official code, this guy Div am Gupta, this other repository from this person
扩散器库, 我在我的采样器代码中大部分都是基于它的.
here, which I used very much actually as a base, and the diffusers library from this
因为我认为使用它更好, 因为我们实际上只是应用一些公式
Because I think it's better to use, because we are actually just applying some
没有必要从头开始写.
formulas, there is no point in writing it from zero.
重点实际上是理解这些公式在做什么以及我们为什么要这样做.
The point is actually understanding what is happening with these formulas and why we
一如既往, 完整的代码是可用的, 我也会为你们提供所有的
And as usual the full code is available, I will also male all the slides available
我希望, 如果你在中国, 你也能和我一起度过一个美好的
and I hope if you are in China you also have a great l holiday wit lh me and if you*re
假期, 如果你不在, 我希望你和家人朋友以及所有人都能
not in China I hope you b ave a great time with your family and friends and everyone
度过一个愉快的时光.
not in China I hope you have a great time wit lh your family and friends and everyone
度过一个愉快的时光.
else so
所以随时欢迎回到我的频道, 请随时评论, 给我发消息
else so
如果你有不明白的地方或者想让我解释得更清楚, 因为我总是
comment or if you didn't understand something or if you want me to explain something
如果你有不明白的地方或者想让我解释得更清楚, 因为我总是
better because
而且, 伙计们, 我做这个当然不是全职工作, 我是兼职做
I'm alwvays arailable for explanation and guys I do this not as my full-time job of
的, 最近我还在做咨询, 所以我很忙, 但有时我会抽时间
course I do it as a part-time and lately I'm
录制视频, 所以如果你喜欢, 请分享我的频道, 分享我的
doing consulting so I°m very busy but sometimes I take time to record videos and so
视频给其他人, 这样我的频道可以成长, 我有更多的动力
please share my channel, share my vide e with people if you like it and so that my
视频给其他人, 这样我的频道可以成长, 我有更多的动力
channel can
视频给其他人, 这样我的频道可以成长, 我有更多的动力
grow and I have more motivation to kep doing this kind of videos which take really a
继续做这类视频, 这真的需要很多时间, 因为准备这样一个
lot of time because to prepare a video lile this I spend around many we elks ef
我做这个是出于热情, 不是作为工作, 我真的花了很多时间
research but this is okay, I do it as a passion, I don't do it as a job
准备所有的幻灯片, 准备所有的演讲, 准备代码并清理和
And I spend really a lot of time preparing all the slides and preparing all the
注释它等等.
speeches and preparing the code and cleaning it and commenting it, etc, etc.
我总是免费做这些.
I always de it for free.
所以如果你想支持我, 最好的方式就是订阅, 喜欢我的视频
So if you would like to support me, the best way is to subscribe, like my video and
并分享给其他人:
share it with ct her people.
