
前文，我们演示了模型推理方面方面的核心问题，将计算过程或者说推理过程分散到更多的 token 中，而且确实有研究表明，较长的中间思考过程（我们称之为思维链，CoT）有助于提升问题的解决率，如果你经常使用 deepseek-R1 这种推理模型，也的确会发现，它总是喜欢滔滔不绝的思考，想必你应该清楚了其背后的一些原因。

而且我们发现模型其实有很多不擅长的领域，需要借助外部的一些工具才能加以解决，对此，我们想在这里就模型存在智能缺陷的一些点多演示一些内容，其实还是回归到我们这系列科普文写作的最初目地上，消除普通大众对 LLMs 的众多误解或盲目炒作。

LLMs 现在可以在数学、物理、代码等众多代表人类智商顶峰的一些领域中实现近乎超人的智商水平，然而实际上，有很多在我们人类看来非常简单，简单到幼儿园小朋友都可以解决的问题，但模型却不能很好的解决。

让我们一起来看几个例子。

## 十、智能缺陷

首先第一个例子就是计数，模型实际上并不擅长精确计数，原因完全相同。你要求单个 token 承载的信息量太大了。让我给你展示一个简单的例子。

![[pointdemo.png|500]]

（实际上是 126 个）

因此，在单个 token 中，它必须计算其上下文窗口中的点的数量。而且它必须在网络的一次前向传递中完成这一操作。正如我们之前讨论的，在网络的一次前向传递中，能够进行的计算量是有限的。你可以把这想象成那里发生的计算量非常小。

### 重新审视分词：模型在拼写方面表现欠佳

![[pointtokenizer.png|500]]

如果我们把这些点放在分词器里，实际上它只把这些点分成了 4 个 token，第一个 ID 为 43369 的 token 自己就包含了 64 个点，反正出于某种原因，它们就这样被分开了，实际上并不清楚为何会这样，这与分词器的细节有关，但事实证明，模型基本上看到的是这些 token ID，然后从这些 token ID 中，它需要数出总数。

当然我们仍有办法解决这个问题，那就是再次呼叫代码。

![[chatgptpointcount.png|500]]

这里稍微有点小瑕疵，它本身并未执行代码告诉我结果，是我手动点击运行的，不管怎么样，代码统计出了 126。你可能会想，为什么这能行？其实这有点微妙，也有点意思。所以当我说用代码时，我其实觉得这能行。这里的情况是，虽然看起来不像，但我实际上已经把问题分解成了对模型来说更容易处理的小问题。我知道模型不会计数，无法进行心算，但我也知道模型在复制粘贴方面其实相当擅长。

所以当我说“使用 Python 代码统计”时，它会在 Python 中创建一个字符串。而将我的输入复制粘贴到代码中，上图中展示了这一过程，因为对于模型来说，它看到的这个字符串，对它来说只是这 4 个 token 和其他一些文字。因此，模型要复制粘贴这些 token ID 并将它们在这里解包成点是非常简单的。于是它创建了这个字符串，然后调用 Python 的 count 方法，最终得出正确答案。

所以是 Python 解释器在进行计数，而不是模型的心算在计数。这再次说明了一个简单的例子：模型需要 token 来进行思考，不要依赖它们的心算能力。这也是为什么模型在计数方面表现不佳的原因。如果你需要它们执行计数任务，务必让它们借助工具完成。

目前这些模型还存在其他各种细微的认知缺陷，这些就像是技术发展过程中需要留意的瑕疵问题。

![[chatgptbug.png|500]]

再举个例子，这些模型在处理各类拼写相关任务时表现欠佳。他们在这方面并不擅长。其原因是，模型看不到字符，它们看到的是 token，它们的整个世界都围绕着这些由小段文本构成的分词标记运转。因此，它们无法像人眼那样识别字符。于是，连非常基础的字符级任务也常常失败。

举个例子，我输入一个字符串"ubiquitous"，要求它从第一个字符开始，每隔两个字符输出一个（即打印第 1、4、7...个字符）。所以我们从 u 开始，正确答案是 uqts。

（我这里使用的是 API 的方式，但我也在网页版中尝试过这个问题，它可以解决掉，我们不清楚网页上其背后使用的模型版本是什么，以及是否存在一些其他处理等，而且经验表明，各类开源模型都或多或少存在这样的问题，释放出来的模型较其官方网页开放的模型性能要低一些）

![[chatgptbug2.png|500]]

这里的问题关键在于：如果你去查看 TickTokenizer 对 "ubiquitous" 的处理，会发现它被分成了三个token，你和我在看到 "ubiquitous" 这个词时，可以轻松识别出每个字母，因为我们能直观地看到它们。当这个词出现在我们视觉工作记忆中时，我们就能非常容易地定位到每第三个字母，从而完成这个任务。

但模型无法访问单个字母。它们将这些视为三个 token。记住，这些模型是在互联网上从零开始训练的。所有这些 token，本质上模型需要发现有多少不同的字母被压缩进这些不同的 token 中。我们之所以使用 token，主要是出于效率考虑。但我认为很多人都希望完全摒弃 token。就像我们确实应该开发字符级别或字节级别的模型。只不过那样会产生很长的序列，而目前人们还不知道如何处理这种情况。因此，在采用分词机制的情况下，任何拼写任务实际上都不太可能表现得特别好。

近一个非常著名的例子就是 “strawberry” 中有多少个字母 “r”？这个问题多次在网上疯传。现在这些模型基本上都能答对了。它们会说草莓（strawberry）这个单词里有三个 r。但在很长一段时间里，所有最先进的模型都坚称草莓这个单词里只有两个 r。这引起了很多"骚动"。因为这就像是，为什么这些模型如此出色？它们能解出数学奥赛题，却数不清“草莓”这个词里有几个 “r”。至于原因，我想你大概清楚了，至于他们是如何解决掉这个问题的，是真的现在清楚这个单词里有几个 r 了吗，不，也许只是像“你是由谁创作的” 一样，通过硬编码的形式塞进去的，因为你随便换一个单词，它还是无法回答对。

如果你愿意自己体验一下，你可以随便输入一段文本，让模型翻转，逆序输出，这其实还是字符级的任务，留给你自己去尝试体验，模型还是不能很好的处理该问题，总之这些问题都需要借助外部工具才能得以解决。

我们稍微总结一下这些现象，首先，模型看不到字符，它们看到的是 token。其次，它们不太擅长计数。所以我们这里是把识别字符的困难和计数的困难结合在了一起。这就是为什么模型在这方面遇到了困难。说实话，我觉得到现在为止，OpenAI 可能已经对这个答案进行了硬编码，或者我不确定他们具体做了什么。但现在这个特定的查询已经可以正常工作了。所以模型在拼写方面也表现不佳。还有很多其他小问题。我就不一一列举了。我只是想举几个例子，让你了解需要注意的地方。在实际使用这些模型时，我并不打算在这里全面分析模型存在的各种不足。我只是想指出，这里确实存在一些不够完善的地方。

#### 智商缺陷

上面的这些问题，是由于我们采用了 token 的训练机制，这是一种来自设计层面的缺陷，或者可以称之为先天性缺陷，这不能把失败的原因归咎到模型上，而是我们人类的设计存在不完善的地方，但也有一些问题，看起来就像是后天性智商缺陷，即使你深入了解这些模型的工作原理，这些问题也会让人摸不着头脑，最近就有一个很好的例子。

**9.11 对比 9.9 谁更大？**

![[98911.png|600]]

这张图是我较早一些时间测试过程中的截图，左侧是 deepseek-r1，右侧是 gpt4o 和 个gpt-o1-mini 两个版本的回答。更有意思的是，deepseek 把这个案例以及上面 “strawberry” 有有几个 r 的案例写进了官方文档教程示例中。

![[deepseektutorial.png|500]]


这些模型对于像这样非常简单的问题并不擅长。这让很多人感到震惊，因为它们能解决复杂的数学问题。它们回答博士级别的物理、化学、生物学问题等。但有时在像这样超级简单的问题上却会出错。

![[911big99.png|500]]

如果你多问几次，模型有时候能答对，有时候就答不对，甚至是一些较强的推理模型上也存在这个问题，这着实有点令人费解，网络上存在很多这样的讨论以及相关研究，但该问题目前依然存在，而且还有一些类似这样零散的问题。

总之这就是为什么我们要实事求是地看待它——一个既神奇又不可全信的随机系统。你应该把它当作工具来使用，而不是放任它随意解决问题然后直接复制粘贴结果。

