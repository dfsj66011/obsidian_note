
你好，我叫安德烈，从事深度神经网络训练已有十多年。在这节课中，我想向大家展示神经网络训练的内部运作原理。具体来说，我们将从一个空白的 Jupyter 笔记本开始。

在这节课结束时，我们将定义并训练一个神经网络，你将有机会深入了解其内部运作机制，直观感受它是如何运作的。具体来说，我想带你们一步步构建 micrograd。micrograd 是我大约两年前在 GitHub 上发布的一个库。

但当时我只上传了源代码，你得自己深入研究它的工作原理。所以在这节课上，我会一步步带你梳理，并对各个部分进行讲解。

## micrograd 概览

那么，什么是 micrograd？它为什么有趣？micrograd 本质上是一个自动微分引擎。Autograd 是自动梯度（automatic gradient）的缩写。它实际上实现了反向传播算法。反向传播是一种能够高效计算神经网络权重相对于某种损失函数梯度的算法。

这样一来，我们就能通过迭代调整神经网络各层权值，使损失函数最小化，从而提升网络预测精度。反向传播算法正是现代深度学习框架（如 PyTorch）的数学核心所在。要理解 micrograd 的功能特性，我认为通过具体案例演示最为直观。

那么如果我们往下滚动，你会看到 micrograd 基本上允许你构建数学表达式。这里我们正在构建一个表达式，其中有两个输入 A 和 B。你会看到 A 和 B 分别是 -4 和 2，但我们会把这些值封装进这个 Value 对象中，作为 micrograd 的一部分来构建。这个 Value 对象会将这些数字本身封装起来。

然后我们将在这里构建一个数学表达式，其中 a 和 b 会被转换为 c、d，最终转化为 e、f 和 g。我将展示 micrograd 的部分功能及其支持的操作。比如你可以对两个值对象进行加法、乘法运算，可以将它们提升为常数幂，可以偏移 1 个单位，取负值，在零处压缩，平方，除以常数，或者相互除等等。现在我们正用这两个输入 a 和 b 构建一个表达式图，并生成输出值 g。micrograd会在后台自动构建出这个完整的数学表达式。

例如，它将知道 c 也是一个值，c 是加法运算的结果，而 c 的子节点是 a 和 b，因为它会维护指向 a 和 b 值对象的指针。因此，它基本上能准确掌握所有这些是如何布局的。这样一来，我们不仅能进行所谓的正向传递（即实际查看 g 的值，这当然相当直接，我们会通过 `.data` 属性来访问它）。

因此，前向传播的输出，即 g 的值，结果是 24.7。但关键在于，我们还可以对这个 g 值对象调用 `.backward` 方法。这基本上会在节点 g 处初始化反向传播。反向传播的作用是从 g 开始，沿着表达式图向后回溯，递归地应用微积分中的链式法则。这样我们就能计算 g 相对于所有内部节点（如 e、d 和 c）以及输入 a 和 b 的导数。例如，我们可以查询 g 相对于 a 的导数，即 `a.grad`。在这个例子中，它恰好是 138，g 对 b 的导数，也就是这里的 645。

这个导数我们稍后会看到非常重要，因为它告诉我们 a 和 b 是如何通过这个数学表达式影响 g 的。具体来说，a 的梯度是 138。所以如果我们稍微调整 a，使其略微增大，138 告诉我们 g 将会增长，而增长的斜率将是 138。而 b 的增长斜率将是 645。这将告诉我们，如果 a 和 b 在正向方向上微调一点点，g 将如何响应。

好的？现在，你可能会对我们在这里构建的这个表达式感到困惑。顺便说一下，这个表达式完全没有意义。我只是随便编的。我只是在炫耀一下 micrograd 支持的各种运算。我们真正关心的是神经网络。但事实证明，神经网络其实也就是数学表达式，就像这个一样，甚至可能还没这么复杂呢。

神经网络只是一个数学表达式。它们将输入数据作为输入，并将神经网络的权重作为输入。它是一个数学表达式。而输出则是你的神经网络预测结果，或者说损失函数。我们稍后会看到这一点。但基本上，神经网络恰好是某一类数学表达式。

但反向传播实际上要通用得多。它其实根本不在乎神经网络。它只关心任意的数学表达式。然后我们恰好利用这套机制来训练神经网络。现在，我想在此补充说明的是，正如你们所见，micrograd 是一个标量值自动求导引擎。它工作在单个标量的层面上，比如 -4 和 2。我们将神经网络层层拆解，一直分解到这些最基本的标量单元，以及所有微小的加法和乘法运算。

这实在是太过分了。显然，在生产环境中你绝不会做任何这类操作。这么做纯粹是出于教学目的，因为它能让我们不必处理那些在现代深度神经网络库中会用到的 $n$ 维张量。所以这样做是为了让你理解并重构反向传播和链式法则，以及理解神经网络的训练过程。然后，如果你真的想训练更大的网络，就必须使用这些张量。但数学原理没有任何变化。这样做纯粹是为了提高效率。我们基本上是把所有的标量值打包成张量，这些张量其实就是这些标量的数组。

由于我们拥有这些大型数组，我们可以对这些数组进行操作，从而充分利用计算机的并行处理能力。所有这些操作都可以并行执行，从而使整个程序运行得更快。但实际上，数学的本质并未改变。这些做法纯粹是为了提高效率。因此，我认为从零开始学习张量在教学方法上并不实用。

这就是我编写 micrograd 的根本原因，因为你可以从基础层面理解事物是如何运作的。之后你可以再对它进行加速。好了，接下来就是有趣的部分了。

我的观点是，micrograd 就是训练神经网络所需的一切，其他都只是效率问题。所以你可能以为 micrograd 会是一段非常复杂的代码。但事实证明并非如此。如果我们直接进入 micrograd，你会看到这里只有两个文件。这就是实际的引擎。它对神经网络一无所知。

这就是构建在 micrograd 之上的完整神经网络库。包括 engine.py 和 nn.py 两部分。实际实现反向传播自动微分的引擎，赋予你神经网络能力的核心代码，仅用 100 行极其简洁的 Python 就完成了——这堂课结束前我们就能完全掌握它。而基于这个自动微分引擎构建的 nn.py 神经网络库，简直简单得像个玩笑。

就像这样，我们得先定义什么是神经元。接着，我们要定义什么是神经元层。然后我们才能定义什么是多层感知机，它其实就是一连串的神经元层。所以这简直是个天大的笑话。说白了，区区 150 行代码就能产生巨大的威力。你只需要理解这些就能掌握神经网络训练的核心，其他一切都不过是效率问题。当然，效率方面还有很多值得探讨的地方。但从根本上说，这就是正在发生的一切。

好了，现在让我们直接开始，一步步实现微梯度（micrograd）。

## 一个单输入简单函数的导数

当然，效率方面还有很多要考虑的。但归根结底，这就是全部了。好了，现在让我们直接深入，一步步实现 micrograd。

首先，我想确保你们对导数有一个非常直观的理解，并清楚它具体提供了哪些信息。让我们从一些基本的导入开始，这些代码我每次在 Jupyter Notebook 中都会复制粘贴。

```python
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


然后，我们来定义一个标量值函数 $$f(x)=3x^2-4x+5$$

```python
def f(x):
    return 3*x**2 - 4*x + 5
```

所以我只是随便编了这个函数。我只是想要一个标量值函数，它接收一个标量 $x$ 并返回单个标量 $y$。当然我们可以调用这个函数，比如传入 $3.0$ 就能得到 $20$。现在，我们还可以绘制这个函数的图像来了解它的形状。

```python
f(3.0)
# 20.0
```

从数学表达式可以看出，这很可能是一条抛物线。这是一个二次函数。如果我们创建一个标量值集合，比如使用从 $-5$ 到 $5$、步长为 $0.25$ 的范围作为输入。也就是说，$x$ 值从 $-5$ 到 $5$（不包括 $5$），步长为$0.25$。

```python
xs = np.arange(-5, 5, 0.25)

# array([-5.  , -4.75, -4.5 , -4.25, -4.  , -3.75, -3.5 , -3.25, -3.  ,
#       -2.75, -2.5 , -2.25, -2.  , -1.75, -1.5 , -1.25, -1.  , -0.75,
#       -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,
#        1.75,  2.  ,  2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,
#        4.  ,  4.25,  4.5 ,  4.75])
```

我们也可以直接对这个 NumPy 数组调用这个函数。因此，如果我们对 `xs` 调用 $f$ 函数，就会得到一组 $y$ 值。这些 $y$ 值本质上也是独立地对每个元素应用函数的结果。

```python
ys = f(xs)
# array([100.    ,  91.6875,  83.75  ,  76.1875,  69.    ,  62.1875,
#        55.75  ,  49.6875,  44.    ,  38.6875,  33.75  ,  29.1875,
#        25.    ,  21.1875,  17.75  ,  14.6875,  12.    ,   9.6875,
#         7.75  ,   6.1875,   5.    ,   4.1875,   3.75  ,   3.6875,
#         26.    ,   4.6875,   5.75  ,   7.1875,   9.    ,  11.1875,
#        13.75  ,  16.6875,  20.    ,  23.6875,  27.75  ,  32.1875,
#        27.    ,  42.1875,  47.75  ,  53.6875])
```

我们可以用 `matplotlib` 来绘制这个结果。

```python
plt.plot(xs, ys)
```

![|350](data:image/png;base64,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)

所以 `plt.plot` 绘制 $x$ 和 $y$，我们得到一个漂亮的抛物线。

现在我们思考一下这个函数在任一 $x$ 处的导数是多少？如果你还记得微积分课，你可能已经推导过导数。

于是我们看到这个数学表达式后，你会把它写在纸上，然后运用乘积法则和其他所有规则，推导出原函数导数的数学表达式。之后你可以代入不同的 $x$ 值来观察导数是多少。不过我们实际上不会这么做，因为在神经网络领域，根本没人会去写出神经网络的数学表达式。这将是一个庞大的表达式。它会有成千上万个项。当然，实际上没有人会去求这个导数。

因此，我们不会采用这种符号化的方法。相反，我想做的是仔细看看导数的定义，确保我们真正理解导数在衡量什么，它告诉我们关于函数的哪些信息。如果我们直接查阅导数的定义，会发现这并不是一个很好的定义。

$$L = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}
$$

【维基】

这是关于可微性的定义。但如果你还记得微积分中的内容，它就是当 $h$ 趋近于 $0$ 时，$[f(x+h) - f(x)]/h$ 的极限。基本上它表达的意思是：如果你在某一点 $x$（或 $a$）处稍微增加一个很小的数 $h$，函数会如何响应？它的响应灵敏度是多少？该点的斜率是多少？函数是上升还是下降？变化幅度有多大？这就是该函数在该点的斜率，即响应的斜率。因此，我们可以通过取一个非常小的 $h$ 来数值计算这里的导数。当然，定义要求我们让 $h$ 趋近于 $0$。我们只需要选取一个非常小的 $h$，比如 $0.001$。假设我们关注的点是 $3.0$。那么我们可以把 $f(x)$ 看作 20。

现在来看 $f(x + h)$，如果我们稍微向正方向推动 $x$，函数会如何响应？仅从这一点来看，你预计 $f(x + h)$ 会略大于 $20$，还是略小于 $20$？由于这里的 $3$ 和 $20$ 的存在，如果我们稍微向正方向移动，函数会正向响应。因此，你会预期这个值会略大于 $20$。而这个差值的大小则告诉你斜率的强度，也就是斜率的大小。

所以 $f(x+h)$ 减去 $f(x)$，这就是函数在正向的响应量。我们需要用横轴变化量来归一化。因此我们用纵轴变化量除以横轴变化量来得到斜率。

```python
```python
h = 0.001
x = 3.0
f(x)
# 20.0
f(x+h)
# 20.014003000000002
f(x+h) - f(x)
# 0.01400300000000243
(f(x+h) - f(x)) / h
# 14.00300000000243
```

当然，这只是斜率的数值近似，因为我们必须让 $h$ 非常非常小才能收敛到精确值。但如果用了太多零，在某些情况下会得到错误答案，因为我们使用的是浮点运算，而所有这些数字在计算机内存中的表示都是有限的，到某个临界点就会出现问题。不过通过这种方法，我们能够逐步逼近正确答案。

```python
h = 0.00000001
x = 3.0
(f(x+h) - f(x)) / h
# 14.00000009255109

h = 0.00000000001
x = 3.0
(f(x+h) - f(x)) / h
# 14.000178794049134

h = 0.0000000000000000001
x = 3.0
(f(x+h) - f(x)) / h
# 0.0
```

但基本上，在 $x=3$ 时，斜率是 $14$。你可以通过计算 $3x² - 4x + 5$ 的导数来验证这一点。导数是 $6x-4$，然后代入 $x=3$。所以 $18-4=14$。

所以这是正确的。那么在 $3$ 点处的斜率就是这样。那么，在 $-3$ 点处的斜率呢？你预计斜率会是多少？现在，要说出确切的值确实很难，但这个斜率的符号是什么呢？在 $-3$ 点处，如果我们稍微往 $x$ 的正方向移动一点，函数实际上会下降。这就告诉你斜率会是负的，所以我们会得到一个略低于 20 的数字。

```python
h = 0.00000001
x = -3.0
(f(x+h) - f(x)) / h
# -22.00000039920269
```

因此，如果我们计算斜率，我们预计会得到一个负值，大约是 $-22$。在某个点上，斜率当然会是 $0$。对于这个特定的函数，我之前查过，这个点出现在 $2/3$ 处。所以大约在 $2/3$ 的位置，也就是这里的某个地方，这个导数会是 $0$。基本上，在那个精确的点上，如果我们往正方向稍微推动一下，函数不会有任何反应。

```python
h = 0.000001
x = 2/3
(f(x+h) - f(x)) / h
# 2.999378523327323e-06
```

这几乎保持不变。因此，斜率为 0。好的，现在我们来看一个稍微复杂一些的情况。我们要开始，你知道的，稍微复杂化一点。

```python
# les get more complex
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
print(d)
# 4
```

现在我们有一个函数，其输出变量 $d$ 是三个标量输入 $a$、$b$ 和 $c$ 的函数。$a$、$b$ 和 $c$ 是一些具体的值，作为我们表达式图的三个输入，而 $d$ 是单一输出。如果我们直接打印 $d$，会得到 $4$。现在我想再次查看 $d$ 对 $a$、$b$ 和 $c$ 的导数，并思考这些导数究竟告诉我们什么。为了计算这些导数，我们需要采用一些技巧。我们将再次使用一个非常小的 $h$ 值，然后将输入固定在某个我们感兴趣的数值上。

比如说，我们要看 $d$ 对 $a$ 的导数。我们会取 $a$，给它增加一个 $h$，然后得到 `d2`，特别是，`d1` 为4。那么 `d2` 会是略大于 $4$ 还是略小于 $4$？这将告诉我们导数的符号。$a$ 会稍微变大，但 $b$ 是一个负数。所以我们实际 $d$ 会变小。

```python
h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)
# d1 4.0
# d2 3.999699999999999
# slope -3.000000000010772
```

然后，导数的确切数值就是 $-3$。你也可以从数学和分析的角度验证 $-3$ 是正确的答案，现在如果我们对 $b$ 做同样的操作，我们会得到不同的斜率。

```python
...
b += h
...

# d1 4.0
# d2 4.0002
# slope 2.0000000000042206
```

接着，如果我们对 $c$ 稍微增加 $h$ 呢？

```python
...
b += h
...

# d1 4.0
# d2 4.0001
# slope 0.9999999999976694
```



What does that do to the function? It makes it slightly bit higher, because we're simply adding c. And it makes it slightly bit higher by the exact same amount that we added to c. And so that tells you that the slope is 1. That will be the rate at which d will increase as we scale c. Okay, so we now have some intuitive sense of what this derivative is telling you about the function. And we'd like to move to neural networks. Now, as I mentioned, neural networks will be pretty massive expressions, mathematical expressions. 

So we need some data structures that maintain these expressions. And that's what we're going to start to build out now. So we're going to build out this value object that I showed you in the readme page of Micrograph. 

So let me copy-paste a skeleton of the first very simple value object. So class value takes a single scalar value that it wraps and keeps track of. And that's it. 

So we can, for example, do value of 2.0. And then we can look at its content. And Python will internally use the wrapper function to return this string like that. So this is a value object with data equals 2 that we're creating here. 

Now what we'd like to do is we'd like to be able to have not just two values, but we'd like to do a.b. We'd like to add them. So currently, you would get an error because Python doesn't know how to add two value objects. So we have to tell it. 

So here's addition. So you have to basically use these special double underscore methods in Python to define these operators for these objects. So if we call the, if we use this plus operator, Python will internally call a.add of b. That's what will happen internally. 

And so b will be the other and self will be a. And so we see that what we're going to return is a new value object. And it's just, it's going to be wrapping the plus of their data. But remember now, because data is the actual like numbered Python number. 

So this operator here is just a typical floating point plus addition now. It's not an addition of value objects and we'll return a new value. So now a plus b should work and it should print value of negative one because that's two plus minus three. 

There we go. Okay. Let's now implement multiply just so we can recreate this expression here. 

So multiply, I think it won't surprise you will be fairly similar. So instead of add, we're going to be using mul. And then here, of course, we want to do times. 

And so now we can create a C value object, which will be 10.0. And now we should be able to do a times b. Well, let's just do a times b first. That's value of negative six now. And by the way, I skipped over this a little bit. 

Suppose that I didn't have the wrapper function here. Then it's just that you'll get some kind of an ugly expression. So what wrapper is doing is it's providing us a way to print out like a nicer looking expression in Python. 

So we don't just have something cryptic. We actually are, you know, it's a value of negative six. So this gives us a times, and then this, we should now be able to add C to it because we've defined and told the Python how to do mul and add. 

And so this will call, this will basically be equivalent to a.mul of b. And then this new value object will be dot add of C. And let's see if that worked. Yep. So that worked well. 

That gave us four, which is what we expect from bit four. And I believe we can just call them manually as well. There we go. 

So yeah. Okay. So now what we are missing is the connected tissue of this expression. 

As I mentioned, we want to keep these expression graphs. So we need to know and keep pointers about what values produce what other values. So here, for example, we are going to introduce a new variable, which we'll call children. 

And by default, it will be an empty tuple. And then we're actually going to keep a slightly different variable in the class, which we'll call underscore prev, which will be the set of children. This is how I done it. 

I did it in the original micro grad, looking at my code here. I can't remember exactly the reason. I believe it was efficiency, but this underscore children will be a tuple for convenience. 

But then when we actually maintain it in the class, it will be just this set, I believe for efficiency. So now when we are creating a value like this with a constructor, children will be empty and prev will be the empty set. But when we're creating a value through addition or multiplication, we're going to feed in the children of this value, which in this case is self and other. 

So those are the children here. So now we can do d.prev and we'll see that the children of d we now know are this value of negative six and value of 10. And this of course is the value resulting from a times b and the c value, which is 10. 

Now the last piece of information we don't know. So we know now the children of every single value, but we don't know what operation created this value. So we need one more element here, let's call it underscore op. 

And by default, this is the empty set for leaves. And then we'll just maintain it here. And now the operation will be just a simple string.

And in the case of addition, it's plus in the case of multiplication is times. So now we not just have d.prev, we also have a d.op. And we know that d was produced by an addition of those two values. And so now we have the full mathematical expression, and we're building out this data structure, and we know exactly how each value came to be by what expression and from what other values. 

Now, because these expressions are about to get quite a bit larger, we'd like a way to nicely visualize these expressions that we're building out. So for that, I'm going to copy paste a bunch of slightly scary code that's going to visualize these expression graphs for us. So here's the code, and I'll explain it in a bit. 

But first, let me just show you what this code does. Basically, what it does is it creates a new function draw dot that we can call on some root node, and then it's going to visualize it. So if we call draw dot on d, which is this final value here, that is a times b plus c, it creates something like this. 

So this is d. And you see that this is a times b, creating an integer value, plus c, gives us this output node, d. So that's draw dot of d. And I'm not going to go through this in complete detail. You can take a look at GraphVis and its API. GraphVis is an open source graph visualization software.

And what we're doing here is we're building out this graph in GraphVis API. And you can basically see that trace is this helper function that enumerates all the nodes and edges in the graph. So that just builds a set of all the nodes and edges. 

And then we iterate through all the nodes, and we create special node objects for them using dot node. And then we also create edges using dot dot edge. And the only thing that's slightly tricky here is you'll notice that I basically add these fake nodes, which are these operation nodes. 

So for example, this node here is just a plus node. And I create these special op nodes here. And I connect them accordingly.

So these nodes, of course, are not actual nodes in the original graph. They're not actually a value object. The only value objects here are the things in squares. 

Those are actual value objects or representations thereof. And these op nodes are just created in this draw dot routine so that it looks nice. Let's also add labels to these graphs just so we know what variables are where.

So let's create a special underscore label. Or let's just do label equals empty by default and save it in each node. And then here, we're going to do label as A, label as B, label as C. And then let's create a special E equals A times B. And E dot label will be E. It's kind of naughty. 

And E will be E plus C. And a D dot label will be B. Okay, so nothing really changes. I just added this new E function, new E variable. And then here, when we are printing this, I'm going to print the label here. 

So this will be a percent S bar. And this will be N dot label. And so now, we have the label on the left here. 

So this is A, B creating E. And then E plus C creates D, just like we have it here. And finally, let's make this expression just one layer deeper. So D will not be the final output node. 

Instead, after D, we are going to create a new value object called F. We're going to start running out of variables soon. F will be negative 2.0. And its label will, of course, just be F. And then L, capital L, will be the output of our graph. And L will be P times F. So L will be negative 8, is the output.

So now, we don't just draw a D, we draw L. Okay. And somehow, the label of L is undefined. Oops.

Oh, that label has to be explicitly given to it. There we go. So L is the output.

So let's quickly recap what we've done so far. We are able to build out mathematical expressions using only plus and times so far. They are scalar-valued along the way. 

And we can do this forward pass and build out a mathematical expression. So we have multiple inputs here, A, B, C, and F, going into a mathematical expression that produces a single output L. And this here is visualizing the forward pass. So the output of the forward pass is negative 8. That's the value. 

Now, what we'd like to do next is we'd like to run backpropagation. And in backpropagation, we are going to start here at the end, and we're going to reverse and calculate the gradient along all these intermediate values. And really, what we're computing for every single value here, we're going to compute the derivative of that node with respect to L. So the derivative of L with respect to L is just

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

And then we're going to derive what is the derivative of L with respect to F, with respect to D, with respect to C, with respect to E, with respect to B, and with respect to A. And in a neural network setting, you'd be very interested in the derivative of basically this loss function L with respect to the weights of a neural network. And here, of course, we have just these variables A, B, C, and F, but some of these will eventually represent the weights of a neural net. And so we'll need to know how those weights are impacting the loss function. 

So we'll be interested basically in the derivative of the output with respect to some of its leaf nodes, and those leaf nodes will be the weights of the neural net. And the other leaf nodes, of course, will be the data itself. But usually, we will not want or use the derivative of the loss function with respect to data, because the data is fixed, but the weights will be iterated on using the gradient information.

So next, we are going to create a variable inside the value class that maintains the derivative of L with respect to that value. And we will call this variable grad. So there's a .data, and there's a self.grad, and initially, it will be zero. 

And remember that zero basically means no effect. So at initialization, we're assuming that every value does not impact, does not affect the output. Because if the gradient is zero, that means that changing this variable is not changing the loss function. 

So by default, we assume that the gradient is zero. And then now that we have grad, and it's 0.0, we are going to be able to visualize it here after data. So here, grad is .4f, and this will be in .grad. And now we are going to be showing both the data and the grad initialized at zero. 

And we are just about getting ready to calculate the backpropagation. And of course, this grad, again, as I mentioned, is representing the derivative of the output, in this case, L, with respect to this value. So this is the derivative of L with respect to f, with respect to b, and so on. 

So let's now fill in those gradients and actually do backpropagation manually. So let's start filling in these gradients and start all the way at the end, as I mentioned here. First, we are interested to fill in this gradient here. 

So what is the derivative of L with respect to L? In other words, if I change L by a tiny amount h, how much does L change? It changes by h. So it's proportional, and therefore the derivative will be 1. We can, of course, measure these or estimate these numerical gradients just like we've seen before. So if I take this expression and I create a def lol function here and put this here. Now, the reason I'm creating a gating function lol here is because I don't want to pollute or mess up the global scope here. 

This is just kind of like a little staging area. And as you know, in Python, all of these will be local variables to this function, so I'm not changing any of the global scope here. So here, L1 will be L, and then copy-pasting this expression, we're going to add a small amount h in, for example, A. And this would be measuring the derivative of L with respect to A. So here, this will be L2, and then we want to print that derivative. 

So print L2 minus L1, which is how much L changed, and then normalize it by h. So this is the rise over run. And we have to be careful because L is a value node, so we actually want its data so that these are floats, dividing by h. And this should print the derivative of L with respect to A, because A is the one that we bumped a little bit by h. So what is the derivative of L with respect to A? It's 6. And obviously, if we change L by h, then that would be here, effectively. This looks really awkward, but changing L by h, you see the derivative here is 1. That's kind of like the base case of what we are doing here.

So basically, we come up here, and we can manually set L.grad to 1. This is our manual backpropagation. L.grad is 1, and let's redraw. And we'll see that we filled in grad is 1 for L. We're now going to continue the backpropagation. 

So let's here look at the derivatives of L with respect to D and F. Let's do D first. So what we are interested in, if I create a markdown on here, is we'd like to know, basically, we have that L is D times F, and we'd like to know what is DL by DD. What is that? And if you know your calculus, L is D times F, so what is DL by DD? It would be F. And if you don't believe me, we can also just derive it, because the proof would be fairly straightforward. 

We go to the definition of the derivative, which is F of X plus H minus F of X. Divide H. As a limit, limit of H goes to 0 of this kind of expression. So when we have L is D times F, then increasing D by H would give us the output of D plus H times F. That's basically F of X plus H, right? Minus D times F. And then divide H. And symbolically, expanding out here, we would have basically D times F plus H times F minus D times F. Divide H. And then you see how the DF minus DF cancels, so you're left with H times F. Divide H, which is F. So in the limit as H goes to 0 of the derivative definition, we just get F in the case of D times F. So symmetrically, DL by DF will just be D. So what we have is that F dot grad, we see now, is just the value of D, which is 4. And we see that D dot grad is just the value of F. And so the value of F is negative 2. So we'll set those manually. Let me erase this markdown node. 

And then let's redraw what we have, okay? And let's just make sure that these were correct. So we seem to think that DL by DD is negative 2, so let's double check. Let me erase this plus H from before. 

And now we want the derivative with respect to F. So let's just come here when I create F, and let's do a plus H here. And this should print the derivative of L with respect to F, so we expect to see 4. Yeah, and this is 4, up to floating point funkiness. And then DL by DD should be F, which is negative 2. Grad is negative 2. So if we, again, come here and we change D, D dot data plus equals H right here.

So we expect, so we've added a little H, and then we see how L changed. And we expect to print negative 2. There we go. So we've numerically verified. 

What we're doing here is kind of like an inline gradient check. Gradient check is when we are deriving this backpropagation and getting the derivative with respect to all the intermediate results. And then numerical gradient is just, you know, estimating it using small step size. 

Now we're getting to the crux of backpropagation. So this will be the most important node to understand, because if you understand the gradient for this node, you understand all of backpropagation and all of training of neural nets, basically. So we need to derive DL by BC. 

In other words, derivative of L with respect to C, because we've computed all these other gradients already. Now we're coming here and we're continuing the backpropagation manually. So we want DL by DC, and then we'll also derive DL by DE. 

Now here's the problem. How do we derive DL by DC? We actually know the derivative of L with respect to D. So we know how L is sensitive to D. But how is L sensitive to C? So if we wiggle C, how does that impact L through D? So we know DL by DC, and we also here know how C impacts D. And so just very intuitively, if you know the impact that C is having on D and the impact that D is having on L, then you should be able to somehow put that information together to figure out how C impacts L. And indeed, this is what we can actually do. So in particular, we know, just concentrating on D first, let's look at what is the derivative basically of D with respect to C. So in other words, what is DD by DC? So here we know that D is C times C plus E. That's what we know. 

And now we're interested in DD by DC. If you just know your calculus again and you remember that differentiating C plus E with respect to C, you know that that gives you 1.0. And we can also go back to the basics and derive this. Because again, we can go to our f of x plus h minus f of x divided by h. That's the definition of a derivative as h goes to zero. 

And so here, focusing on C and its effect on D, we can basically do the f of x plus h will be C is incremented by h plus E. That's the first evaluation of our function minus C plus E. And then divide h. And so what is this? Just expanding this out, this will be C plus h plus E minus C minus E divide h. And then you see here how C minus C cancels, E minus E cancels. We're left with h over h, which is 1.0. And so by symmetry also, DD by DE will be 1.0 as well. So basically the derivative of a sum expression is very simple. 

And this is the local derivative. So I call this the local derivative because we have the final output value all the way at the end of this graph. And we're now like a small node here. 

And this is a little plus node. And the little plus node doesn't know anything about the rest of the graph that it's embedded in. All it knows is that it did a plus. 

It took a C and an E, added them and created D. And this plus node also knows the local influence of C on D, or rather the derivative of D with respect to C. And it also knows the derivative of D with respect to E. But that's not what we want. That's the local derivative. What we actually want is DL by DC. 

And L is here just one step away. But in a general case, this little plus node could be embedded in a massive graph. So again, we know how L impacts D. And now we know how C and E impact D. How do we put that information together to write DL by DC? And the answer, of course, is the chain rule in calculus.

And so I pulled up a chain rule here from Wikipedia. And I'm going to go through this very briefly. So chain rule, Wikipedia sometimes can be very confusing. 

And calculus can be very confusing. Like this is the way I learned chain rule. And it was very confusing. 

Like what is happening? It's just complicated. So I like this expression much better. If a variable Z depends on a variable Y, which itself depends on a variable X, then Z depends on X as well, obviously, through the intermediate variable Y. And in this case, the chain rule is expressed as, if you want DZ by DX, then you take the DZ by DY and you multiply it by DY by DX.

So the chain rule fundamentally is telling you how we chain these derivatives together correctly. So to differentiate through a function composition, we have to apply a multiplication of those derivatives. So that's really what chain rule is telling us. 

And there's a nice little intuitive explanation here, which I also think is kind of cute. The chain rule states that knowing the instantaneous rate of change of Z with respect to Y and Y relative to X allows one to calculate the instantaneous rate of change of Z relative to X as a product of those two rates of change, simply the product of those two. So here's a good one. 

If a car travels twice as fast as a bicycle and the bicycle is four times as fast as walking men, then the car travels two times four, eight times as fast as a man. And so this makes it very clear that the correct thing to do sort of is to multiply. So cars twice as fast as bicycle and bicycle is four times as fast as man. 

So the car will be eight times as fast as the man. And so we can take these intermediate rates of change, if you will, and multiply them together. And that justifies the chain rule intuitively. 

So have a look at chain rule. But here, really what it means for us is there's a very simple recipe for deriving what we want, which is dL by dC. And what we have so far is we know want, and we know what is the impact of d on L. So we know dL by dD, the derivative of L with respect to dD. 

We know that that's negative two. And now because of this local reasoning that we've done here, we know dD by dC. So how does C impact D? And in particular, this is a plus node. 

So the local derivative is simply 1.0. It's very simple. And so the chain rule tells us that dL by dC, going through this intermediate variable, will just be simply dL by dD times dD by dC. That's chain rule. 

So this is identical to what's happening here, except Z is our L, Y is our D, and X is our C. So we literally just have to multiply these. And because these local derivatives, like dD by dC, are just 1, we basically just copy over dL by dD, because this is just times 1. So because dL by dD is negative 2, what is dL by dC? Well, it's the local gradient, 1.0, times dL by dD, which is negative 2. So literally, what a plus node does, you can look at it that way, is it literally just routes the gradient, because the plus node's local derivatives are just 1. And so in the chain rule, 1 times dL by dD is just dL by dD. And so that derivative just gets routed to both C and to E in this case.

So basically, we have that E dot grad, or let's start with C, since that's the one we've looked at, is negative 2 times 1, negative 2. And in the same way, by symmetry, E dot grad will be negative 2. That's the claim. So we can set those. We can redraw. 

And you see how we just assign negative to negative 2? So this backpropagating signal, which is carrying the information of what is the derivative of L with respect to all the intermediate nodes, we can imagine it almost like flowing backwards through the graph, and a plus node will simply distribute the derivative to all the children nodes of it. So this is the claim, and now let's verify it. So let me remove the plus H here from before. 

And now instead, what we want to do is we want to increment C. So C dot data will be incremented by H. And when I run this, we expect to see negative 2, negative 2. And then, of course, for E, so E dot data plus equals H, and we expect to see negative 2. Simple. So those are the derivatives of these internal nodes. And now we're going to recurse our way backwards again, and we're again going to apply the chain rule. 

So here we go, our second application of chain rule, and we will apply it all the way through the graph. We just happen to only have one more node remaining. We have that dL by dE, as we have just calculated, is negative 2. So we know that. 

So we know the derivative of L with respect to E. And now we want dL by dA, right? And the chain rule is telling us that that's just dL by dE, negative 2, times the local gradient. So what is the local gradient? Basically dE by dA. We have to look at that. 

So I'm a little times node inside a massive graph, and I only know that I did A times B, and I produced an E. So now what is dE by dA, and dE by dB? That's the only thing that I sort of know about. That's my local gradient. So because we have that E is A times B, we're asking what is dE by dA? And of course we just did that here. 

We had a times, so I'm not going to re-derive it, but if you want to differentiate this with respect to A, you'll just get B, right? The value of B, which in this case is negative 3.0. So basically we have that dL by dA. Well, let me just do it right here. We have that A dot grad, and we are applying chain rule here, is dL by dE, which we see here is negative 2, times what is dE by dA? It's the value of B, which is negative 3. That's it. 

And then we have B dot grad is again dL by dE, which is negative 2, just the same way, times what is dE by dB? It's the of A, which is 2.0. That's the value of A. So these are our claimed derivatives. Let's redraw. And we see here that A dot grad turns out to be 6, because that is negative 2 times negative 3. And B dot grad is negative 4 times, sorry, is negative 2 times 2, which is negative 4. So those are our claims. 

Let's delete this and let's verify them. We have A here, A dot data plus equals H. So the claim is that A dot grad is 6. Let's verify. 6. And we have B dot data plus equals H. So nudging B by H and looking at what happens, we claim it's negative 4. And indeed, it's negative 4, plus minus, again, float oddness.

And that's it. That was the manual backpropagation all the way from here to all the leaf nodes. And we've done it piece by piece. 

And really all we've done is, as you saw, we iterated through all the nodes one by one and locally applied the chain rule. We always know what is the derivative of L with respect to this little output. And then we look at how this output was produced.

This output was produced through some operation, and we have the pointers to the children nodes of this operation. And so in this little operation, we know what the local derivatives are, and we just multiply them onto the derivative always. So we just go through and recursively multiply on the local derivatives. 

And that's what backpropagation is. It's just a recursive application of chain rule backwards through the computation graph. Let's see this power in action just very briefly. 

What we're going to do is we're going to nudge our inputs to try to make L go up. So in particular, what we're doing is we want A.data. We're going to change it. And if we want L to go up, that means we just have to go in the direction of the gradient.

So A should increase in the direction of gradient by some small step amount. This is the step size. And we don't just want this for B, but also for B, also for C, also for F. Those are leaf nodes, which we usually have control over. 

And if we nudge in direction of the gradient, we expect a positive influence on L. So we expect L to go up positively. So it should become less negative. It should go up to, say, negative six or something like that.

It's hard to tell exactly. And we'd have to rerun the forward pass. So let me just do that here. 

This would be the forward pass. F would be unchanged. This is effectively the forward pass. 

And now if we print L.data, we expect, because we nudged all the values, all the inputs in the direction of gradient, we expect a less negative L. We expect it to go up. So maybe it's negative six or so. Let's see what happens. 

Okay, negative seven. And this is basically one step of an optimization that we'll end up running. And really, this gradient just gives us some power, because we know how to influence the final outcome. 

And this will be extremely useful for training all that as well as CMC. So now I would like to do one more example of manual backpropagation using a bit more complex and useful example. We are going to backpropagate through a neuron. 

So we want to eventually build out neural networks. And in the simplest case, these are multilayer perceptrons, as they're called. So this is a two-layer neural net. 

And it's got these hidden layers made up of neurons. And these neurons are fully connected to each other. Now, biologically, neurons are very complicated devices. 

But we have very simple mathematical models of them. And so this is a very simple mathematical model of a neuron. You have some inputs, x's. 

And then you have these synapses that have weights on them. So the W's are weights. And then the synapse interacts with the input to this neuron multiplicatively. 

So what flows to the cell body of this neuron is W times x. But there's multiple inputs. So there's many W times x's flowing to the cell body. The cell body then has also some bias. 

So this is kind of like the innate trigger happiness of this neuron. So this bias can make it a bit more trigger happy or a bit less trigger happy, regardless of the input. But basically, we're taking all the W times x of all the inputs, adding the bias. 

And then we take it through an activation function. And this activation function is usually some kind of a squashing function, like a sigmoid or tanh or something like that. So as an example, we're going to use the tanh in this example. 

NumPy has a np.tanh. So we can call it on a range. And we can plot it. So this is the tanh function. 

And you see that the inputs, as they come in, get squashed on the y-coordinate here. So right at 0, we're going to get exactly 0. And then as you go more positive in the input, then you'll see that the function will only go up to 1 and then plateau out. And so if you pass in very positive inputs, we're going to cap it smoothly at 1. And on the negative side, we're going to cap it smoothly to negative 1. So that's tanh.

And that's the squashing function or an activation function. And what comes out of this neuron is the activation function applied to the dot product of the weights and the inputs. So let's write one out. 

I'm going to copy-paste because I don't want to type too much. But OK, so here we have the inputs x1, x2. So this is a two-dimensional neuron. 

So two inputs are going to come in. These are thought of as the weights of this neuron, weights w1, w2. And these weights, again, are the synaptic strings for each input. 

And this is the bias of the neuron b. And now what we want to do is, according to this model, we need to multiply x1 times w1 and x2 times w2. And then we need to add bias on top of it. And it gets a little messy here, but all we are trying to do is x1 w1 plus x2 w2 plus b. And these are multiplied here.

Except I'm doing it in small steps so that we actually have pointers to all these intermediate nodes. So we have x1 w1 variable, x times x2 w2 variable, and I'm also labeling them. So n is now the cell body raw activation without the activation function for now.

And this should be enough to basically plot it. So draw dot of n gives us x1 times w1, x2 times w2 being added. Then the bias gets added on top of this. 

And this n is this sum. So we're now going to take it through an activation function. And let's say we use the tanh so that we produce the output. 

So what we'd like to do here is we'd like to do the output, and I'll call it o, is n dot tanh. But we haven't yet written the tanh. Now, the reason that we need to implement another tanh function here is that tanh is a hyperbolic function, and we've only so far implemented a plus and a times. 

And you can't make a tanh out of just pluses and times. You also need exponentiation. So tanh is this kind of a formula here. 

You can use either one of these. And you see that there is exponentiation involved, which we have not implemented yet for our little value node here. So we're not going to be able to produce tanh yet, and we have to go back up and implement something like it. 

Now, one option here is we could actually implement exponentiation, right? And we could return the exp of a value instead of a tanh of a value. Because if we had exp, then we have everything else that we need, because we know how to add and we know how to multiply. So we'd be able to create tanh if we knew how to exp. 

But for the purposes of this example, I specifically wanted to show you that we don't necessarily need to have the most atomic pieces in this value object. We can actually create functions at arbitrary points of abstraction. They can be complicated functions, but they can be also very, very simple functions like a plus. 

And it's totally up to us. The only thing that matters is that we know how to differentiate through any one function. So we take some inputs and we make an output. 

The only thing that matters, it can be an arbitrarily complex function, as long as you know how to create the local derivative. If you know the local derivative of how the inputs impact the output, then that's all you need. So we're going to cluster up all of this expression, and we're not going to break it down to its atomic pieces. 

We're just going to directly implement tanh. So let's do that. dev tanh. 

And then out will be a value. And we need this expression here. So let me actually copy-paste. 

Let's grab n, which is solve.theta. And then this, I believe, is the tanh. math.exp of n-1 over 2n plus 1. Maybe I can call this x, just so that it matches exactly. And now this will be t and children of this node.

There's just one child. And I'm wrapping it in a tuple. So this is a tuple of one object, just self. 

And here, the name of this operation will be tanh. And we're going to return that.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Should be implementing 10h and now we can scroll all the way down here and we can actually do n.10h and that's going to return the 10h-ed output of n. And now we should be able to draw a dot of o, not of n. So let's see how that worked. There we go, n went through 10h to produce this output. So now 10h is a sort of our little micrograd-supported node here as an operation.

And as long as we know the derivative of 10h, then we'll be able to backpropagate through it. Now let's see this 10h in action. Currently it's not squashing too much because the input to it is pretty low. 

So if the bias was increased to say 8, then we'll see that what's flowing into the 10h now is 2 and 10h is squashing it to 0.96. So we're already hitting the tail of this 10h and it will sort of smoothly go up to 1 and then plateau out over there. Okay, so now I'm going to do something slightly strange. I'm going to change this bias from 8 to this number, 6.88, etc.

And I'm going to do this for specific reasons because we're about to start backpropagation and I want to make sure that our numbers come out nice. They're not like very crazy numbers, they're nice numbers that we can sort of understand in our head. Let me also add O's label. 

O is short for output here. So that's the O. Okay, so 0.88 flows into 10h, comes out 0.7, and so on. So now we're going to do backpropagation and we're going to fill in all the gradients.

So what is the derivative O with respect to all the inputs here? And of course in a typical neural network setting, what we really care about the most is the derivative of these neurons on the weights specifically, the W2 and W1, because those are the weights that we're going to be changing part of the optimization. And the other thing that we have to remember is here we have only a single neuron, but in the neural net you typically have many neurons and they're connected. So this is only like one small neuron, a piece of a much bigger puzzle, and eventually there's a loss function that sort of measures the accuracy of the neural net, and we're backpropagating with respect to that accuracy and trying to increase it.

Okay, so let's start off backpropagation here in the end. What is the derivative of O with respect to O? The base case sort of we know always is that the gradient is just 1.0. So let me fill it in and then let me split out the drawing function here, and then here cell clear this output here. Okay, so now when we draw O, we'll see that that gradient is 1. So now we're going to backpropagate through the tanh.

So to backpropagate through tanh, we need to know the local derivative of tanh. So if we have that O is tanh of n, then what is do by dn? Now what you could do is you could come here and you could take this expression and you could do your calculus derivative taking, and that would work. But we can also just scroll down Wikipedia here into a section that hopefully tells us that derivative d by dx of tanh of x is any of these.

I like this one, 1 minus tanh squared of x. So this is 1 minus tanh of x squared. So basically what this is saying is that do by dn is 1 minus tanh of n squared. And we already have tanh of n, it's just O. So it's 1 minus O squared.

So O is the output here. So the output is this number, O dot theta is this number. And then what this is saying is that do by dn is 1 minus this squared.

So 1 minus O dot theta squared is 0.5 conveniently. So the local derivative of this tanh operation here is 0.5. And so that would be do by dn. So we can fill in that n dot grad is 0.5. We'll just fill it in.

So this is exactly 0.5, one half. So now we're going to continue the backpropagation. This is 0.5 and this is a plus node.

So how is backprop going to, what is backprop going to do here? And if you remember our previous example, a plus is just a distributor of gradient. So this gradient will simply flow to both of these equally. And that's because the local derivative of this operation is 1 for every one of its nodes.

So 1 times 0.5 is 0.5. So therefore we know that this node here, which we called this, it's grad, it's just 0.5. And we know that b dot grad is also 0.5. So let's set those and let's draw. So those are 0.5. Continuing, we have another plus. 0.5 again, we'll just distribute.

So 0.5 will flow to both of these. So we can set, there's x2w2 as well, dot grad is 0.5. And let's redraw. Pluses are my favorite operations to backpropagate through because it's very simple.

So now it's flowing into these expressions as 0.5. And so really, again, keep in mind what the derivative is telling us at every point in time along here. This is saying that if we want the output of this neuron to increase, then the influence on these expressions is positive on the output. Both of them are positive contribution to the output.

So now backpropagating to x2 and w2 first. This is a times node. So we know that the local derivative is the other term.

So if we want to calculate x2 dot grad, then can you think through what it's going to be? So x2 dot grad will be w2 dot data times this x2w2 dot grad, right? And w2 dot grad will be x2 dot data times x2w2 dot grad, right? So that's the little local piece of chain rule. Let's set them and let's redraw. So here we see that the gradient on our weight 2 is 0 because x2's data was 0, right? But x2 will have the gradient 0.5 because data here was 1. And so what's interesting here, right, is because the input x2 was 0, then because of the way the times works, of course, this gradient will be 0. And think about intuitively why that is.

Derivative always tells us the influence of this on the final output. If I wiggle w2, how is the output changing? It's not changing because we're multiplying by 0. So because it's not changing, there is no derivative. And 0 is the correct answer because we're squashing that 0. And let's do it here.

0.5 should come here and flow through this times. And so we'll have that x1 dot grad is, can you think through a little bit what this should be? The local derivative of times with respect to x1 is going to be w1. So w1's data times x1 w1 dot grad.

And w1 dot grad will be x1 dot data times x1 w1 dot grad. Let's see what those came out to be. So this is 0.5, so this would be negative 1.5, and this would be 1. And we've back propagated through this expression.

These are the actual final derivatives. So if we want this neuron's output to increase, we know that what's necessary is that w2, we have no gradient. w2 doesn't actually matter to this neuron right now, but this neuron, this weight should go up.

So if this weight goes up, then this neuron's output would have gone up and proportionally because the gradient is 1. Okay, so doing the back propagation manually is obviously ridiculous. So we are now going to put an end to this suffering, and we're going to see how we can implement the backward pass a bit more automatically. We're not going to be doing all of it manually out here.

It's now pretty obvious to us by example how these pluses and times are back propagating gradients. So let's go up to the value object, and we're going to start co-define what we've seen in the examples below. So we're going to do this by storing a special self.backward and underscore backward, and this will be a function which is going to do that little piece of chain rule.

At each little node that took inputs and produced output, we're going to store how we are going to chain the output's gradient into the input's gradients. So by default, this will be a function that doesn't do anything. So, and you can also see that here in the value in micrograd.

So with this backward function, by default, it doesn't do anything. This is an empty function. And that would be sort of the case, for example, for a leaf node.

For a leaf node, there's nothing to do. But now if when we're creating these out values, these out values are an addition of self and other. And so we want to set out's backward to be the function that propagates the gradient.

So let's define what should happen. And we're going to store it in a closure. Let's define what should happen when we call out's grad.

For addition, our job is to take out's grad and propagate it into self's grad and other grad. So basically, we want to solve self.grad to something. And we want to set others.grad to something.

And the way we saw below how chain rule works, we want to take the local derivative times the sort of global derivative, I should call it, which is the derivative of the final output of the expression with respect to out's data. With respect to out. So the local derivative of self in an addition is 1.0. So it's just 1.0 times out's grad.

That's the chain rule. And others.grad will be 1.0 times out's grad. And basically, what you're seeing here is that out's grad will simply be copied onto self's grad and other's grad, as we saw happens for an addition operation.

So we're going to later call this function to propagate the gradient, having done an addition. Let's now do the multiplication. We're going to also define a dot backward.

And we're going to set its backward to be backward. And we want to chain out's grad into self.grad and others.grad. And this will be a little piece of chain rule for multiplication. So we'll have, so what should this be? Can you think through? So what is the local derivative here? The local derivative was others.data and then times out's grad.

That's chain rule. And here we have self.data times out's grad. That's what we've been doing.

And finally here for tanh, dot backward. And then we want to set out's backward to be just backward. And here we need to backpropagate.

We have out's grad and we want to chain it into self.grad. And self.grad will be the local derivative of this operation that we've done here, which is tanh. And so we saw that the local gradient is 1 minus the tanh of x squared, which here is t. That's the local derivative because t is the output of this tanh. So 1 minus t squared is the local derivative.

And then gradient has to be multiplied because of the chain rule. So out's grad is chained through the local gradient into self.grad. And that should be basically it. So we're going to redefine our value node.

We're going to swing all the way down here. And we're going to redefine our expression. Make sure that all the grads are zero.

OK. But now we don't have to do this manually anymore. We are going to basically be calling the dot backward in the right order.

So first we want to call out's dot backward. So o was the outcome of tanh, right? So calling out's backward will be this function. This is what it will do.

Now we have to be careful because there's a times out dot grad. And out dot grad, remember, is initialized to zero. So here we see grad zero.

So as a base case, we need to set o's dot grad to 1.0 to initialize this with 1. And then once this is 1, we can call o dot backward. And what that should do is it should propagate this grad through tanh. So the local derivative times the global derivative, which is initialized at 1. So this should do.

So I thought about redoing it, but I figured I should just leave the error in here because it's pretty funny. Why is not an object not callable? It's because I screwed up. We're trying to save these functions.

So this is correct. This here, we don't want to call the function because that returns none. These functions return none.

We just want to store the function. So let me redefine the value object. And then we're going to come back in, redefine the expression, draw dot.

Everything is great. O dot grad is 1. O dot grad is 1. And now this should work, of course. OK, so o dot backward, this grad should now be 0.5 if we redraw.

And if everything went correctly, 0.5. Yay. OK, so now we need to call ns dot grad. ns dot backward, sorry.

ns backward. So that seems to have worked. So ns dot backward routed the gradient to both of these.

So this is looking great. Now we could, of course, call b dot grad. b dot backward, sorry.

What's going to happen? Well, b doesn't have a backward. b is backward because b is a leaf node. b is backward is by initialization the empty function.

So nothing would happen. But we can call it on it. But when we call this one, this backward, then we expect this 0.5 to get further routed, right? So there we go, 0.5, 0.5. And then finally, we want to call it here on x2w2 and on x1w1.

Let's do both of those. And there we go. So we get 0, 0.5, negative 1.5, and 1, exactly as we did before.

But now we've done it through calling the backward manually. So we have one last piece to get rid of, which is us calling underscore backward manually. So let's think through what we are actually doing.

We've laid out a mathematical expression, and now we're trying to go backwards through that expression. So going backwards through the expression just means that we never want to call a dot backward for any node before we've done sort of everything after it. So we have to do everything after it before we're ever going to call dot backward on any one node.

We have to get all of its full dependencies. Everything that it depends on has to propagate to it before we can continue backpropagation. So this ordering of graphs can be achieved using something called topological sort.

So topological sort is basically a laying out of a graph such that all the edges go only from left to right, basically. So here we have a graph. It's a directory acyclic graph, a DAG.

And this is two different topological orders of it, I believe, where basically you'll see that it's a laying out of the nodes such that all the edges go only one way, from left to right. And implementing topological sort, you can look in Wikipedia and so on. I'm not going to go through it in detail.

But basically, this is what builds a topological graph. We maintain a set of visited nodes. And then we are going through starting at some root node, which for us is O. That's where I want to start the topological sort.

And starting at O, we go through all of its children, and we need to lay them out from left to right. And basically, this starts at O. If it's not visited, then it marks it as visited. And then it iterates through all of its children and calls buildTopological on them.

And then after it's gone through all the children, it adds itself. So basically, this node that we're going to call it on, like say O, is only going to add itself to the topo list after all of the children have been processed. And that's how this function is guaranteeing that you're only going to be in the list once all your children are in the list.

And that's the invariant that is being maintained. So if we buildTopo on O and then inspect this list, we're going to see that it ordered our value objects. And the last one is the value of 0.707, which is the output.

So this is O, and then this is N, and then all the other nodes get laid out before it. So that builds the topological graph. And really what we're doing now is we're just calling dot underscore backward on all of the nodes in a topological order.

So if we just reset the gradients, they're all 0. What did we do? We started by setting O.grad to be 1. That's the base case. Then we built a topological order. And then we went for node in reversed of topo.

Now, in the reverse order, because this list goes from, you know, we need to go through it in reversed order. So starting at O, node dot backward. And this should be it.

There we go. Those are the correct derivatives. Finally, we are going to hide this functionality.

So I'm going to copy this, and we're going to hide it inside the value class. Because we don't want to have all that code lying around. So instead of an underscore backward, we're now going to define an actual backward.

So that's backward without the underscore. And that's going to do all the stuff that we just derived. So let me just clean this up a little bit.

So we're first going to build the topological graph starting at self. So build topo of self will populate the topological order into the topo list, which is a local variable. Then we set self dot grads to be one.

And then for each node in the reversed list, so starting at us and going to all the children, underscore backward. And that should be it. So save.

Come down here, redefine. Okay, all the grads are zero. And now what we can do is O dot backward without the underscore.

And there we go. And that's backpropagation. At least for one neuron.

We shouldn't be too happy with ourselves, actually, because we have a bad bug. And we have not surfaced the bug because of some specific conditions that we have to think about right now. So here's the simplest case that shows the bug.

Say I create a single node A and then I create a B that is A plus A. And then I call backward. So what's going to happen is A is three and then B is A plus A. So there's two arrows on top of each other here. Then we can see that B is, of course, the forward pass works.

B is just A plus A, which is six. But the gradient here is not actually correct. That we calculated automatically.

And that's because, of course, just doing calculus in your head, the derivative of B with respect to A should be two. One plus one. It's not one.

Intuitively, what's happening here, right? So B is the result of A plus A, and then we call backward on it. So let's go up and see what that does. B is a result of addition.

So out is B. And then when we call backward, what happened is self.grad was set to one and then other.grad was set to one. But because we're doing A plus A, self and other are actually the exact same object. So we are overriding the gradient.

We are setting it to one and then we are setting it again to one. And that's why it stays at one. So that's a problem.

There's another way to see this in a little bit more complicated expression. So here we have A and B. And then D will be the multiplication of the two and E will be the addition of the two. And then we multiply E times D to get F. And then we call F dot backward.

And these gradients, if you check, will be incorrect. So fundamentally, what's happening here, again, is basically we're going to see an issue anytime we use a variable more than once. Until now, in these expressions above, every variable is used exactly once.

So we didn't see the issue. But here, if a variable is used more than once, what's going to happen during backward pass? We're backpropagating from F to E to D. So far, so good. But now E calls a backward and it deposits its gradients to A and B. But then we come back to D and call backward and it overwrites those gradients at A and B. So that's obviously a problem.

And the solution here, if you look at the multivariate case of the chain rule and its generalization there, the solution there is basically that we have to accumulate these gradients. These gradients add. And so instead of setting those gradients, we can simply do plus equals.

We need to accumulate those gradients. Plus equals, plus equals, plus equals, plus equals. And this will be okay, remember, because we are initializing them at zero.

So they start at zero. And then any contribution that flows backwards will simply add. So now if we redefine this one, because the plus equals, this now works.

Because A dot grad started at zero and we called B dot backward, we deposit one and then we deposit one again. And now this is two, which is correct. And here, this will also work and we'll get correct gradients.

Because when we call E dot backward, we will deposit the gradients from this branch. And then we get to D dot backward, it will deposit its own gradients. And then those gradients simply add on top of each other.

And so we just accumulate those gradients and that fixes the issue. Okay, now, before we move on, let me actually do a bit of cleanup here and delete some of this intermediate work. So I'm not gonna need any of this now that we've derived all of it.

Um, we are going to keep this because I want to come back to it. Delete the 10H, delete our modifying example, delete the step, delete this, keep the code that draws, and then delete this example and leave behind only the definition of value. And now let's come back to this non-linearity here that we implemented the 10H.

Now, I told you that we could have broken down 10H into its explicit atoms in terms of other expressions if we had the exp function. So if you remember, 10H is defined like this. And we chose to develop 10H as a single function.

And we can do that because we know it's derivative and we can back propagate through it. But we can also break down 10H into and express it as a function of exp. And I would like to do that now because I want to prove to you that you get all the same results and all the same gradients.

But also because it forces us to implement a few more expressions. It forces us to do exponentiation, addition, subtraction, division, and things like that. And I think it's a good exercise to go through a few more of these.

Okay, so let's scroll up to the definition of value. And here, one thing that we currently can't do is we can do like a value of, say, 2.0. But we can't do, you know, here, for example, we want to add a constant one. And we can't do something like this.

And we can't do it because it says int object has no attribute data. That's because a plus one comes right here to add. And then other is the integer one.

And then here, Python is trying to access one dot data. And that's not a thing. And that's because basically one is not a value object.

And we only have addition for value objects. So as a matter of convenience, so that we can create expressions like this and make them make sense, we can simply do something like this. Basically, we let other alone if other is an instance of value.

But if it's not an instance of value, we're going to assume that it's a number, like an integer or a float. And we're going to simply wrap it in value. And then other will just become value of other.

And then other will have a data attribute. And this should work. So if I just say this, redefine value, then this should work.

There we go. Okay, and now let's do the exact same thing for multiply. Because we can't do something like this, again, for the exact same reason.

So we just have to go to mul. And if other is not a value, then let's wrap it in value. Let's redefine value.

And now this works. Now, here's a kind of unfortunate and not obvious part. A times two works.

We saw that. But two times A, is that going to work? You'd expect it to, right? But actually, it will not. And the reason it won't is because Python doesn't know.

Like when you do A times two, basically, so A times two, Python will go and it will basically do something like A.mul of two. That's basically what it will call. But to it, two times A is the same as two.mul of A. And it doesn't, two can't multiply value.

And so it's really confused about that. So instead, what happens is in Python, the way this works is you are free to define something called the rmul. And rmul is kind of like a fallback.

So if Python can't do two,

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)