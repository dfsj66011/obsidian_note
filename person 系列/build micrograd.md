
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

好了，现在我们对这个导数在函数中的含义有了直观的理解。接下来我们想转向神经网络。正如我提到的，神经网络会是非常庞大的数学表达式。因此我们需要一些数据结构来维护这些表达式。这就是我们现在要开始构建的内容。我们将构建这个我在 micrograd 的 README 页面上展示的值对象。

我们先构建一个非常简单的 Value 对象的框架。这个类 Value 接受一个单一的标量值，将其包装并跟踪。

```python
class Value:
    
    def __init__(self, data):
        self.data = data
        
    def __repr__(self):
        return f"Value(data={self.data})"
```

例如，我们可以设置一个值为 2.0，然后查看其内容。

```python
a = Value(2.0)
a
# Value(data=2.0)
```

Python 内部会使用包装函数返回这样的字符串。因此，我们在这里创建的是一个数据等于 2 的 Value 对象。

现在，我们希望实现的功能不仅仅是处理两个数值，而是能够进行 `a+b` 这样的操作。我们想要将它们相加。但目前，你会得到一个错误，因为 Python 不知道如何将两个值对象相加。

```python
a = Value(2.0)
b = Value(-3.0)
a + b
#  TypeError: unsupported operand type(s) for +: 'Value' and 'Value'
```

所以我们需要告诉它如何操作。

```python
def __add__(self, other):
    out = Value(self.data + other.data)
    return out
```

这就是加法运算。基本上，在 Python 中你需要使用这些特殊的双下划线方法来为这些对象定义运算符。因此，如果我们调用，或者说使用这个加号运算符，Python 内部会调用 `a.__add__(b)`。这就是内部实际发生的过程。因此，`b` 将成为另一个对象，而 `self` 将是 `a`。于是我们看到，我们将返回的是一个新的 Value 对象。它实际上只是封装了它们数据的加法运算。但请记住，因为 `.data` 实际上是类似 Python 数字的具体数值。所以这里的运算符现在只是一个典型的浮点数加法运算。它不是对值对象的加法操作，我们会返回一个新的值。

```python
a = Value(2.0)
b = Value(-3.0)
a + b
# Value(data=-1.0)
```

因此，现在 `a+b` 应该可以运行，并且应该打印出 $-1$ 的值，因为那是 $2+(-3)$ 的结果。

现在让我们来实现乘法功能，这样我们就可以重新创建这个表达式了。

```python
def __mul__(self, other):
    out = Value(self.data * other.data)
    return out
```

所以乘法，我想你不会感到惊讶，结果会相当相似。这里的 `a*b` 实际上就是内部调用的 `a.__mul__(b)`

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
d
# Value(data=4.0)
```

正如前面提到的，我们希望保留这些表达式图。因此我们需要了解并保存关于哪些值生成其他值的指针。例如在这里，我们将引入一个新变量，我们称之为 `_children`，默认情况下，它将是一个空元组。然后我们实际上会在类中保留一个稍有不同的变量，我们称之为 `_prev`，它将是子节点的集合。

```python
def __init__(self, data, _children=()):
	self.data = data
	self._prev = set(_children)
```

我在最初的 micrograd 项目中就是这么做的，具体原因记不太清了，应该是出于效率考虑，不过为了方便起见，这个 `_children` 会是个元组。但在实际在类中维护时，为了效率考虑，我认为它应该就是这个集合。因此，当我们像这样通过构造函数创建值时，`_children` 会是空的，而 `_prev` 会是空集。

```python
def __add__(self, other):
	out = Value(self.data + other.data, (self, other))
	return out

def __mul__(self, other):
	out = Value(self.data * other.data, (self, other))
	return out
```

然而，当我们通过加法或乘法创建值时，我们会传入该值的 `_children`，在这里就是 `self` 和 `other`。这些就是这里的子节点。

```python
d._prev
# {Value(data=-6.0), Value(data=10.0)}
```

现在我们可以执行 `d._prev` 操作，然后会看到 `d` 的子节点现在已知是 $-6$ 和 $10$ 这两个值。当然，这个值是由 `a*b` 以及 `c` 的值得出的结果。

现在我们还有最后一条信息未知。我们已经知道了每个值的子节点，但还不知道是哪个运算生成了这个值。因此，我们还需要一个元素，我们称之为 `_op`。

```python
class Value:
    
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out
```

```python
d._op
# '+'
```

现在我们不仅有 `d._prev`，还有 `d._op`。我们知道 `d` 是由这两个值相加产生的。

因此，现在我们有了完整的数学表达式，我们正在构建这个数据结构，并且我们确切地知道每个值是如何通过哪个表达式以及从哪些其他值产生的。

现在，由于这些表达式即将变得相当庞大，我们需要一种方法来清晰地可视化我们正在构建的这些表达式。为此，我将用一段看起来有点吓人的代码，它将帮助我们可视化这些表达式图。

```python
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ data %.4f }" % (n.data, ), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

```python
draw_dot(d)
```

简单来说，我们可以调用  `draw_dot` 函数作用于某个根节点，然后它会将其可视化。因此，如果我们对`d`（也就是这里的最终值，即 `a*b+c`）调用该函数，就会生成类似这样的效果。

![|450](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%20standalone%3D%22no%22%3F%3E%0A%3C!DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%0A%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%0A%3C!--%20Generated%20by%20graphviz%20version%202.44.0%20(0)%0A%20--%3E%0A%3C!--%20Pages%3A%201%20--%3E%0A%3Csvg%20width%3D%22578pt%22%20height%3D%22128pt%22%0A%20viewBox%3D%220.00%200.00%20578.00%20128.00%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%3E%0A%3Cg%20id%3D%22graph0%22%20class%3D%22graph%22%20transform%3D%22scale(1%201)%20rotate(0)%20translate(4%20124)%22%3E%0A%3Cpolygon%20fill%3D%22white%22%20stroke%3D%22transparent%22%20points%3D%22-4%2C4%20-4%2C-124%20574%2C-124%20574%2C4%20-4%2C4%22%2F%3E%0A%3C!--%20140081151755376%20--%3E%0A%3Cg%20id%3D%22node1%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151755376%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22469%2C-27.5%20469%2C-63.5%20570%2C-63.5%20570%2C-27.5%20469%2C-27.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22519.5%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%204.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151755376%2B%20--%3E%0A%3Cg%20id%3D%22node2%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151755376%2B%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22406%22%20cy%3D%22-45.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22406%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E%2B%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151755376%2B%26%2345%3B%26gt%3B140081151755376%20--%3E%0A%3Cg%20id%3D%22edge1%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151755376%2B%26%2345%3B%26gt%3B140081151755376%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M433.14%2C-45.5C440.91%2C-45.5%20449.75%2C-45.5%20458.73%2C-45.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22458.88%2C-49%20468.88%2C-45.5%20458.88%2C-42%20458.88%2C-49%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151764736%20--%3E%0A%3Cg%20id%3D%22node3%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151764736%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%220%2C-83.5%200%2C-119.5%20107%2C-119.5%20107%2C-83.5%200%2C-83.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2253.5%22%20y%3D%22-97.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B3.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758688*%20--%3E%0A%3Cg%20id%3D%22node6%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151758688*%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22170%22%20cy%3D%22-73.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22170%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E*%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151764736%26%2345%3B%26gt%3B140081151758688*%20--%3E%0A%3Cg%20id%3D%22edge6%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151764736%26%2345%3B%26gt%3B140081151758688*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M107.06%2C-88.65C116.44%2C-86.35%20125.99%2C-84.02%20134.69%2C-81.89%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22135.57%2C-85.28%20144.46%2C-79.5%20133.91%2C-78.48%20135.57%2C-85.28%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151760656%20--%3E%0A%3Cg%20id%3D%22node4%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151760656%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%223%2C-28.5%203%2C-64.5%20104%2C-64.5%20104%2C-28.5%203%2C-28.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2253.5%22%20y%3D%22-42.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%202.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151760656%26%2345%3B%26gt%3B140081151758688*%20--%3E%0A%3Cg%20id%3D%22edge5%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151760656%26%2345%3B%26gt%3B140081151758688*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M104.06%2C-58.19C114.26%2C-60.59%20124.81%2C-63.08%20134.35%2C-65.33%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22133.71%2C-68.77%20144.25%2C-67.66%20135.32%2C-61.96%20133.71%2C-68.77%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758688%20--%3E%0A%3Cg%20id%3D%22node5%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151758688%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22234.5%2C-55.5%20234.5%2C-91.5%20341.5%2C-91.5%20341.5%2C-55.5%20234.5%2C-55.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22288%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B6.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758688%26%2345%3B%26gt%3B140081151755376%2B%20--%3E%0A%3Cg%20id%3D%22edge3%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151758688%26%2345%3B%26gt%3B140081151755376%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M341.57%2C-60.81C351.41%2C-58.43%20361.47%2C-56.01%20370.58%2C-53.81%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22371.52%2C-57.18%20380.42%2C-51.43%20369.88%2C-50.38%20371.52%2C-57.18%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758688*%26%2345%3B%26gt%3B140081151758688%20--%3E%0A%3Cg%20id%3D%22edge2%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151758688*%26%2345%3B%26gt%3B140081151758688%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M197.03%2C-73.5C205.26%2C-73.5%20214.74%2C-73.5%20224.39%2C-73.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22224.46%2C-77%20234.46%2C-73.5%20224.46%2C-70%20224.46%2C-77%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758304%20--%3E%0A%3Cg%20id%3D%22node7%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081151758304%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22233%2C-0.5%20233%2C-36.5%20343%2C-36.5%20343%2C-0.5%20233%2C-0.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22288%22%20y%3D%22-14.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%2010.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081151758304%26%2345%3B%26gt%3B140081151755376%2B%20--%3E%0A%3Cg%20id%3D%22edge4%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081151758304%26%2345%3B%26gt%3B140081151755376%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M343.26%2C-31.13C352.49%2C-33.28%20361.84%2C-35.45%20370.37%2C-37.44%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22369.79%2C-40.9%20380.32%2C-39.76%20371.38%2C-34.08%20369.79%2C-40.9%22%2F%3E%0A%3C%2Fg%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E%0A)

这就是 `d` 的绘制图。我不会详细讲解这部分内容。你可以查看 GraphVis 及其 API。GraphVis 是一个开源的图形可视化软件。我们正在使用 GraphVis API 构建这个图。你可以看到 `trace` 是一个辅助函数，它会枚举图中的所有节点和边。这样就能生成所有节点和边的集合。然后我们遍历所有节点，并使用 `dot.node` 为它们创建特殊的节点对象。接着，我们也使用 `dot.edge` 创建边。这里唯一有点棘手的是，你会注意到我基本上添加了这些假节点，也就是这些操作节点。例如，这里的这个节点只是一个加法节点。我在这里创建了这些特殊的操作节点，并按相应方式连接它们。所以这些节点，当然不是原图中的实际节点。它们实际上并不是一个 Value 对象。这里唯一的 Value 对象是方框中的那些东西。这些都是实际的值对象或其表示形式。而这些操作节点只是为了看起来美观才在这个绘制点例程中创建的。

我们还可以给这些图表加上标签，这样就能知道变量都在哪里了。那就让我们创建一个标签。

```python
def __init__(self, data, _children=(), _op="", label=""):
	self.data = data
	self._prev = set(_children)
	self._op = _op
	self.label = label

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
```

```python
# 代码更新
dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
```

![|450](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%20standalone%3D%22no%22%3F%3E%0A%3C!DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%0A%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%0A%3C!--%20Generated%20by%20graphviz%20version%202.44.0%20(0)%0A%20--%3E%0A%3C!--%20Pages%3A%201%20--%3E%0A%3Csvg%20width%3D%22654pt%22%20height%3D%22127pt%22%0A%20viewBox%3D%220.00%200.00%20654.00%20127.00%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%3E%0A%3Cg%20id%3D%22graph0%22%20class%3D%22graph%22%20transform%3D%22scale(1%201)%20rotate(0)%20translate(4%20123)%22%3E%0A%3Cpolygon%20fill%3D%22white%22%20stroke%3D%22transparent%22%20points%3D%22-4%2C4%20-4%2C-123%20650%2C-123%20650%2C4%20-4%2C4%22%2F%3E%0A%3C!--%20140081109805072%20--%3E%0A%3Cg%20id%3D%22node1%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109805072%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22519%2C-54.5%20519%2C-90.5%20646%2C-90.5%20646%2C-54.5%20519%2C-54.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22532%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ed%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22545%2C-54.5%20545%2C-90.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22595.5%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%204.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109805072%2B%20--%3E%0A%3Cg%20id%3D%22node2%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109805072%2B%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22456%22%20cy%3D%22-72.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22456%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E%2B%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109805072%2B%26%2345%3B%26gt%3B140081109805072%20--%3E%0A%3Cg%20id%3D%22edge1%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109805072%2B%26%2345%3B%26gt%3B140081109805072%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M483.1%2C-72.5C490.71%2C-72.5%20499.44%2C-72.5%20508.47%2C-72.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22508.76%2C-76%20518.76%2C-72.5%20508.76%2C-69%20508.76%2C-76%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109805120%20--%3E%0A%3Cg%20id%3D%22node3%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109805120%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22259%2C-82.5%20259%2C-118.5%20393%2C-118.5%20393%2C-82.5%20259%2C-82.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22271%22%20y%3D%22-96.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ec%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22283%2C-82.5%20283%2C-118.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22338%22%20y%3D%22-96.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%2010.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109805120%26%2345%3B%26gt%3B140081109805072%2B%20--%3E%0A%3Cg%20id%3D%22edge3%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109805120%26%2345%3B%26gt%3B140081109805072%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M393.25%2C-86.01C402.62%2C-83.96%20411.92%2C-81.92%20420.36%2C-80.08%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22421.14%2C-83.49%20430.16%2C-77.93%20419.64%2C-76.65%20421.14%2C-83.49%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804784%20--%3E%0A%3Cg%20id%3D%22node4%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109804784%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%223.5%2C-55.5%203.5%2C-91.5%20129.5%2C-91.5%20129.5%2C-55.5%203.5%2C-55.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2216%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ea%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%2228.5%2C-55.5%2028.5%2C-91.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2279%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%202.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804928*%20--%3E%0A%3Cg%20id%3D%22node7%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109804928*%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22196%22%20cy%3D%22-45.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22196%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E*%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804784%26%2345%3B%26gt%3B140081109804928*%20--%3E%0A%3Cg%20id%3D%22edge4%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109804784%26%2345%3B%26gt%3B140081109804928*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M129.76%2C-59.83C140.25%2C-57.52%20150.79%2C-55.21%20160.25%2C-53.13%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22161.03%2C-56.54%20170.04%2C-50.98%20159.52%2C-49.71%20161.03%2C-56.54%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804304%20--%3E%0A%3Cg%20id%3D%22node5%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109804304%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%220%2C-0.5%200%2C-36.5%20133%2C-36.5%20133%2C-0.5%200%2C-0.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2213%22%20y%3D%22-14.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Eb%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%2226%2C-0.5%2026%2C-36.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2279.5%22%20y%3D%22-14.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B3.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804304%26%2345%3B%26gt%3B140081109804928*%20--%3E%0A%3Cg%20id%3D%22edge6%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109804304%26%2345%3B%26gt%3B140081109804928*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M133.12%2C-32.4C142.49%2C-34.38%20151.8%2C-36.35%20160.25%2C-38.14%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22159.57%2C-41.58%20170.08%2C-40.22%20161.02%2C-34.73%20159.57%2C-41.58%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804928%20--%3E%0A%3Cg%20id%3D%22node6%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140081109804928%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22260%2C-27.5%20260%2C-63.5%20392%2C-63.5%20392%2C-27.5%20260%2C-27.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22272.5%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ee%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22285%2C-27.5%20285%2C-63.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22338.5%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B6.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804928%26%2345%3B%26gt%3B140081109805072%2B%20--%3E%0A%3Cg%20id%3D%22edge5%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109804928%26%2345%3B%26gt%3B140081109805072%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M392.12%2C-59.24C401.7%2C-61.26%20411.23%2C-63.27%20419.88%2C-65.09%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22419.42%2C-68.57%20429.93%2C-67.21%20420.87%2C-61.72%20419.42%2C-68.57%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140081109804928*%26%2345%3B%26gt%3B140081109804928%20--%3E%0A%3Cg%20id%3D%22edge2%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140081109804928*%26%2345%3B%26gt%3B140081109804928%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M223.21%2C-45.5C231.19%2C-45.5%20240.39%2C-45.5%20249.93%2C-45.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22249.96%2C-49%20259.96%2C-45.5%20249.96%2C-42%20249.96%2C-49%22%2F%3E%0A%3C%2Fg%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E%0A)

最后，让我们把这个表达式再加深一层，在 `d` 之后，创建 `f` 的 Value 对象，值将为 $-2.0$。`L` 将是我们图的输出。`L=d*f`。因此，输出结果  `L` 将为 $-8$。

```python
f = Value(-2， label="f")
L = d * f; L.label ="L"

draw_dot(L)
```

![|600](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%20standalone%3D%22no%22%3F%3E%0A%3C!DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%0A%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%0A%3C!--%20Generated%20by%20graphviz%20version%202.44.0%20(0)%0A%20--%3E%0A%3C!--%20Pages%3A%201%20--%3E%0A%3Csvg%20width%3D%22913pt%22%20height%3D%22128pt%22%0A%20viewBox%3D%220.00%200.00%20913.00%20128.00%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%3E%0A%3Cg%20id%3D%22graph0%22%20class%3D%22graph%22%20transform%3D%22scale(1%201)%20rotate(0)%20translate(4%20124)%22%3E%0A%3Cpolygon%20fill%3D%22white%22%20stroke%3D%22transparent%22%20points%3D%22-4%2C4%20-4%2C-124%20909%2C-124%20909%2C4%20-4%2C4%22%2F%3E%0A%3C!--%20140014807801376%20--%3E%0A%3Cg%20id%3D%22node1%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140014807801376%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22519%2C-82.5%20519%2C-118.5%20648%2C-118.5%20648%2C-82.5%20519%2C-82.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22530%22%20y%3D%22-96.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ef%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22541%2C-82.5%20541%2C-118.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22594.5%22%20y%3D%22-96.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B2.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526748720*%20--%3E%0A%3Cg%20id%3D%22node3%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526748720*%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22711%22%20cy%3D%22-72.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22711%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E*%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140014807801376%26%2345%3B%26gt%3B140013526748720*%20--%3E%0A%3Cg%20id%3D%22edge9%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140014807801376%26%2345%3B%26gt%3B140013526748720*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M648.35%2C-86.25C657.65%2C-84.18%20666.92%2C-82.11%20675.33%2C-80.23%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22676.13%2C-83.64%20685.13%2C-78.05%20674.6%2C-76.81%20676.13%2C-83.64%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526748720%20--%3E%0A%3Cg%20id%3D%22node2%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526748720%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22774%2C-54.5%20774%2C-90.5%20905%2C-90.5%20905%2C-54.5%20774%2C-54.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22786%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3EL%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22798%2C-54.5%20798%2C-90.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22851.5%22%20y%3D%22-68.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B8.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526748720*%26%2345%3B%26gt%3B140013526748720%20--%3E%0A%3Cg%20id%3D%22edge1%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526748720*%26%2345%3B%26gt%3B140013526748720%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M738.21%2C-72.5C745.92%2C-72.5%20754.76%2C-72.5%20763.92%2C-72.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22763.96%2C-76%20773.96%2C-72.5%20763.96%2C-69%20763.96%2C-76%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751264%20--%3E%0A%3Cg%20id%3D%22node4%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526751264%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22260%2C-55.5%20260%2C-91.5%20392%2C-91.5%20392%2C-55.5%20260%2C-55.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22272.5%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ee%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22285%2C-55.5%20285%2C-91.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22338.5%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B6.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751216%2B%20--%3E%0A%3Cg%20id%3D%22node10%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526751216%2B%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22456%22%20cy%3D%22-45.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22456%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E%2B%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751264%26%2345%3B%26gt%3B140013526751216%2B%20--%3E%0A%3Cg%20id%3D%22edge8%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526751264%26%2345%3B%26gt%3B140013526751216%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M392.12%2C-59.25C401.89%2C-57.12%20411.62%2C-54.99%20420.41%2C-53.07%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22421.21%2C-56.47%20430.23%2C-50.92%20419.72%2C-49.64%20421.21%2C-56.47%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751264*%20--%3E%0A%3Cg%20id%3D%22node5%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526751264*%3C%2Ftitle%3E%0A%3Cellipse%20fill%3D%22none%22%20stroke%3D%22black%22%20cx%3D%22196%22%20cy%3D%22-73.5%22%20rx%3D%2227%22%20ry%3D%2218%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22196%22%20y%3D%22-69.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3E*%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751264*%26%2345%3B%26gt%3B140013526751264%20--%3E%0A%3Cg%20id%3D%22edge2%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526751264*%26%2345%3B%26gt%3B140013526751264%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M223.21%2C-73.5C231.19%2C-73.5%20240.39%2C-73.5%20249.93%2C-73.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22249.96%2C-77%20259.96%2C-73.5%20249.96%2C-70%20249.96%2C-77%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013528028736%20--%3E%0A%3Cg%20id%3D%22node6%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013528028736%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%223.5%2C-83.5%203.5%2C-119.5%20129.5%2C-119.5%20129.5%2C-83.5%203.5%2C-83.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2216%22%20y%3D%22-97.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ea%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%2228.5%2C-83.5%2028.5%2C-119.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2279%22%20y%3D%22-97.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%202.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013528028736%26%2345%3B%26gt%3B140013526751264*%20--%3E%0A%3Cg%20id%3D%22edge7%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013528028736%26%2345%3B%26gt%3B140013526751264*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M129.76%2C-87.83C140.25%2C-85.52%20150.79%2C-83.21%20160.25%2C-81.13%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22161.03%2C-84.54%20170.04%2C-78.98%20159.52%2C-77.71%20161.03%2C-84.54%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526750064%20--%3E%0A%3Cg%20id%3D%22node7%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526750064%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%220%2C-28.5%200%2C-64.5%20133%2C-64.5%20133%2C-28.5%200%2C-28.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2213%22%20y%3D%22-42.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Eb%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%2226%2C-28.5%2026%2C-64.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%2279.5%22%20y%3D%22-42.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%20%26%2345%3B3.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526750064%26%2345%3B%26gt%3B140013526751264*%20--%3E%0A%3Cg%20id%3D%22edge5%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526750064%26%2345%3B%26gt%3B140013526751264*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M133.12%2C-60.4C142.49%2C-62.38%20151.8%2C-64.35%20160.25%2C-66.14%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22159.57%2C-69.58%20170.08%2C-68.22%20161.02%2C-62.73%20159.57%2C-69.58%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526750592%20--%3E%0A%3Cg%20id%3D%22node8%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526750592%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22259%2C-0.5%20259%2C-36.5%20393%2C-36.5%20393%2C-0.5%20259%2C-0.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22271%22%20y%3D%22-14.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ec%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22283%2C-0.5%20283%2C-36.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22338%22%20y%3D%22-14.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%2010.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526750592%26%2345%3B%26gt%3B140013526751216%2B%20--%3E%0A%3Cg%20id%3D%22edge6%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526750592%26%2345%3B%26gt%3B140013526751216%2B%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M393.25%2C-32.47C402.62%2C-34.45%20411.92%2C-36.41%20420.36%2C-38.19%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22419.65%2C-41.62%20430.16%2C-40.26%20421.1%2C-34.77%20419.65%2C-41.62%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751216%20--%3E%0A%3Cg%20id%3D%22node9%22%20class%3D%22node%22%3E%0A%3Ctitle%3E140013526751216%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22520%2C-27.5%20520%2C-63.5%20647%2C-63.5%20647%2C-27.5%20520%2C-27.5%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22533%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Ed%3C%2Ftext%3E%0A%3Cpolyline%20fill%3D%22none%22%20stroke%3D%22black%22%20points%3D%22546%2C-27.5%20546%2C-63.5%20%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22596.5%22%20y%3D%22-41.8%22%20font-family%3D%22Times-Roman%22%20font-size%3D%2214.00%22%3Edata%204.0000%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751216%26%2345%3B%26gt%3B140013526748720*%20--%3E%0A%3Cg%20id%3D%22edge4%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526751216%26%2345%3B%26gt%3B140013526748720*%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M647.25%2C-59C656.94%2C-61.08%20666.63%2C-63.17%20675.4%2C-65.06%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22674.69%2C-68.48%20685.21%2C-67.17%20676.17%2C-61.64%20674.69%2C-68.48%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%20140013526751216%2B%26%2345%3B%26gt%3B140013526751216%20--%3E%0A%3Cg%20id%3D%22edge3%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E140013526751216%2B%26%2345%3B%26gt%3B140013526751216%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22black%22%20d%3D%22M483%2C-45.5C491%2C-45.5%20500.22%2C-45.5%20509.76%2C-45.5%22%2F%3E%0A%3Cpolygon%20fill%3D%22black%22%20stroke%3D%22black%22%20points%3D%22509.77%2C-49%20519.77%2C-45.5%20509.77%2C-42%20509.77%2C-49%22%2F%3E%0A%3C%2Fg%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E%0A)


让我们快速回顾一下到目前为止的成果。目前我们已经能够仅用加法和乘法构建数学表达式，且这些表达式在计算过程中都是标量值。我们可以进行前向传递并构建出一个数学表达式。这里有多个输入 `a`、`b`、`c` 和 `f`，它们进入一个数学表达式后产生单个输出 `L`。这里展示的就是前向传递的可视化过程。前向传递的输出结果是 $-8$，这就是最终数值。

接下来我们要做的是运行反向传播算法。在反向传播过程中，我们将从最终结果开始，逆向计算所有这些中间值的梯度。实际上，我们会为这里的每个值计算该节点相对于 `L` 的导数—— `L` 对 `L` 的导数自然是 $1$。然后我们将依次推导出 `L` 对 `f`、`D`、`C`、`E`、`B` 以及 对 `A` 的导数。在神经网络环境中，我们最关注的是这个损失函数 `L` 相对于神经网络权重的导数。虽然这里我们只有变量 `a`、`b`、`c` 和 `f`，但其中某些变量最终会代表神经网络的权重。因此我们需要了解这些权重是如何影响损失函数的。


---

因此，我们主要关注的是输出相对于某些叶节点的导数，而这些叶节点将是神经网络的权重。当然，其他叶节点将是数据本身。但通常，我们不会想要或使用损失函数相对于数据的导数，因为数据是固定的，而权重将利用梯度信息进行迭代更新。

接下来，我们将在值类中创建一个变量，用于保存L相对于该值的导数。我们将这个变量称为grad。因此，有一个.data和一个self.grad，初始时它为零。

请记住，零基本上意味着没有影响。所以在初始化时，我们假设每个值都不会影响输出。因为如果梯度为零，就意味着改变这个变量不会改变损失函数。

默认情况下，我们假设梯度为零。既然现在有了grad，并且它的值是0.0，我们就能在这里的数据之后将其可视化。这里的grad是.4f格式，它会出现在.grad中。现在我们将同时展示数据和初始化为零的grad。

我们即将开始计算反向传播。当然，正如我提到的，这个梯度（grad）代表的是输出（在这里是L）相对于这个值的导数。因此，这是L相对于f的导数，相对于b的导数，以此类推。

那么现在让我们来填充这些梯度，并实际手动进行反向传播。正如我在这里提到的，让我们从最末端开始填充这些梯度。首先，我们感兴趣的是填充这里的这个梯度。

那么L对L的导数是什么呢？换句话说，如果我将L改变一个微小的量h，L会改变多少？它会改变h。所以它是成比例的，因此导数将是1。当然，我们可以像之前一样测量或估计这些数值梯度。所以如果我采用这个表达式，在这里创建一个def lol函数并把它放在这里。现在，我在这里创建一个名为lol的封装函数的原因是我不想污染或搞乱这里的全局作用域。

这就像是一个小小的准备区。如你所知，在Python中，所有这些都将是该函数的局部变量，所以我不会改变任何全局作用域。在这里，L1将是L，然后复制粘贴这个表达式，我们将在其中添加一个小的增量h，例如A。这将测量L对A的导数。所以在这里，这将是L2，然后我们想要打印这个导数。

所以打印L2减去L1，也就是L的变化量，然后用h进行归一化。这就是上升比上移动。我们需要注意，因为L是一个值节点，所以我们实际上需要它的数据，这样这些就是浮点数，除以h。这应该打印出L对A的导数，因为A是我们通过h稍微改变的那个。那么L对A的导数是多少呢？是6。显然，如果我们用h改变L，那么实际上就在这里。这看起来很奇怪，但用h改变L，你会看到这里的导数是1。这有点像我们在这里做的基本情况。

简单来说，我们来到这里，可以手动将L.grad设为1。这就是我们手动进行的反向传播。L.grad设为1后，我们重新绘制一下。可以看到我们已将L的grad填充为1。现在我们要继续反向传播的过程。

那么让我们来看看L对D和F的导数。先看D。我们感兴趣的是——如果在这里做个标记的话——基本上我们想知道，既然L等于D乘以F，那么DL对DD的导数是什么？这是什么？如果你懂微积分，L等于D乘以F，那么DL对DD的导数是什么？应该是F。如果你不信，我们也可以直接推导一下，因为证明过程会相当直接。

我们来看导数的定义，即F(X+H)减去F(X)再除以H。当H趋近于0时，这个表达式的极限就是导数。因此，当L等于D乘以F时，将D增加H会得到输出D加H乘以F。这实际上就是F(X+H)，对吧？然后减去D乘以F，再除以H。符号化展开后，我们基本上得到D乘以F加H乘以F减D乘以F，再除以H。可以看到DF减DF相互抵消，剩下H乘以F除以H，结果就是F。因此，在H趋近于0的导数定义极限下，对于D乘以F的情况，我们最终得到F。对称地，DL对DF的导数就是D。所以我们现在看到F点grad的值就是D的值，也就是4。而D点grad的值就是F的值，F的值是负2。因此，我们将手动设置这些值。让我擦除这个标记节点。

然后我们重新画一下现有的内容，好吗？让我们确认一下这些是否正确。我们似乎认为DL除以DD等于负2，所以再检查一遍。让我把之前这个加H擦掉。

现在我们想求关于F的导数。那么让我们回到创建F的地方，在这里加上H。这应该会打印出L关于F的导数，所以我们预期会看到4。没错，这里显示的是4，考虑到浮点数的微小误差。然后DL关于DD应该是F，也就是负2。梯度是负2。所以如果我们再次回到这里，改变D，D点数据在这里加上等于H。

所以我们预计，所以我们加了一个小H，然后看看L是如何变化的。我们预计会打印出负2。就是这样。所以我们通过数值进行了验证。

我们在这里所做的有点像一种内联梯度检查。梯度检查就是在推导反向传播时，针对所有中间结果求导的过程。而数值梯度则是通过小步长来估算这个导数。

现在我们来探讨反向传播的核心问题。这将是需要理解的最重要的节点，因为如果你理解了这个节点的梯度，基本上就理解了整个反向传播和神经网络训练的全部内容。因此，我们需要推导DL对BC的偏导数。

换句话说，就是L对C的导数，因为我们已经计算了所有其他梯度。现在我们来到这里，继续手动进行反向传播。所以我们想要DL对DC的导数，然后我们也会推导出DL对DE的导数。

现在问题来了：如何通过DC推导出DL？我们其实已经知道L对D的导数，也就是说我们清楚L对D的敏感度。但L对C的敏感度又是怎样的呢？如果我们扰动C，这会如何通过D来影响L？既然我们知道DL/DC，同时也了解C是如何影响D的。那么凭直觉来说，如果你知道C对D的影响以及D对L的影响，就应该能将这些信息整合起来，推算出C是如何影响L的。事实上，这正是我们能做的。具体来说，我们先聚焦于D，看看D对C的导数究竟是什么。换句话说，DD/DC是多少？这里我们知道D等于C乘以C加E，这是已知条件。

现在我们关注的是DC对DD的求导。如果你还记得微积分的基本知识，就会知道对C加E关于C求导的结果是1.0。我们也可以回归基础来推导这个结果。因为我们可以使用f(x+h)减去f(x)再除以h这个定义，当h趋近于零时，这就是导数的定义。

因此，在这里我们关注C及其对D的影响，基本上可以这样计算：f(x + h)会使C增加h加上E。这是我们函数的第一次评估结果减去C加E，然后除以h。那么这等于什么呢？展开来看，就是C加h加E减去C减E，再除以h。可以看到，这里的C减C相互抵消，E减E也相互抵消。剩下的就是h除以h，等于1.0。同理，DD对DE的导数也是1.0。所以，求和表达式的导数其实非常简单。

这就是局部导数。我之所以称之为局部导数，是因为我们在这个图的末端得到了最终的输出值。而我们现在就像这里的一个小节点。

这是一个小小的加法节点。这个小加法节点对它所在的图的其余部分一无所知。它只知道它做了一个加法运算。

它取了一个C和一个E，将它们相加得到D。而这个加法节点还知道C对D的局部影响，或者说D相对于C的导数。它也知道D相对于E的导数。但那不是我们想要的。那是局部导数。我们真正想要的是DL对DC的导数。

而L就在这里，仅一步之遥。但在一般情况下，这个小加号节点可能嵌入在一个庞大的图中。所以，我们再次知道L如何影响D。现在我们也知道C和E如何影响D。那么，我们如何将这些信息结合起来，写出DL对DC的表达式呢？答案当然是微积分中的链式法则。

于是我从维基百科上找到了链式法则的说明。我会非常简要地过一遍。维基百科上的链式法则有时候会让人非常困惑。

微积分可能会让人非常困惑。比如我是这样学习链式法则的，当时就觉得很费解。

到底发生了什么？这确实很复杂。所以我更喜欢这种表达方式。如果一个变量Z依赖于变量Y，而Y本身又依赖于变量X，那么显然Z也通过中间变量Y依赖于X。在这种情况下，链式法则可以表示为：如果你想求DZ对DX的导数，那么你需要先求DZ对DY的导数，再乘以DY对DX的导数。

链式法则本质上是在告诉我们如何正确地将这些导数串联起来。因此，要对一个复合函数进行微分，我们必须对这些导数进行乘法运算。这就是链式法则真正要告诉我们的内容。

这里有一个非常直观的小解释，我觉得还挺有意思的。链式法则告诉我们，如果知道Z相对于Y的瞬时变化率，以及Y相对于X的瞬时变化率，那么就可以通过这两个变化率的乘积来计算Z相对于X的瞬时变化率，简单来说就是两者相乘。所以这确实是个很好的例子。

如果汽车的速度是自行车的两倍，而自行车的速度又是行人步行速度的四倍，那么汽车的速度就是行人步行速度的两倍乘以四倍，即八倍。这样就很清楚地表明，正确的做法应该是将倍数相乘。因此，汽车的速度是自行车的两倍，自行车的速度又是行人步行速度的四倍。

所以汽车的速度将是人的八倍。因此，我们可以将这些中间变化率（如果你愿意这么称呼的话）相乘。这样就能直观地理解链式法则的合理性。

那么来看看链式法则。但在这里，对我们来说真正重要的是有一个非常简单的公式可以推导出我们想要的结果，即dL/dC。到目前为止，我们已知的是我们想要什么，以及d对L的影响。所以我们知道dL/dD，即L关于dD的导数。

我们知道那是负二。由于我们在这里进行的局部推理，现在我们知道dD对dC的导数。那么C如何影响D呢？具体来说，这是一个加法节点。

因此，局部导数就是1.0，非常简单。链式法则告诉我们，通过这个中间变量，dL对dC的导数就等于dL对dD乘以dD对dC。这就是链式法则。

这与这里发生的情况完全相同，只是Z对应我们的L，Y对应我们的D，X对应我们的C。所以我们实际上只需要将这些相乘。由于这些局部导数，比如dD/dC，只是1，我们基本上可以直接复制dL/dD的值，因为这相当于乘以1。既然dL/dD是-2，那么dL/dC是多少呢？它就是局部梯度1.0乘以dL/dD，也就是-2。所以，从某种意义上说，加法节点的作用就是简单地传递梯度，因为加法节点的局部导数就是1。在链式法则中，1乘以dL/dD就等于dL/dD。因此，在这种情况下，这个导数会同时传递给C和E。

基本上，我们有E点grad，或者让我们从C开始，因为这是我们看过的那个，是负2乘以1，负2。同样地，根据对称性，E点grad将是负2。这就是我们的主张。所以我们可以设定这些。我们可以重新绘制。

你看到我们如何直接将负号赋给负2了吗？所以这个反向传播的信号，它携带了关于L对所有中间节点的导数信息，我们可以想象它几乎像是沿着图反向流动，而一个加法节点会简单地将导数分配给它的所有子节点。这就是我们的主张，现在让我们来验证它。让我先把之前的加H去掉。

而现在，我们要做的是增加C的值。因此，C.data将增加H。当我运行这段代码时，我们预期会看到-2、-2。然后，当然对于E来说，E.data += H，我们预期会看到-2。很简单。这些就是这些内部节点的导数。现在我们将再次递归回溯，并再次应用链式法则。

那么现在，我们进入链式法则的第二次应用，并将这一法则贯穿整个计算图。恰巧我们只剩下一个节点需要处理。正如刚才计算的那样，dL对dE的导数等于负2。这一点我们已经明确了。

所以我们知道了L对E的导数。现在我们想要dL对dA的导数，对吧？链式法则告诉我们，那就是dL对dE（即负2）乘以局部梯度。那么局部梯度是什么呢？其实就是dE对dA。我们需要仔细看看这个。

所以我是这个庞大计算图中的一个微小时间节点，我只知道自己完成了A乘以B的运算，并输出了E。那么现在dE/dA和dE/dB是多少呢？这就是我仅有的认知——我的局部梯度。既然E等于A乘以B，我们要求解dE/dA的值？当然我们刚才已经在这里推导过了。



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