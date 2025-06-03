
你好，我叫安德烈，从事深度神经网络训练已有十多年。在这节课中，我想向大家展示神经网络训练的内部运作原理。具体来说，我们将从一个空白的 Jupyter 笔记本开始。

在这节课结束时，我们将定义并训练一个神经网络，你将有机会深入了解其内部运作机制，直观感受它是如何运作的。具体来说，我想带你们一步步构建 micrograd。micrograd 是我大约两年前在 GitHub 上发布的一个库。

但当时我只上传了源代码，你得自己深入研究它的工作原理。所以在这节课上，我会一步步带你梳理，并对各个部分进行讲解。那么，什么是 micrograd？它为什么有趣？micrograd 本质上是一个自动微分引擎。

Autograd 是自动梯度（automatic gradient）的缩写。它实际上实现了反向传播算法。反向传播是一种能够高效计算神经网络权重相对于某种损失函数梯度的算法。

这样一来，我们就能通过迭代调整神经网络各层权值，使损失函数最小化，从而提升网络预测精度。反向传播算法正是现代深度学习框架（如 PyTorch）的数学核心所在。要理解 micrograd 的功能特性，我认为通过具体案例演示最为直观。

那么如果我们往下滚动，你会看到 micrograd 基本上允许你构建数学表达式。这里我们正在构建一个表达式，其中有两个输入 A 和 B。你会看到 A 和 B 分别是 -4 和 2，但我们会把这些值封装进这个 Value 对象中，作为 micrograd 的一部分来构建。这个 Value 对象会将这些数字本身封装起来。

然后我们将在这里构建一个数学表达式，其中 a 和 b 会被转换为 c、d，最终转化为 e、f 和 g。我将展示 micrograd 的部分功能及其支持的操作。比如你可以对两个值对象进行加法、乘法运算，可以将它们提升为常数幂，可以偏移 1 个单位，取负值，在零处压缩，平方，除以常数，或者相互除等等。现在我们正用这两个输入 a 和 b 构建一个表达式图，并生成输出值 g。micrograd会在后台自动构建出这个完整的数学表达式。

例如，它将知道 c 也是一个值，c 是加法运算的结果，而 c 的子节点是 a 和 b，因为它会维护指向 a 和 b 值对象的指针。因此，它基本上能准确掌握所有这些是如何布局的。这样一来，我们不仅能进行所谓的正向传递（即实际查看 g 的值，这当然相当直接，我们会通过 `.data` 属性来访问它）。

因此，前向传播的输出，即 g 的值，结果是 24.7。但关键在于，我们还可以对这个 g 值对象调用 `.backward` 方法。这基本上会在节点 g 处初始化反向传播。反向传播的作用是从 g 开始，沿着表达式图向后回溯，递归地应用微积分中的链式法则。这样我们就能计算 g 相对于所有内部节点（如 e、d 和 c）以及输入 a 和 b 的导数。例如，我们可以查询 g 相对于 a 的导数，即 `a.grad`。在这个例子中，它恰好是 138，g 对 b 的导数，也就是这里的 645。

这个导数我们稍后会看到非常重要，因为它告诉我们 a 和 b 是如何通过这个数学表达式影响 g 的。具体来说，a 的梯度是 138。所以如果我们稍微调整 a，使其略微增大，138 告诉我们 g 将会增长，而增长的斜率将是 138。而 b 的增长斜率将是 645。这将告诉我们，如果 a 和 b 在正向方向上微调一点点，g 将如何响应。



Okay? Now, you might be confused about what this expression is that we built out here. And this expression, by the way, is completely meaningless. I just made it up. 

I'm just flexing about the kinds of operations that are supported by micrograd. What we actually really care about are neural networks. But it turns out that neural networks are just mathematical expressions, just like this one, but actually slightly a bit less crazy even. 

Neural networks are just a mathematical expression. They take the input data as an input, and they take the weights of a neural network as an input. And it's a mathematical expression. 

And the output are your predictions of your neural net, or the loss function. We'll see this in a bit. But basically, neural networks just happen to be a certain class of mathematical expressions. 

But backpropagation is actually significantly more general. It doesn't actually care about neural networks at all. It only cares about arbitrary mathematical expressions. 

And then we happen to use that machinery for training of neural networks. Now, one more note I would like to make at this stage is that, as you see here, micrograd is a scalar-valued autograd engine. So it's working on the level of individual scalars, like negative 4 and 2. And we're taking neural nets, and we're breaking them down all the way to these atoms of individual scalars and all the little pluses and times. 

And it's just excessive. And so obviously, you would never be doing any of this in production. It's really just done for pedagogical reasons, because it allows us to not have to deal with these n-dimensional tensors that you would use in modern deep neural network library. 

So this is really done so that you understand and refactor out backpropagation and chain rule and understanding of neural training. And then if you actually want to train bigger networks, you have to be using these tensors. But none of the math changes. 

This is done purely for efficiency. We are basically taking all the scalar values. We're packaging them up into tensors, which are just arrays of these scalars. 

And then because we have these large arrays, we're making operations on those large arrays that allows us to take advantage of the parallelism in a computer. And all those operations can be done in parallel. And then the whole thing runs faster. 

But really, none of the math changes. And they're done purely for efficiency. So I don't think that it's pedagogically useful to be dealing with tensors from scratch. 

And that's why I fundamentally wrote micrograd, because you can understand how things work at the fundamental level. And then you can speed it up later. Okay, so here's the fun part. 

My claim is that micrograd is what you need to train neural networks, and everything else is just efficiency. So you'd think that micrograd would be a very complex piece of code. And that turns out to not be the case. 

So if we just go to micrograd, and you'll see that there's only two files here in micrograd. This is the actual engine. It doesn't know anything about neural nets. 

And this is the entire neural nets library on top of micrograd. So engine and nn.py. So the actual backpropagation autograd engine that gives you the power of neural networks is literally 100 lines of code of very simple Python, which we'll understand by the end of this lecture. And then nn.py, this neural network library built on top of the autograd engine, is like a joke. 

It's like, we have to define what is a neuron. And then we have to define what is a layer of neurons. And then we define what is a multilayer perceptron, which is just a sequence of layers of neurons. 

And so it's just a total joke. So basically, there's a lot of power that comes from only 150 lines of code. And that's all you need to understand to understand neural network training, and everything else is just efficiency. 

And of course, there's a lot to efficiency. But fundamentally, that's all that's happening. Okay, so now let's dive right in and implement micrograd step by step. 

The first thing I'd like to do is I'd like to make sure that you have a very good understanding intuitively of what a derivative is, and exactly what information it gives you. So let's start with some basic imports that I copy-paste in every Jupyter Notebook always. And let's define a function, a scalar value function, f of x, as follows. 

So I just made this up randomly. I just wanted a scalar value function that takes a scalar x and returns a single scalar y. And we can call this function, of course, so we can pass in, say, 3.0 and get 20 back. Now, we can also plot this function to get a sense of its shape. 

You can tell from the mathematical expression that this is probably a parabola. It's a quadratic. And so if we just create a set of scalar values that we can feed in using, for example, a range from negative 5 to 5 in steps of 0.25. So x is just from negative 5 to 5, not including 5, in steps of 0.25. And we can actually call this function on this NumPy array as well. 

So we get a set of y's if we call f on x's. And these y's are basically also applying a function on every one of these elements independently. And we can plot this using MathPlotLib. 

So plt.plot x's and y's, and we get a nice parabola. So previously here, we fed in 3.0 somewhere here, and we received 20 back, which is here, the y-coordinate. So now I'd like to think through what is the derivative of this function at any single input point x. So what is the derivative at different points x of this function? Now, if you remember back to your calculus class, you've probably derived derivatives. 

So we take this mathematical expression, 3x squared minus 4x plus 5, and you would write it out on a piece of paper, and you would apply the product rule and all the other rules and derive the mathematical expression of the great derivative of the original function. And then you could plug in different x's and see what the derivative is. We're not going to actually do that, because no one in neural networks actually writes out the expression for neural net. 

It would be a massive expression. It would be thousands, tens of thousands of terms. No one actually derives the derivative, of course. 

And so we're not going to take this kind of symbolic approach. Instead, what I'd like to do is I'd like to look at the definition of derivative and just make sure that we really understand what the derivative is measuring, what it's telling you about the function. And so if we just look up derivative, we see that this is not a very good definition of derivative. 

This is a definition of what it means to be differentiable. But if you remember from your calculus, it is the limit as h goes to 0 of f of x plus h minus f of x over h. So basically what it's saying is if you slightly bump up, you're at some point x that you're interested in, or a, and if you slightly bump up, you slightly increase it by a small number h, how does the function respond? With what sensitivity does it respond? What is the slope at that point? Does the function go up or does it go down? And by how much? And that's the slope of that function, the slope of that response at that point. And so we can basically evaluate the derivative here numerically by taking a very small h. Of course, the definition would ask us to take h to 0. We're just going to pick a very small h, 0.001. And let's say we're interested in 0.3.0. So we can look at f of x, of course, as 20. 

And now f of x plus h, so if we slightly nudge x in a positive direction, how is the function going to respond? And just looking at this, do you expect f of x plus h to be slightly greater than 20, or do you expect it to be slightly lower than 20? And since this 3 is here, and this is 20, if we slightly go positively, the function will respond positively. So you'd expect this to be slightly greater than 20. And by how much is telling you the strength of that slope, the size of that slope. 

So f of x plus h minus f of x, this is how much the function responded in the positive direction. And we have to normalize by the run. So we have the rise over run to get the slope. 

So this, of course, is just a numerical approximation of the slope, because we have to make h very, very small to converge to the exact amount. Now, if I'm doing too many zeros, at some point, I'm going to get an incorrect answer, because we're using floating point arithmetic, and the representations of all these numbers in computer memory is finite, and at some point we get into trouble. So we can converge towards the right answer with this approach. 

But basically, at 3, the slope is 14. And you can see that by taking 3x squared minus 4x plus 5, and differentiating it in our head. So 3x squared would be 6x minus 4, and then we plug in x equals 3. So that's 18 minus 4 is 14. 

So this is correct. So that's at 3. Now, how about the slope at, say, negative 3? What would you expect for the slope? Now, telling the exact value is really hard, but what is the sign of that slope? So at negative 3, if we slightly go in the positive direction at x, the function would actually go down. And so that tells you that slope would be negative. 

So we'll get a slight number below 20. And so if we take the slope, we expect something negative, negative 22. And at some point here, of course, the slope would be 0. Now, for this specific function, I looked it up previously, and it's at point 2 over 3. So at roughly 2 over 3, that's somewhere here, this derivative would be 0. So basically, at that precise point, yeah, at that precise point, if we nudge in a positive direction, the function doesn't respond. 

This stays the same almost. And so that's why the slope is 0. Okay, now let's look at a bit more complex case. So we're going to start, you know, complexifying a bit. 

So now we have a function here with output variable b that is a function of three scalar inputs, a, b, and c. So a, b, and c are some specific values, three inputs into our expression graph, and a single output, d. And so if we just print d, we get 4. And now what I'd like to do is I'd like to, again, look at the derivatives of d with respect to a, b, and c, and think through, again, just the intuition of what this derivative is telling us. So in order to evaluate this derivative, we're going to get a bit hacky here. We're going to, again, have a very small value of h. And then we're going to fix the inputs at some values that we're interested in. 

So this is the point a, b, c at which we're going to be evaluating the derivative of d with respect to all a, b, and c at that point. So there are the inputs, and now we have d1 is that expression. And then we're going to, for example, look at the derivative of d with respect to a. So we'll take a and we'll bump it by h, and then we'll get d2 to be the exact same function. 

And now we're going to print f1, d1 is d1, d2 is d2, and print slope. So the derivative for slope here will be, of course, d2 minus d1 divide h. So d2 minus d1 is how much the function increased when we bumped the specific input that we're interested in by a tiny amount. And this is the normalized by h to get the slope.

So yeah. So I just run this. We're going to print d1, which we know is 4. Now d2 will be bumped, a will be bumped by h. So let's just think through a little bit what d2 will be printed out here. 

In particular, d1 will be 4. Will d2 be a number slightly greater than 4 or slightly lower than 4? And that's going to tell us the sign of the derivative. So we're bumping a by h, b is minus 3, c is 10. So you can just intuitively think through the derivative and what it's doing.

a will be slightly more positive, but b is a negative number. So if a is slightly more positive, because b is negative 3, we're actually going to be adding less to d. So you'd actually expect that the value of the function will go down. So let's just see this.

Yeah. And so we went from 4 to 3.9996. And that tells you that the slope will be negative. And then it will be a negative number because we went down. 

And then the exact number of slope will be exact amount of slope is negative 3. And you can also convince yourself that negative 3 is the right answer mathematically and analytically, because if you have a times b plus c and you have calculus, then differentiating a times b plus c with respect to a gives you just b. And indeed, the value of b is negative 3, which is the derivative that we have. So you can tell that that's correct. So now if we do this with b, so if we bump b by a little bit in a positive direction, we'd get different slopes. 

So what is the influence of b on the output d? So if we bump b by a tiny amount in a positive direction, then because a is positive, we'll be adding more to d. And now what is the sensitivity? What is the slope of that addition? And it might not surprise you that this should be 2. And why is it 2? Because d of d by db, differentiating with respect to b, would give us a. And the value of a is 2. So that's also working well. And then if c gets bumped a tiny amount in h by h, then of course a times b is unaffected. And now c becomes slightly bit higher.

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