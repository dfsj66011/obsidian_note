
大家好，今天我们继续实现 Makemore。在上节课中，我们实现了二元语言模型，既使用了计数方法，也使用了一个超级简单的神经网络，该网络只有一个线性层。这是我们上节课构建的 Jupyter 笔记本，我们看到我们的方法是只查看前一个字符，然后预测序列中下一个字符的分布，我们通过计数并将它们归一化为概率来实现这一点，因此这里的每一行总和为一。

现在，如果你只有一个字符的上下文背景，这种方法确实可行且易于理解。但问题在于，由于模型只考虑了一个字符的上下文，其预测结果并不理想，生成的内容听起来不太像名字。然而，这种方法更大的问题在于，如果我们想在预测序列中的下一个字符时考虑更多的上下文，情况很快就会变得复杂起来。这个表格的大小会迅速膨胀，实际上它会随着上下文长度的增加而呈指数级增长。

因为如果我们每次只取一个字符，上下文的可能性只有 27 种。但如果取前两个字符来预测第三个字符，这个矩阵的行数（可以这样理解）就会突然变成 27 乘以 27，也就是有 729 种可能的上下文组合。如果取三个字符作为上下文，那么上下文的可能性会激增至 20,000 种。因此，这个矩阵的行数实在太多了，每种可能性的计数又太少，整个模型就会崩溃，效果很差。

所以今天我们就要转向这个要点，我们将实现一个多层感知机模型来预测序列中的下一个字符。我们采用的这种建模方法遵循了Bengio 等人在 2003 年发表的这篇论文。所以我这里打开了这篇论文。虽然这不是最早提出使用多层感知机或神经网络来预测序列中下一个字符或标记的论文，但它无疑是当时非常有影响力的一篇。它经常被引用作为这一思想的代表，我认为这是一篇非常出色的文章。

因此，这就是我们要先研究再实施的论文。这篇论文共有 19 页，我们没有时间深入探讨所有细节，但我建议大家去阅读它。文章通俗易懂、引人入胜，里面还包含了许多有趣的见解。


在引言部分，他们描述的问题与我刚才所述完全一致。为了解决这个问题，他们提出了以下模型。请注意，我们正在构建的是字符级语言模型，因此我们的工作是基于字符层面的。而在这篇论文中，他们使用了17,000个可能单词的词汇表，构建的却是单词级语言模型。

但我们仍会坚持使用这些字符，只是采用相同的建模方法。现在，他们的做法基本上是建议将这17,000个单词中的每一个单词都关联到一个30维的特征向量上。因此，每个单词现在都被嵌入到一个30维的空间中。

你可以这样理解。我们有17,000个点或向量在一个30维的空间里，你可能会觉得，这非常拥挤。对于这么小的空间来说，点的数量太多了。

最初，这些单词的向量是完全随机初始化的，因此它们在空间中随机分布。但接下来，我们将通过反向传播来调整这些单词的嵌入向量。因此，在这个神经网络的训练过程中，这些点或者说向量基本上会在这个空间中移动。

你可能会想象，例如，那些意义非常相似或实际上是同义词的单词最终会出现在空间中非常相近的位置。相反，意义截然不同的单词则会出现在空间中的其他地方。除此之外，他们的建模方法与我们的完全相同。

他们使用多层神经网络，根据前面的单词预测下一个单词。为了训练神经网络，他们像我们一样，最大化训练数据的对数似然。因此，建模方法本身是相同的。

现在，他们有了一个关于这种直觉的具体例子。为什么它能奏效？基本上，假设你正在尝试预测“一只狗在空白处奔跑”这样的句子。现在，假设训练数据中从未出现过“一只狗在空白处”这个确切的短语。

而现在，当模型部署到某个地方进行测试时，它试图生成一个句子。比如它说："一只狗在____里奔跑"。由于它在训练集中从未遇到过这个确切的短语，就像我们所说的，你超出了分布范围。

比如，你根本没有任何理由去怀疑接下来会发生什么。但这种方法实际上可以让你绕过这个问题。因为也许你没有看到确切的短语，一只狗正在某个地方奔跑。


But maybe you've seen similar phrases. Maybe you've seen the phrase, the dog was running in a blank. And maybe your network has learned that a and the are like frequently are interchangeable with each other. 

And so maybe it took the embedding for a and the embedding for the, and it actually put them like nearby each other in the space. And so you can transfer knowledge through that embedding. And you can generalize in that way. 

Similarly, the network could know that cats and dogs are animals, and they co-occur in lots of very similar contexts. And so even though you haven't seen this exact phrase, or if you haven't seen exactly walking or running, you can, through the embedding space, transfer knowledge. And you can generalize to novel scenarios. 

So let's now scroll down to the diagram of the neural network. They have a nice diagram here. And in this example, we are taking three previous words.

And we are trying to predict the fourth word in a sequence. Now, these three previous words, as I mentioned, we have a vocabulary of 17,000 possible words. So every one of these basically are the index of the incoming word. 

And because there are 17,000 words, this is an integer between 0 and 16,999. Now, there's also a lookup table that they call C. This lookup table is a matrix that is 17,000 by, say, 30. And basically, what we're doing here is we're treating this as a lookup table. 

And so every index is plucking out a row of this embedding matrix, so that each index is converted to the 30-dimensional vector that corresponds to the embedding vector for that word. So here we have the input layer of 30 neurons for three words, making up 90 neurons in total. And here they're saying that this matrix C is shared across all the words. 

So we're always indexing into the same matrix C over and over for each one of these words. Next up is the hidden layer of this neural network. The size of this hidden neural layer of this neural net is a hyperparameter. 

So we use the word hyperparameter when it's kind of like a design choice up to the designer of the neural net. And this can be as large as you'd like or small as you'd like. So for example, the size could be 100. 

And we are going to go over multiple choices of the size of this hidden layer, and we're going to evaluate how well they work. So say there were 100 neurons here, all of them would be fully connected to the 90 words or 90 numbers that make up these three words. So this is a fully connected layer. 

Then there's a 10-inch long linearity. And then there's this output layer. And because there are 17,000 possible words that could come next, this layer has 17,000 neurons, and all of them are fully connected to all of these neurons in the hidden layer. 

So there's a lot of parameters here because there's a lot of words. So most computation is here. This is the expensive layer. 

Now there are 17,000 logits here. So on top of there, we have the softmax layer, which we've seen in our previous video as well. So every one of these logits is exponentiated, and then everything is normalized to sum to 1, so that we have a nice probability distribution for the next word in the sequence.

Now, of course, during training, we actually have the label. We have the identity of the next word in the sequence. That word or its index is used to pluck out the probability of that word, and then we are maximizing the probability of that word with respect to the parameters of this neural net.

So the parameters are the weights and biases of this output layer, the weights and biases of the hidden layer, and the embedding lookup table C. And all of that is optimized using backpropagation. And these dashed arrows, ignore those. That represents a variation of a neural net that we are not going to explore in this video. 

So that's the setup, and now let's implement it. Okay, so I started a brand new notebook for this lecture. We are importing PyTorch, and we are importing Matplotlib so we can create figures. 

Then I am reading all the names into a list of words like I did before, and I'm showing the first eight right here. Keep in mind that we have 32,000 in total. These are just the first eight. 

And then here I'm building out the vocabulary of characters and all the mappings from the characters as strings to integers, and vice versa. Now, the first thing we want to do is we want to compile the dataset for the neural network. And I had to rewrite this code. 

I'll show you in a second what it looks like. So this is the code that I created for the dataset creation. So let me first run it, and then I'll briefly explain how this works. 

So first we're going to define something called block size. And this is basically the context length of how many characters do we take to predict the next one. So here in this example, we're taking three characters to predict the fourth one, so we have a block size of three. 

That's the size of the block that supports the prediction. Then here I'm building out the x and y. The x are the input to the neural net, and the y are the labels for each example inside x. Then I'm iterating over the first five words. I'm doing the first five just for efficiency while we are developing all the code, but then later we are going to come here and erase this so that we use the entire training set. 

So here I'm printing the word Emma, and here I'm basically showing the examples that we can generate, the five examples that we can generate out of the single word Emma. So when we are given the context of just dot dot dot, the first character in a sequence is e. In this context, the label is m. When the context is this, the label is m, and so forth. And so the way I build this out is first I start with a padded context of just zero tokens. 

Then I iterate over all the characters. I get the character in the sequence, and I basically build out the array y of this current character, and the array x, which stores the current running context. And then here, see, I print everything, and here I crop the context and enter the new character in a sequence. 

So this is kind of like a rolling window of context. Now we can change the block size here to, for example, four, and in that case we would be predicting the fifth character given the previous four. Or it can be five, and then it would look like this. 

Or it can be, say, ten, and then it would look something like this. We're taking ten characters to predict the eleventh one, and we're always padding with dots. So let me bring this back to three, just so that we have what we have here in the paper. 

And finally, the data set right now looks as follows. From these five words, we have created a data set of 32 examples, and each input to the neural net is three integers, and we have a label that is also an integer, y. So x looks like this. These are the individual examples. 

And then y are the labels. So given this, let's now write a neural network that takes these x's and predicts the y's. First, let's build the embedding lookup table C. So we have 27 possible characters, and we're going to embed them in a lower dimensional space. 

In the paper, they have 17,000 words, and they embed them in spaces as small dimensional as 30. So they cram 17,000 words into 30-dimensional space. In our case, we have only 27 possible characters, so let's cram them in something as small as, to start with, for example, a two-dimensional space.

So this lookup table will be random numbers, and we'll have 27 rows, and we'll have two columns. Right? So each one of 27 characters will have a two-dimensional embedding. So that's our matrix C of embeddings, in the beginning, initialized randomly.

Now, before we embed all of the integers inside the input x using this lookup table C, let me actually just try to embed a single individual integer, like say 5, so we get a sense of how this works. Now, one way this works, of course, is we can just take the C, and we can index into row 5, and that gives us a vector, the fifth row of C. And this is one way to do it. The other way that I presented in the previous lecture is actually seemingly different, but actually identical. 

So in the previous lecture, what we did is we took these integers, and we used the one-hot encoding to first encode them. So f dot one-hot, we want to encode integer 5, and we want to tell it that the number of classes is 27. So that's the 26-dimensional vector of all zeros, except the fifth bit is turned on.

Now, this actually doesn't work. The reason is that this input actually must be a torstadt tensor. And I'm making some of these errors intentionally, just so you get to see some errors and how to fix them. 

So this must be a tensor, not an int, fairly straightforward to fix. We get a one-hot vector, the fifth dimension is 1, and the shape of this is 27. And now notice that, just as I briefly alluded to in a previous video, if we take this one-hot vector and we multiply it by C, then what would you expect? Well, number one, first you'd expect an error, because expected scalar type long, but found float. 

So a little bit confusing, but the problem here is that one-hot, the data type of it is long. It's a 64-bit integer, but this is a float tensor. And so PyTorch doesn't know how to multiply an int with a float, and that's why we had to explicitly cast this to a float, so that we can multiply.

Now, the output actually here is identical, and that it's identical because of the way the matrix multiplication here works. We have the one-hot vector multiplying columns of C, and because of all the zeros, they actually end up masking out everything in C, except for the fifth row, which is plucked out. And so we actually arrive at the same result.

And that tells you that here we can interpret this first piece here, this embedding of the integer, we can either think of it as the integer indexing into a lookup table C, but equivalently, we can also think of this little piece here as a first layer of this bigger neural net. This layer here has neurons that have no non-linearity, there's no tanh, they're just linear neurons, and their weight matrix is C. And then we are encoding integers into one-hot and feeding those into a neural net, and this first layer basically embeds them. So those are two equivalent ways of doing the same thing. 

We're just going to index because it's much, much faster, and we're going to discard this interpretation of one-hot inputs into neural nets, and we're just going to index integers and use embedding tables. Now, embedding a single integer like 5 is easy enough. We can simply ask PyTorch to retrieve the fifth row of C, or the row index 5 of C. But how do we simultaneously embed all of these 32 by 3 integers stored in array x? Luckily, PyTorch indexing is fairly flexible and quite powerful. 

So it doesn't just work to ask for a single element 5 like this. You can actually index using lists. So for example, we can get the rows 5, 6, and 7, and this will just work like this. 

We can index with a list. It doesn't just have to be a list, it can also be actually a tensor of integers, and we can index with that. So this is an integer tensor 5, 6, 7, and this will just work as well. 

In fact, we can also, for example, repeat row 7 and retrieve it multiple times, and that same index will just get embedded multiple times here. So here we are indexing with a one-dimensional tensor of integers, but it turns out that you can also index with multi-dimensional tensors of integers. Here we have a two-dimensional tensor of integers. 

So we can simply just do C at x, and this just works. And the shape of this is 32 by 3, which is the original shape. And now for every one of those 32 by 3 integers, we've retrieved the embedding vector here. 

So basically, we have that as an example. The 13th, or example index 13, the second dimension, is the integer 1 as an example. And so here, if we do C of x, which gives us that array, and then we index into 13 by 2 of that array, then we get the embedding here. 

And you can verify that C at 1, which is the integer at that location, is indeed equal to this. You see they're equal. So basically, long story short, PyTorch indexing is awesome, and to embed simultaneously all of the integers in x, we can simply do C of x, and that is our embedding, and that just works.

Now let's construct this layer here, the hidden layer. So we have that w1, as I'll call it, are these weights, which we will initialize randomly. Now the number of inputs to this layer is going to be 3 times 2, right? Because we have two-dimensional embeddings, and we have three of them, so the number of inputs is 6. And the number of neurons in this layer is a variable up to us. 

Let's use 100 neurons as an example. And then biases will be also initialized randomly as an example, and we just need 100 of them. Now the problem with this is we can't simply... Normally we would take the input, in this case that's embedding, and we'd like to multiply it with these weights, and then we would like to add the bias. 

This is roughly what we want to do. But the problem here is that these embeddings are stacked up in the dimensions of this input tensor, so this will not work, this matrix multiplication, because this is a shape 32 by 3 by 2, and I can't multiply that by 6 by 100. So somehow we need to concatenate these inputs here together so that we can do something along these lines, which currently does not work.

So how do we transform this 32 by 3 by 2 into a 32 by 6, so that we can actually perform this multiplication over here? I'd like to show you that there are usually many ways of implementing what you'd like to do in Torch, and some of them will be faster, better, shorter, etc. And that's because Torch is a very large library, and it's got lots and lots of functions. So if we just go to the documentation and click on Torch, you'll see that my slider here is very tiny, and that's because there are so many functions that you can call on these tensors to transform them, create them, multiply them, add them, perform all kinds of different operations on them. 

And so this is kind of like the space of possibility, if you will. Now one of the things that you can do is if we can Ctrl-F for concatenate, and we see that there's a function Torch.cat, short for concatenate. And this concatenates a given sequence of tensors in a given dimension, and these tensors must have the same shape, etc. 

So we can use the concatenate operation to, in a naive way, concatenate these three embeddings for each input. So in this case, we have mp of this shape. And really what we want to do is we want to retrieve these three parts and concatenate them. 

So we want to grab all the examples. We want to grab first the zeroth index, and then all of this. So this plucks out the 32 by 2 embeddings of just the first word here. 

And so basically we want this guy, we want the first dimension, and we want the second dimension. And these are the three pieces individually. And then we want to treat this as a sequence, and we want to Torch.cat on that sequence. 

So this is the list. Torch.cat takes a sequence of tensors, and then we have to tell it along which dimension to concatenate. So in this case, all of these are 32 by 2, and we want to concatenate not across dimension 0, but across dimension 1. So passing in 1 gives us the result that the shape of this is 32 by 6, exactly as we'd like. 

So that basically took 32 and squashed these by concatenating them into 32 by 6. Now this is kind of ugly because this code would not generalize if we want to later change the block size. Right now we have three inputs, three words, but what if we had five? Then here we would have to change the code because I'm indexing directly. Well, Torch comes to rescue again because there turns out to be a function called unbind, and it removes a tensor dimension. 

So it removes the tensor dimension, returns a tuple of all slices along a given dimension without it. So this is exactly what we need, and basically when we call Torch.unbind of m and pass in dimension 1, index 1, this gives us a list of tensors exactly equivalent to this. So running this gives us a line 3, and it's exactly this list. 

So we can call Torch.cat on it and along the first dimension, and this works, and this shape is the same. But now it doesn't matter if we have block size 3 or 5 or 10, this will just work. So this is one way to do it, but it turns out that in this case there's actually a significantly better and more efficient way, and this gives me an opportunity to hint at some of the internals of Torch.tensor. So let's create an array here of elements from 0 to 17, and the shape of this is just 18. 

It's a single vector of 18 numbers. It turns out that we can very quickly re-represent this as different sized n-dimensional tensors. We do this by calling a view, and we can say that actually this is not a single vector of 18, this is a 2 by 9 tensor, or alternatively this is a 9 by 2 tensor, or this is actually a 3 by 3 by 2 tensor. 

As long as the total number of elements here multiply to be the same, this will just work. And in PyTorch, this operation calling that view is extremely efficient, and the reason for that is that in each tensor there's something called the underlying storage. And the storage is just the numbers always as a one-dimensional vector, and this is how this tensor is represented in the computer memory. 

It's always a one-dimensional vector. But when we call that view, we are manipulating some of the attributes of that tensor that dictate how this one-dimensional sequence is interpreted to be an n-dimensional tensor. And so what's happening here is that no memory is being changed, copied, moved, or created when we call that view. 

The storage is identical, but when you call that view, some of the internal attributes of the view of this tensor are being manipulated and changed. In particular, there's something called storage offset, strides, and shapes, and those are manipulated so that this one-dimensional sequence of bytes is seen as different n-dimensional arrays. There's a blog post here from Eric called PyTorch Internals where he goes into some of this with respect to tensor and how the view of a tensor is represented.

And this is really just like a logical construct of representing the physical memory. And so this is a pretty good blog post that you can go into. I might also create an entire video on the internals of TorchTensor and how this works. 

For here, we just note that this is an extremely efficient operation. And if I delete this and come back to our EMB, we see that the shape of our EMB is 32 by 3 by 2. But we can simply ask for PyTorch to view this instead as a 32 by 6. And the way this gets flattened into a 32 by 6 array just happens that these two get stacked up in a single row. And so that's basically the concatenation operation that we're after. 

And you can verify that this actually gives the exact same result as what we had before. So this is an element y equals, and you can see that all the elements of these two tensors are the same. And so we get the exact same result. 

So long story short, we can actually just come here. And if we just view this as a 32 by 6 instead, then this multiplication will work and give us the hidden states that we're after. So if this is h, then h-shaped is now the 100 dimensional activations for every one of our 32 examples. 

And this gives the desired result. Let me do two things here. Number one, let's not use 32. 

We can, for example, do something like EMB.shape at 0 so that we don't hard code these numbers. And this would work for any size of this EMB. Or alternatively, we can also do negative 1. When we do negative 1, PyTorch will infer what this should be. 

Because the number of elements must be the same, we're saying that this is 6, PyTorch will derive that this must be 32, or whatever else it is if EMB is of different size. The other thing is here, one more thing I'd like to point out is here when we do the concatenation, this actually is much less efficient because this concatenation would create a whole new tensor with a whole new storage. So new memory is being created because there's no way to concatenate tensors just by manipulating the view attributes. 

So this is inefficient and creates all kinds of new memory. So let me delete this now. We don't need this. 

And here to calculate h, we want to also dot 10h of this to get our h. So these are now numbers between negative 1 and 1 because of the 10h. And we have that the shape is 32 by 100. And that is basically this hidden layer of activations here for every one of our 32 examples. 

Now there's one more thing I've lost over that we have to be very careful with, and that's this plus here. In particular, we want to make sure that the broadcasting will do what we like. The shape of this is 32 by 100, and B1's shape is 100.

So we see that the addition here will broadcast these two. And in particular, we have 32 by 100 broadcasting to 100. So broadcasting will align on the right, create a fake dimension here.

So this will become a 1 by 100 row vector. And then it will copy vertically for every one of these rows of 32 and do an element-wise addition. So in this case, the correct thing will be because the same bias vector will be added to all the rows of this matrix. 

So that is correct. That's what we'd like. And it's always good practice to just make sure so that you don't shoot yourself in the foot. 

And finally, let's create the final layer here. So let's create W2 and B2. The input now is 100, and the output number of neurons will be for us 27, because we have 27 possible characters that come next. 

So the biases will be 27 as well. So therefore, the logits, which are the outputs of this neural net, are going to be H multiplied by W2 plus B2. Logits.shape is 32 by 27, and the logits look good.

Now, exactly as we saw in the previous video, we want to take these logits, and we want to first exponentiate them to get our field.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)