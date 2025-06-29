
Hi, everyone. Today, we are continuing our implementation of Makemore.

Now, in the last lecture, we implemented the multilayer perceptron along the lines of Benji Hotel 2003 for character-level language modeling. So we followed this paper, took in a few characters in the past, and used an MLP to predict the next character in a sequence. So what we'd like to do now is we'd like to move on to more complex and larger neural networks, like recurrent neural networks, and their variations like the GRU, LSTM, and so on.

Now, before we do that, though, we have to stick around the level of multilayer perceptron for a bit longer. And I'd like to do this because I would like us to have a very good intuitive understanding of the activations in the neural net during training, and especially the gradients that are flowing backwards, and how they behave, and what they look like. And this is going to be very important to understand the history of the development of these architectures, because we'll see that recurrent neural networks, while they are very expressive in that they are a universal approximator and can in principle implement all the algorithms, we'll see that they are not very easily optimizable with the first-order gradient-based techniques that we have available to us and that we use all the time.

And the key to understanding why they are not optimizable easily is to understand the activations and the gradients and how they behave during training. And we'll see that a lot of the variants since recurrent neural networks have tried to improve that situation. And so that's the path that we have to take, and let's get started.

So the starting code for this lecture is largely the code from before, but I've cleaned it up a little bit. So you'll see that we are importing all the Torch and Matplotlib utilities. We're reading in the words just like before.

These are eight example words. There's a total of 32,000 of them. Here's a vocabulary of all the lowercase letters and the special dot token.

Here we are reading the dataset and processing it and creating three splits, the train, dev, and the test split. Now in the MLP, this is the identical same MLP, except you see that I've removed a bunch of magic numbers that we had here. And instead we have the dimensionality of the embedding space of the characters and the number of hidden units in the hidden layer.

And so I've pulled them outside here so that we don't have to go and change all these magic numbers all the time. With the same neural net with 11,000 parameters that we optimize now over 200,000 steps with a batch size of 32. And you'll see that I've refactored the code here a little bit, but there are no functional changes.

I just created a few extra variables, a few more comments, and I removed all the magic numbers. And otherwise it's the exact same thing. Then when we optimize, we saw that our loss looked something like this.

We saw that the train and val loss were about 2.16 and so on. Here, I refactored the code a little bit for the evaluation of arbitrary splits. So you pass in a string of which split you'd like to evaluate.

And then here, depending on train, val, or test, I index in and I get the correct split. And then this is the forward pass of the network and evaluation of the loss and printing it. So just making it nicer.

One thing that you'll notice here is I'm using a decorator torch.nograd, which you can also look up and read documentation of. Basically what this decorator does on top of a function is that whatever happens in this function is assumed by Torch to never require any gradients. So it will not do any of the bookkeeping that it does to keep track of all the gradients in anticipation of an eventual backward pass.

It's almost as if all the tensors that get created here have a requires grad of false. And so it just makes everything much more efficient because you're telling Torch that I will not call dot backward on any of this computation and you don't need to maintain the graph under the hood. So that's what this does.

And you can also use a context manager with Torch.nograd and you can look those up. Then here we have the sampling from a model just as before, just a forward pass of a neural net, getting the distribution, sampling from it, adjusting the context window and repeating until we get the special end token. And we see that we are starting to get much nicer looking words sample from the model.

It's still not amazing and they're still not fully name-like, but it's much better than when we had the bigram model. So that's our starting point. Now, the first thing I would like to scrutinize is the initialization.

I can tell that our network is very improperly configured at initialization and there's multiple things wrong with it, but let's just start with the first one. Look here on the zeroth iteration, the very first iteration, we are recording a loss of 27 and this rapidly comes down to roughly one or two or so. So I can tell that the initialization is all messed up because this is way too high.

In training of neural nets, it is almost always the case that you will have a rough idea for what loss to expect at initialization. And that just depends on the loss function and the problem setup. In this case, I do not expect 27.

I expect a much lower number and we can calculate it together. Basically at initialization, what we'd like is that there's 27 characters that could come next for any one training example. At initialization, we have no reason to believe any characters to be much more likely than others.

And so we'd expect that the probability distribution that comes out initially is a uniform distribution assigning about equal probability to all the 27 characters. So basically what we'd like is the probability for any character would be roughly one over 27. That is the probability we should record.

And then the loss is the negative log probability. So let's wrap this in a tensor and then we can take the log of it. And then the negative log probability is the loss we would expect, which is 3.29, much, much lower than 27.

And so what's happening right now is that at initialization, the neural net is creating probability distributions that are all messed up. Some characters are very confident and some characters are very not confident. And then basically what's happening is that the network is very confidently wrong.

And that's what makes it record very high loss. So here's a smaller four-dimensional example of the issue. Let's say we only have four characters and then we have logits that come out of the neural net and they are very, very close to zero.

Then when we take the softmax of all zeros, we get probabilities that are a diffuse distribution. So sums to one and is exactly uniform. And then in this case, if the label is say two, it doesn't actually matter if the label is two or three or one or zero because it's a uniform distribution, we're recording the exact same loss, in this case, 1.38. So this is the loss we would expect for a four-dimensional example.

And I can see, of course, that as we start to manipulate these logits, we're going to be changing the loss here. So it could be that we lock out and by chance, this could be a very high number like five or something like that. Then in that case, we'll record a very low loss because we're assigning the correct probability at initialization by chance to the correct label.

Much more likely it is that some other dimension will have a high logit. And then what will happen is we start to record a much higher loss. And what can happen is basically the logits come out like something like this, and they take on extreme values and we record really high loss.

For example, if we have Torch.random of four, so these are uniform, sorry, these are normally distributed numbers, four of them. And here we can also print the logits, the probabilities that come out of it and the loss. And so because these logits are near zero, for the most part, the loss that comes out is okay.

But suppose this is like times 10 now. You see how because these are more extreme values, it's very unlikely that you're going to be guessing the correct bucket, and then you're confidently wrong and recording very high loss. If your logits are coming up even more extreme, you might get extremely insane losses like infinity even at initialization.

So basically this is not good, and we want the logits to be roughly zero when the network is initialized. In fact, the logits don't have to be just zero, they just have to be equal. So for example, if all the logits are one, then because of the normalization inside the softmax, this will actually come out okay.

But by symmetry, we don't want it to be any arbitrary positive or negative number. We just want it to be all zeros and record the loss that we expect at initialization. So let's now concretely see where things go wrong in our example.

Here we have the initialization. Let me reinitialize the neural net. And here, let me break after the very first iteration.

So we only see the initial loss, which is 27. So that's way too high. And intuitively, now we can expect the variables involved.

And we see that the logits here, if we just print some of these, if we just print the first row, we see that the logits take on quite extreme values. And that's what's creating the fake confidence in incorrect answers and makes the loss get very, very high. So these logits should be much, much closer to zero.

So now let's think through how we can achieve logits coming out of this neural net to be more closer to zero. You see here that logits are calculated as the hidden states multiplied by W2 plus B2. So first of all, currently we're initializing B2 as random values of the right size.

But because we want roughly zero, we don't actually want to be adding a bias of random numbers. So in fact, I'm going to add a times a zero here to make sure that B2 is just basically zero at initialization. And second, this is H multiplied by W2.

So if we want logits to be very, very small, then we would be multiplying W2 and making that smaller. So for example, if we scale down W2 by 0.1, all the elements, then if I do again just the very first iteration, you see that we are getting much closer to what we expect. So roughly what we want is about 3.29. This is 4.2. I can make this maybe even smaller, 3.32. Okay, so we're getting closer and closer.

Now, you're probably wondering, can we just set this to zero? Then we get, of course, exactly what we're looking for at initialization. And the reason I don't usually do this is because I'm very nervous. And I'll show you in a second why you don't want to be setting Ws or weights of a neural net exactly to zero.

You usually want it to be small numbers instead of exactly zero. For this output layer in this specific case, I think it would be fine, but I'll show you in a second where things go wrong very quickly if you do that. So let's just go with 0.01. In that case, our loss is close enough, but has some entropy.

It's not exactly zero. It's got some little entropy, and that's used for symmetry breaking, as we'll see in a second. The logits are now coming out much closer to zero, and everything is well and good.

So if I just erase these and I now take away the break statement, we can run the optimization with this new initialization. And let's just see what losses we record. Okay, so I'll let it run.

And you see that we started off good, and then we came down a bit. The plot of the loss now doesn't have this hockey shape appearance, because basically what's happening in the hockey stick, the very first few iterations of the loss, what's happening during the optimization is the optimization is just squashing down the logits, and then it's rearranging the logits. So basically we took away this easy part of the loss function, where just the weights were just being shrunk down.

And so therefore we don't get these easy gains in the beginning, and we're just getting some of the hard gains of training the actual neural net. And so there's no hockey stick appearance. So good things are happening in that both, number one, loss at initialization is what we expect.

And the loss doesn't look like a hockey stick. And this is true for any neural net you might train and something to look out for. And second, the loss that came out is actually quite a bit improved.

Unfortunately, I erased what we had here before. I believe this was 2.12 and this was 2.16. So we get a slightly improved result. And the reason for that is because we're spending more cycles, more time optimizing the neural net actually, instead of just spending the first several thousand iterations probably just squashing down the weights because they are so way too high in the beginning of the initialization.

So something to look out for, and that's number one. Now let's look at the second problem. Let me reinitialize our neural net and let me reintroduce the break statement.

So we have a reasonable initial loss. So even though everything is looking good on the level of the loss and we get something that we expect, there's still a deeper problem lurking inside this neural net and its initialization. So the logits are now okay.

The problem now is with the values of H, the activations of the hidden states. Now, if we just visualize this vector, this tensor H, it's kind of hard to see, but the problem here, roughly speaking, is you see how many of the elements are one or negative one. Now recall that torch.10H, the 10H function is a squashing function.

It takes arbitrary numbers and it squashes them into a range of negative one and one, and it does so smoothly. So let's look at the histogram of H to get a better idea of the distribution of the values inside this tensor. We can do this first.

Well, we can see that H is 32 examples and 200 activations in each example. We can view it as negative one to stretch it out into one large vector. And we can then call toList to convert this into one large Python list of floats.

And then we can pass this into plt.hist for histogram. And we say we want 50 bins and a semicolon to suppress a bunch of output values.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)