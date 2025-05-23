
大家好。在本期视频中，我想继续我们面向大众的大型语言模型系列话题，比如 Chatgpt 这类模型。在上期《深入探究大语言模型》的视频里（您可以在我的 YouTube 频道观看），我们深入探讨了这些模型的底层训练原理，以及该如何理解它们的认知机制或心理运作方式。

Hi everyone. So in this video, I would like to continue our general audience series on large language models like Chatsheepd. Now in the previous video, Deep Dive into LLMs, that you can find on my YouTube, we went into a lot of the under the hood fundamentals of how these models are trained and how you should think about their cognition or psychology.

Now in this video, I want to go into more practical applications of these tools. I want to show you lots of examples. I want to take you through all the different settings that are available, and I want to show you how I use these tools and how you can also use them in your own life and work. 

So let's dive in. Okay. So first of all, the webpage that I have pulled up here is Chatsheepd.com. Now, as you might know, Chatsheepd was developed by OpenAI and deployed in 2022. 

So this was the first time that people could actually just kind of like talk to a large language model through a text interface. And this went viral and all over the place on the and this was huge. Now, since then though, the ecosystem has grown a lot. 

So I'm going to be showing you a lot of examples of Chatsheepd specifically, but now in 2025, there's many other apps that are kind of like Chatsheepd like, and this is now a much bigger and richer ecosystem. So in particular, I think Chatsheepd by OpenAI is this original gangster incumbent. It's most popular and most feature rich also because it's been around the longest, but there are many other clones available, I would say. 

I don't think it's too unfair to say, but in some cases there are kind of like unique experiences that are not found in Chatsheepd and we're going to see examples of those. So for example, Big Tech has followed with a lot of kind of Chatsheepd-like experiences. So for example, Gemini, Meta.ai and Copilot from Google, Meta and Microsoft respectively.

And there's also a number of startups. So for example, Anthropic has Clod, which is kind of like a Chatsheepd equivalent. XAI, which is Elon's company, has Grok and there's many others.

So all of these here are from the United States companies, basically. DeepSeek is a Chinese company and LeChat is a French company, Mistral. Now, where can you find these and how can you keep track of them? Well, number one, on the internet somewhere, but there are some leaderboards and in the previous video I've shown you, Chatbot Arena is one of them. 

So here you can come to some ranking of different models and you can see sort of their strength or Elo score. And so this is one place where you can keep track of them. I would say another place maybe is this SEAL leaderboard from Scale. 

And so here you can also see different kinds of evals and different kinds of models and how well they rank. And you can also come here to see which models are currently performing the best on a wide variety of tasks. So understand that the ecosystem is fairly rich, but for now I'm going to start with OpenAI because it is the incumbent and is most feature-rich, but I'm going to show you others over time as well. 

So let's start with ChatGPT. What is this text box and what do we put in here? Okay, so the most basic form of interaction with the language model is that we give a text and then we get some text back in response. So as an example, we can ask to get a haiku about what it's like to be a large language model. 

So this is a good kind of example task for a language model because these models are really good at writing. So writing haikus or poems or cover letters or resumes or email replies, they're just good at writing. So when we ask for something like this, what happens looks as follows. 

The model basically responds, words flow like a stream, endless echoes nevermind, ghost of thought unseen. Okay, it's pretty dramatic. But what we're seeing here in ChatGPT is something that looks a bit like a conversation that you would have with a friend.

These are kind of like chat bubbles. Now we saw in the previous video is that what's going on under the hood here is that this is what we call a user query, this piece of text. And this piece of text and also the response from the model, this piece of text is chopped up into little text chunks that we call tokens. 

So this sequence of text is under the hood, a token sequence, one-dimensional token sequence. Now the way we can see those tokens is we can use an app like for example, TickTokenizer. So making sure that GPT-40 is selected, I can paste my text here. 

And this is actually what the model sees under the hood. My piece of text to the model looks like a sequence of exactly 15 tokens. And these are the little text chunks that the model sees.

Now there's a vocabulary here of 200,000 roughly of possible tokens. And then these are the token IDs corresponding to all these little text chunks that are part of my query. And you can play with this and update it. 

Then you can see that for example, this is case sensitive, you would get different tokens and you can kind of edit it and see live how the token sequence changes. So our query was 15 tokens and then the model response is right here. And it responded back to us with a sequence of exactly 19 tokens. 

So that haiku is this sequence of 19 tokens. Now, so we said 15 tokens and it said 19 tokens back. Now, because this is a conversation and we want to actually maintain a lot of the metadata that actually makes up a conversation object, this is not all that's going on under the hood. 

And we saw in the previous video a little bit about the conversation format. So it gets a little bit more complicated in that we have to take our user query and we have to actually use this chat format. So let me delete the system message. 

I don't think it's very important for the purposes of understanding what's going on. Let me paste my message as the user. And then let me paste the model response as an assistant. 

And then let me crop it here properly. The tool doesn't do that properly. So here we have it as it actually happens under the hood. 

There are all these special tokens that basically begin a message from the user. And then the user says, and this is the content of what we said. And then the user ends. 

And then the assistant begins and says this, etc. Now the precise details of the conversation format are not important. What I want to get across here is that what looks to you and I as little chat bubbles going back and forth under the hood, we are collaborating with the model and we're both writing into a token stream. 

And these two bubbles back and forth were in a sequence of exactly 42 tokens under the hood. I contributed some of the first tokens and then the model continued the sequence of tokens with its response. And we could alternate and continue adding tokens here.

And together we are building out a token window, a one-dimensional sequence of tokens. Okay, so let's come back to ChatGPT now. What we are seeing here is kind of like little bubbles going back and forth between us and the model. 

Under the hood, we are building out a one-dimensional token sequence. When I click new chat here, that wipes the token window. That resets the tokens to basically zero again and restarts the conversation from scratch. 

Now the cartoon diagram that I have in my mind when I'm speaking to a model looks something like this. When we click new chat, we begin a token sequence. So this is a one-dimensional sequence of tokens.

The user, we can write tokens into this stream and then when we hit enter, we transfer control over to the language model. And the language model responds with its own token streams. And the language model has a special token that basically says something along the lines of, I'm done. 

So when it emits that token, the ChatGPT application transfers control back to us and we can take turns. Together we are building out the token stream, which we also call the context window. So the context window is kind of like this working memory of tokens and anything that is inside this context window is kind of like in the working memory of this conversation and is very directly accessible by the model. 

Now what is this entity here that we are talking to and how should we think about it? Well this language model here, we saw that the way it is trained in the previous video, we saw there are two major stages. The pre-training stage and the post-training stage. The pre-training stage is kind of like taking all of internet, chopping it up into tokens, and then compressing it into a single kind of like zip file. 

But the zip file is not exact. The zip file is lossy and probabilistic zip file because we can't possibly represent all of internet than just one sort of like say terabyte of zip file because there's just way too much information. So we just kind of get the gestalt or the vibes inside this zip file.

Now what's actually inside the zip file are the parameters of a neural network. And so for example, a one terabyte zip file would correspond to roughly say one trillion parameters inside this neural network. And what this neural network is trying to do is it's trying to basically take tokens and it's trying to predict the next token in a sequence. 

But it's doing that on internet documents so it's kind of like this internet document generator, right? And in the process of predicting the next token in a sequence on internet, the neural network gains a huge amount of knowledge about the world. And this knowledge is all represented and stuffed and compressed inside the one trillion parameters roughly of this language model. Now the pre-training stage also we saw is fairly costly. 

So this can be many tens of millions of dollars, say like three months of training and so on. So this is a costly long phase. For that reason, this phase is not done that often.

So for example, GPT-4.0, this model was pre-trained probably many months ago, maybe like even a year ago by now. And so that's why these models are a little bit out of date. They have what's called a knowledge cutoff, because that knowledge cutoff corresponds to when the model was pre-trained and its knowledge only goes up to that point. 

Now some knowledge can come into the model through the post-training phase, which we'll talk about in a second. But roughly speaking, you should think of these models as kind of like a little bit out of date because pre-training is way too expensive and happens infrequently. So any kind of recent information, like if you wanted to talk to your model about something that happened last week or so on, we're going to need other ways of providing that information to the model because it's not stored in the knowledge of the model. 

So we're going to have various tool use to give that information to the model. Now after pre-training, there's the second stage goes post-training. And the post-training stage is really attaching a smiley face to this zip file. 

Because we don't want to generate internet documents, we want this thing to take on the persona of an assistant that responds to user queries. And that's done in the process of post-training, where we swap out the dataset for a dataset of conversations that are built out by humans. So this is basically where the model takes on this persona and that actually so that we can like ask questions and it responds with answers. 

So it takes on the style of an assistant, that's post-training, but it has the knowledge of all of internet and that's by pre-training. So these two are combined in this artifact. Now the important thing to understand here I think for this section is that what you are talking to is a fully self-contained entity by default. 

This language model, think of it as a one terabyte file on a disk. Secretly that represents one trillion parameters and their precise settings inside the neural network that's trying to give you the next token in a sequence. But this is the fully self-contained entity, there's no calculator, there's no computer and python interpreter, there's no worldwide web browsing, there's none of that, there's no tool use yet in what we've talked about so far. 

You're talking to a zip file, if you stream tokens to it, it will respond with tokens back. And the zip file has the knowledge from pre-training and it has the style and form from post-training. And so that's roughly how you can think about this entity. 

Okay, so if I had to summarize what we talked about so far, I would probably do it in the form of an introduction of ChachiPT in a way that I think you should think about it. So the introduction would be hi, I'm ChachiPT, I'm a one terabyte zip file, my knowledge comes from the internet, which I read in its entirety about six months ago and I only remember vaguely, okay? And my winning personality was programmed by example by human labelers at OpenAI. So the personality is programmed in post-training and the knowledge comes from compressing the internet during pre-training and this knowledge is a little bit out of date and it's a probabilistic and slightly vague.

Some of the things that probably are mentioned very frequently on the internet I will have a lot better recollection of than some of the things that are discussed very rarely, very similar to what you might expect with a human. So let's now talk about some of the repercussions of this entity and how we can talk to it and what kinds of things we can expect from it. Now I'd like to use real examples when we actually go through this. 

So for example, this morning I asked ChachiPT the following, how much caffeine is in one shot of Americana? And I was curious because I was comparing it to matcha. Now ChachiPT will tell me that this is roughly 63 milligrams of caffeine or so. Now the reason I'm asking ChachiPT this question that I think this is okay is, number one, I'm not asking about any knowledge that is very recent. 

So I do expect that the model has sort of read about how much caffeine there is in one shot. I don't think this information has changed too much and number two, I think this information is extremely frequent on the internet. This kind of a question and this kind of information has occurred all over the place on the internet and because there were so many mentions of it, I expect the model to have good memory of it in its knowledge. 

So there's no tool use and the model, the zip file, responded that there's roughly 63 milligrams. Now I'm not guaranteed that this is the correct answer. This is just its vague recollection of the internet. 

But I can go to primary sources and maybe I can look up okay, caffeine and Americano and I could verify that yeah, it looks to be about 63 is roughly right and you can look at primary sources to decide if this is true or not. So I'm not strictly speaking guaranteed that this is true but I think probably this is the kind of thing that ChachiPT would know. Here's an example of a conversation I had two days ago actually and there's another example of a knowledge-based conversation and things that I'm comfortable asking of ChachiPT with some caveats. 

So I'm a bit sick, I have runny nose and I want to get meds that help with that. So it told me a bunch of stuff and I want my nose to not be runny so I gave it a clarification based on what it said and then it kind of gave me some of the things that might be helpful with that and then I looked at some of the meds that I have at home and I said does DayQuil or NightQuil work and it went off and it kind of like went over the ingredients of DayQuil and NightQuil and whether or not they help mitigate runny nose. Now when these ingredients are coming here again remember we are talking to a zip file that has a recollection of the internet. 

I'm not guaranteed that these ingredients are correct and in fact I actually took out the box and I looked at the and I made sure that NightQuil ingredients are exactly these ingredients and I'm doing that because I don't always fully trust what's coming out here, right? This is just a probabilistic statistical recollection of the internet but that said conversations of DayQuil and NightQuil, these are very common meds, probably there's tons of information about a lot of this on the internet and this is the kind of things that the model have pretty good recollection of. So actually these were all correct and then I said okay well I have NightQuil, how fast would it act roughly and it kind of tells me and then is acetaminophen basically a Tylenol and it says yes. So this is a good example of how ChachiPT was useful to me. 

It is a knowledge-based query. This knowledge sort of isn't recent knowledge. This is all coming from the knowledge of the model. 

I think this is common information. This is not a high-stakes situation. I'm checking ChachiPT a little bit but also this is not a high-stakes situation so no big deal. 

So I popped a NightQuil and indeed it helped but that's roughly how I'm thinking about what's coming back here. Okay so at this point I want to make two notes. The first note I want to make is that naturally as you interact with these models you'll see that your conversations are growing longer, right? Anytime you are switching topic I encourage you to always start a new chat. 

When you start a new chat as we talked about you are wiping the context window of tokens and resetting it back to zero. If it is the case that those tokens are not any more useful to your next query I encourage you to do this because these tokens in this window are expensive and they're expensive in kind of like two ways. Number one if you have lots of tokens here then the model can actually find it a little bit distracting.

So if this was a lot of tokens the model might, this is kind of like the working memory of the model, the model might be distracted by all the tokens in the past when it is trying to sample tokens much later on. So it could be distracting and it could actually decrease the accuracy of the model and of its performance. And number two the more tokens are in the window the more expensive it is by a little bit, not by too much, but by a little bit to sample the next token in the sequence. 

So your model is actually slightly slowing down. It's becoming more expensive to calculate the next token and the more tokens there are here. And so think of the tokens in the context window as a precious resource. 

Think of that as the working memory of the model and don't overload it with irrelevant information and keep it as short as you can. And you can expect that to work faster and slightly better. Of course if the information actually is related to your task you may want to keep it in there but I encourage you to as often as you can basically start a new chat whenever you are switching topic. 

The second thing is that I always encourage you to keep in mind what model you are actually using. So here on the top left we can drop down and we can see that we are currently using GPT 4.0. Now there are many different models of many different flavors and there are too many actually but we'll go through some of these over time. So we are using GPT 4.0 right now and in everything that I've shown you this is GPT 4.0. Now when I open a new incognito window, so if I go to chatgpt.com and I'm not logged in, the model that I'm talking to here, so if I just say hello, the model that I'm talking to here might not be GPT 4.0. It might be a smaller version. 

Now unfortunately OpenAI does not tell me when I'm not logged in what model I'm using which is kind of unfortunate but it's possible that you are using a smaller kind of dumber model. So if we go to the chatgpt pricing page here we see that they have three basic tiers for individuals. The free, plus, and pro. 

And in the free tier you have access to what's called GPT 4.0 mini and this is a smaller version of GPT 4.0. It is a smaller model with a smaller number of parameters. It's not going to be as creative, like its writing might not be as good, its knowledge is not going to be as good, it's going to probably hallucinate a bit more, etc. But it is kind of like the free offering, the free tier. 

They do say that you have limited access to 4.0 and 3.0 mini but I'm not actually 100% sure. It didn't tell us which model we were using so we just fundamentally don't know. Now when you pay for $20 per month, even though it doesn't say this, I think basically like they're screwing up on how they're describing this but if you go to fine print limit supply we can see that the plus users get 80 messages every three hours for GPT 4.0. So that's the flagship biggest model that's currently available as of today. 

That's available and that's what we want to be using. So if you pay $20 per month you have that with some limits. And then if you pay for $200 per month you get the pro and there's a bunch of additional goodies as well as unlimited GPT 4.0. And we're going to go into some of this because I do pay for pro subscription. 

Now the whole takeaway I want you to get from this is be mindful of the models that you're using. Typically with these companies the bigger models are more expensive to calculate and so therefore the companies charge more for the bigger models. And so make those trade-offs for yourself depending on your usage of LLMs. 

Have a look at if you can get away with the cheaper offerings and if the intelligence is not good enough for you and you're using this professionally you may really want to consider paying for the top tier models that are available from these companies. In my case in my professional work I do a lot of coding and a lot of things like that and this is still very cheap for me so I pay this very gladly because I get access to some really powerful models that I'll show you in a bit. So yeah keep track of what model you're using and make those decisions for yourself.

I also want to show you that all the other LLM providers will all have different pricing tiers with different models at different tiers that you can pay for. So for example if we go to Claude from Anthropic you'll see that I am paying for the professional plan and that gives me access to Claude 3.5 Sonnet. And if you are not paying for a pro plan then probably you only have access to maybe Haiku or something like that. 

And so use the most powerful model that kind of like works for you. Here's an example of me using Claude a while back. I was asking for just travel advice.

So I was asking for a cool city to go to and Claude told me that Zermatt in Switzerland is really cool so I ended up going there for a New Year's break following Claude's advice. But this is just an example of another thing that I find these models pretty useful for is travel advice and ideation and getting pointers that you can research further. Here we also have an example of Gemini.google.com. So this is from Google. 

I got Gemini's opinion on the matter and I asked it for a cool city to go to and it also recommended Zermatt. So that was nice. So I like to go between different models and asking them similar questions and seeing what they think about.

And for Gemini also on the top left we also have a model selector. So you can pay for the more advanced tiers and use those models. Same thing goes for Grok just released. 

We don't want to be asking Grok 2 questions because we know that Grok 3 is the most advanced model. So I want to make sure that I pay enough and such that I have Grok 3 access. So for all these different providers find the one that works best for you. 

Experiment with different providers. Experiment with different pricing tiers for the problems that you are working on. And that's kind of and often I end up personally just paying for a lot of them and then asking all of them the same question. 

And I kind of refer to all these models as my LLM council. So they're kind of like the council of language models. If I'm trying to figure out where to go on a vacation I will ask all of them. 

And so you can also do that for yourself if that works for you. Okay the next topic I want to now turn to is that of thinking models quote-unquote. So we saw in the previous video that there are multiple stages of training. 

Pre-training goes to supervised fine-tuning, goes to reinforcement learning. And reinforcement learning is where the model gets to practice on a large collection of problems that resemble the practice problems in the textbook. And it gets to practice on a lot of math and code problems.

And in the process of reinforcement learning the model discovers thinking strategies that lead to good outcomes. And these thinking strategies when you look at them they very much resemble kind of the inner monologue you have when you go through problem solving. So the model will try out different ideas, it will backtrack, it will revisit assumptions, and it will do things like that. 

Now a lot of these strategies are very difficult to hard-code as a human labeler because it's not clear what the thinking process should be. It's only in the reinforcement learning that the model can try out lots of stuff and it can find the thinking process that works for it with its knowledge and its capabilities. So this is the third stage of training these models. 

This stage is relatively recent, so only a year or two ago. And all of the different LLM labs have been experimenting with these models over the last year. And this is kind of like seen as a large breakthrough recently.

And here we looked at the paper from DeepSeek that was the first to basically talk about it publicly. And they had a nice paper about incentivizing reasoning capabilities in LLMs via reinforcement learning. So that's the paper that we looked at in the previous video.

So we now have to adjust our cartoon a little bit because basically what it looks like is our emoji now has this optional thinking bubble. And when you are using a thinking model which will do additional thinking, you are using the model that has been additionally tuned with reinforcement learning. And qualitatively, what does this look like? Well, qualitatively, the model will do a lot more thinking. 

And what you can expect is that you will get higher accuracies, especially on problems that are, for example, math, code, and things that require a lot of thinking. Things that are very simple might not actually benefit from this, but things that are actually deep and hard might benefit a lot. But basically what you're paying for it is that the models will do thinking and that can sometimes take multiple minutes because the models will emit tons and tons of tokens over a period of many minutes, and you have to wait because the model is thinking just like a human would think. 

But in situations where you have very difficult problems, this might translate to higher accuracy. So let's take a look at some examples. So here's a concrete example when I was stuck on a programming problem recently. 

So something called the gradient check fails, and I'm not sure why, and I copy-pasted the model, my code. So the details of the code are not important, but this is basically an optimization of a multi-layer perceptron, and details are not important. It's a bunch of code that I wrote, and there was a bug because my gradient check didn't work, and I was just asking for advice.

And GPT-4.0, which is the flagship, most powerful model for open AI, but without thinking, just kind of went into a bunch of things that it thought were issues or that I should double-check, but actually didn't really solve the problem. Like all the things that it gave me here are not the core issue of the problem. So the model didn't really solve the issue, and it tells me about how to debug it and so on. 

But then what I did was here in the drop-down, I turned to one of the thinking models. Now, for open AI, all of these models that start with O are thinking models. O1, O3-mini, O3-mini-high, and O1-pro mode are all thinking models, and they're not very good at naming their models, but that is the case. 

And so here they will say something like, uses advanced reasoning, or good at coding logics and stuff like that, but these are basically all tuned with reinforcement learning. And because I am paying for $200 per month, I have access to O1-pro mode, which is best at reasoning. But you might want to try some of the other ones depending on your pricing tier.

And when I gave the same model, the same prompt to O1-pro, which is the best at reasoning model, and you have to pay $200 per month for this one, then the exact same prompt, it went off and it thought for one minute, and it went through a sequence of thoughts, and open AI doesn't fully show you the exact thoughts, they just give you little summaries of the thoughts. But it thought about the code for a while, and then it actually came back with the correct solution. It noticed that the parameters are mismatched in how I pack and unpack them, and et cetera. 

So this actually solved my problem. And I tried out giving the exact same prompt to a bunch of other LLMs. So for example, Claude, I gave Claude the same problem, and it actually noticed the correct issue and solved it. 

And it did that even with Sonnet, which is not a thinking model. So Claude 3.5 Sonnet, to my knowledge, is not a thinking model. And to my knowledge, Anthropic, as of today, doesn't have a thinking model deployed, but this might change by the time you watch this video. 

But even without thinking, this model actually solved the issue. When I went to Gemini, I asked it, and it also solved the issue, even though I also could have tried the thinking model, but it wasn't necessary. I also gave it to Grok, Grok 3 in this case, and Grok 3 also solved the problem after a bunch of stuff. 

So it also solved the issue. And then finally, I went to Perplexity.ai. And the reason I like Perplexity is because when you go to the model dropdown, one of the models that they host is this DeepSeq R1. So this has the reasoning with the DeepSeq R1 model, which is the model that we saw over here. 

This is the paper. So Perplexity just hosts it and makes it very easy to use. So I copy pasted it there and I ran it. 

And I think they really render it terribly. But down here, you can see the raw thoughts of the model. Even though you have to expand them. 

But you see like, okay, the user is having trouble with the gradient check, and then it tries out a bunch of stuff. And then it says, but wait, when they accumulate the gradients, they're doing the thing incorrectly. Let's check the order. 

The parameters are packed as this, and then it notices the issue. And then it kind of like says, that's a critical mistake. And so it kind of like thinks through it and you have to wait a few minutes, and then also comes up with the correct answer. 

So basically, long story short, what do I want to show you? There exists a class of models that we call thinking models. All the different providers may or may not have a thinking model. These models are most effective for difficult problems in math and code and things like that. 

And in those kinds of cases, they can push up the accuracy of your performance. In many cases, like if you're asking for travel,

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

You're not going to benefit out of a thinking model. There's no need to wait for one minute for it to think about some destinations that you might want to go to. So for myself, I usually try out the non-thinking models because their responses are really fast, but when I suspect the response is not as good as it could have been and I want to give the opportunity to the model to think a bit longer about it, I will change it to a thinking model, depending on whichever one you have available to you. 

Now when you go to Grok, for example, when I start a new conversation with Grok, when you put the question here, like, hello, you should put something important here, you see here, think. So let the model take its time. So turn on think and then click go.

And when you click think, Grok under the hood switches to the thinking model and all the different LLM providers will kind of like have some kind of a selector for whether or not you want the model to think or whether it's okay to just like go with the previous kind of generation of the models. Okay, now the next section I want to continue to is to tool use. So far we've only talked to the language model through text and this language model is again this zip file in a folder.

It's inert, it's closed off, it's got no tools, it's just a neural network that can emit tokens. So what we want to do now though is we want to go beyond that and we want to give the model the ability to use a bunch of tools. And one of the most useful tools is an internet search.

And so let's take a look at how we can make models use internet search. So for example, again using concrete examples from my own life, a few days ago I was watching White Lotus Season 3 and I watched the first episode, and I love this TV show by the way, and I was curious when the episode 2 was coming out. And so in the old world you would imagine you go to Google or something like that, you put in like new episodes of White Lotus Season 3 and then you start clicking on these links and maybe you open a few of them or something like that, right? And you start like searching through it and trying to figure it out and sometimes you luck out and you get a schedule.

But many times you might get really crazy ads, there's a bunch of random stuff going on and it's just kind of like an unpleasant experience, right? So wouldn't it be great if a model could do this kind of a search for you, visit all the web pages and then take all those web pages, take all their content and stuff it into the context window and then basically give you the response. And that's what we're going to do now. Basically we have a mechanism or a way, we introduce a mechanism for the model to emit a special token that is some kind of a search the internet token. 

And when the model emits the search the internet token, the ChatGPT application or whatever LLM application it is you're using will stop sampling from the model and it will take the query that the model gave, it goes off, it does a search, it visits web pages, it takes all of their text and it puts everything into the context window. So now we have this internet search tool that itself can also contribute tokens into our context window and in this case it would be like lots of internet web pages and maybe there's 10 of them and maybe it just puts it all together and this could be thousands of tokens coming from these web pages just as we were looking at them ourselves. And then after it has inserted all those web pages into the context window, it will reference back to your question as to hey, when is this season getting released and it will be able to reference the text and give you the correct answer. 

And notice that this is a really good example of why we would need internet search. Without the internet search this model has no chance to actually give us the correct answer because like I mentioned this model was trained a few months ago, the schedule probably was not known back then and so when White Lotus Season 3 is coming out is not part of the real knowledge of the model and it's not in the zip file most likely because this is something that was presumably decided on in the last few weeks and so the model has to basically go off and do internet search to learn this knowledge and it learns it from the web pages just like you and I would without it and then it can answer the question once that information is in the context window. And remember again that the context window is this working memory, so once we load the articles, once all of these articles think of their text as being copy pasted into the context window, now they're in a working memory and the model can actually answer those questions because it's in the context window. 

So basically long story short don't do this manually but use tools like Perplexity as an example. So Perplexity.ai had a really nice sort of LLM that was doing internet search and I think it was like the first app that really convincingly did this. More recently Chachibti also introduced a search button, it says search the web, so we're going to take a look at that in a second. 

For now when are new episodes of White Lotus Season 3 getting released you can just ask and instead of having to do the work manually we just hit enter and the model will visit these web pages, it will create all the queries and then it will give you the answer. So it just kind of did a ton of the work for you and then you can usually there will be citations so you can actually visit those web pages yourself and you can make sure these are not hallucinations from the model and you can actually like double check that this is actually correct because it's not in principle guaranteed it's just you know something that may or may not work. If we take this we can also go to for example Chachibti and say the same thing but now when we put this question in without actually selecting search I'm not actually 100% sure what the model will do. 

In some cases the model will actually like know that this is recent knowledge and that it probably doesn't know and it will create a search in some cases we have to declare that we want to do the search. In my own personal use I would know that the model doesn't know and so I would just select search but let's see first uh let's see if uh what happens. Okay searching the web and then it prints stuff and then it cites so the model actually detected itself that it needs to search the web because it understands that this is some kind of a recent information etc so this was correct. 

Alternatively if I create a new conversation I could have also selected search because I know I need to search. Enter and then it does the same thing searching the web and that's the result. So basically when you're using look for this uh for example grok excuse me let's try grok without it without selecting search okay so the model does some search uh just knowing that it needs to search and gives you the answer. 

So basically uh let's see what Claude does. You see so Claude doesn't actually have the search tool available so it will say as of my last update in April 2024 this last update is when the model went through pre-training and so Claude is just saying as of my last update the knowledge cut off of April 2024 uh it was announced but it doesn't know so Claude doesn't have the internet search integrated as an option and will not give you the answer. I expect that this is something that might be working on. 

Let's try Gemini and let's see what it says. Unfortunately no official release date for White Lotus Season 3 yet. So Gemini 2.0 Pro Experimental does not have access to internet search and doesn't know. 

We could try some of the other ones like 2.0 Flash let me try that. Okay so this model seems to know but it doesn't give citations. Oh wait okay there we go sources and related content. 

So you see how 2.0 Flash actually has the internet search tool but I'm guessing that the 2.0 Pro which is uh the most powerful model that they have this one actually does not have access and it in here it actually tells us 2.0 Pro Experimental lacks access to real-time info and some Gemini features. So this model is not fully wired with internet search. So long story short we can get models to perform Google searches for us, visit the web pages, pull in the information to the context window and answer questions and this is a very very cool feature.

But different models, possibly different apps, have different amount of integration of this capability and so you have to be kind of on the lookout for that and sometimes the model will automatically detect that they need to do search and sometimes you're better off telling the model that you want it to do the search. So when I'm doing GPT 4.0 and I know that this requires a search you probably want to tick that box. So that's search tools. 

I wanted to show you a few more examples of how I use the search tool in my own work. So what are the kinds of queries that I use and this is fairly easy for me to do because usually for these kinds of cases I go to perplexity just out of habit even though chat GPT today can do this kind of stuff as well as do probably many other services as well. But I happen to use perplexity for these kinds of search queries.

So whenever I expect that the answer can be achieved by doing basically something like Google search and visiting a few of the top links and the answer is somewhere in those top links, whenever that is the case I expect to use the search tool and I come to perplexity. So here are some examples. Is the market open today? And this was on precedence day I wasn't 100% sure so perplexity understands what it's today it will do the search and it will figure out that on precedence day this was closed. 

Where's White Lotus season 3 filmed? Again this is something that I wasn't sure that a model would know in its knowledge. This is something niche so maybe there's not that many mentions of it on the internet and also this is more recent so I don't expect a model to know by default. So this was a good fit for the search tool. 

Does Vercel offer PostgreSQL database? So this was a good example of this because this kind of stuff changes over time and the offerings of Vercel which is a company may change over time and I want the latest and whenever something is latest or something changes I prefer to use the search tool so I come to perplexity. What is the Apple launch tomorrow and what are some of the rumors? So again this is something recent. Where is the Singles Inferno season 4 cast? Must know. 

So this is again a good example because this is very fresh information. Why is the Palantir stock going up? What is driving the enthusiasm? When is Civilization 7 coming out exactly? This is an example also like has Brian Johnson talked about the toothpaste he uses? And I was curious basically like what Brian does and again it has the two features. Number one it's a little bit esoteric so I'm not 100% sure if this is at scale on the internet and will be part of like knowledge of a model.

And number two this might change over time so I want to know what toothpaste he uses most recently and so this is a good fit again for a search tool. Is it safe to travel to Vietnam? This can potentially change over time. And then I saw a bunch of stuff on Twitter about a USAID and I wanted to know kind of like what's the deal so I searched about that and then you can kind of like dive in a bunch of ways here. 

But this use case here is kind of along the lines of I see something trending and I'm kind of curious what's happening like what is the gist of it and so I very often just quickly bring up a search of like what's happening and then get a model to kind of just give me a gist of roughly what happened because a lot of the individual tweets or posts might not have the full context just by itself. So these are examples of how I use a search tool. Okay next up I would like to tell you about this capability called Deep Research and this is fairly recent only as of like a month or two ago but I think it's incredibly cool and really interesting and kind of went under the radar for a lot of people even though I think it's sure enough. 

So when we go to Chachapiti pricing here we notice that Deep Research is listed here under pro so it currently requires $200 per month so this is the top tier. However I think it's incredibly cool so let me show you by example in what kinds of scenarios you might want to use it. Roughly speaking Deep Research is a combination of internet search and thinking and rolled out for a long time so the model will go off and it will spend tens of minutes doing with Deep Research and the first sort of company that announced this was Chachapiti as part of its pro offering very recently like a month ago so here's an example. 

Recently I was on the internet buying supplements which I know is kind of crazy but Brian Johnson has this starter pack and I was kind of curious about it and there's a thing called longevity mix right and it's got a bunch of health actives and I want to know what these things are right and of course like so like CAKG like what the hell is this boost energy production for sustained vitality like what does that mean? So one thing you could of course do is you could open up Google search and look at the Wikipedia page or something like that and do everything that you're kind of used to but Deep Research allows you to basically take an alternate route and it kind of like processes a lot of this information for you and explains it a lot better. So as an example we can do something like this. This is my example prompt. 

CAKG is one of the health actives in Brian Johnson's blueprint at 2.5 grams per serving. Can you do research on CAKG? Tell me about why it might be found in the longevity mix. It's possible efficacy in humans or animal models. 

It's potential mechanism of action. Any potential concerns or toxicity or anything like that. Now here I have this button available to me and you won't unless you pay $200 per month right now but I can turn on Deep Research. 

So let me copy paste this and hit go and now the model will say okay I'm going to research this and then sometimes it likes to ask clarifying questions before it goes off. So a focus on human clinical studies, animal models or both. So let's say both. 

Specific sources. All of all sources. I don't know. 

Comparison to other longevity compounds. Not needed. Comparison. 

Just CAKG. We can be pretty brief. The model understands. 

And we hit go. And then okay I'll research CAKG. Starting research. 

And so now we have to wait for probably about 10 minutes or so and if you'd like to click on it you can get a bunch of preview of what the is doing on a high level. So this will go off and it will do a combination of like I said thinking and internet search. But it will issue many internet searches. 

It will go through lots of papers. It will look at papers and it will think and it will come back 10 minutes from now. So this will run for a while. 

Meanwhile while this is running I'd like to show you equivalents of it in the industry. So inspired by this a lot of people were interested in cloning it. And so one example is for example perplexity. 

So perplexity when you go through the model drop down has something called deep research. And so you can issue the same queries here. And we can give this to perplexity. 

And then grok as well has something called deep search instead of deep research. But I think that grok's deep search is kind of like deep research but I'm not 100% sure. So we can issue grok deep search as well. 

Grok 3 deep search go. And this model is going to go off as well. Now I think where is my ChachiPT? So ChachiPT is kind of like maybe a quarter done. 

Perplexity is gonna be done soon. Okay still thinking. And grok is still going as well. 

I like grok's interface the most. It seems like okay so basically it's looking up all kinds of papers, WebMD, browsing results and it's kind of just getting all this. Now while this is all going on of course it's accumulating a giant context window and it's processing all that information trying to kind of create a report for us. 

So key points. What is CAKG and why is it in the longevity mix? How is it associated with longevity etc. And so it will do citations and it will kind tell you all about it. 

And so this is not a simple and short response. This is a kind of like almost like a custom research paper on any topic you would like. And so this is really cool and it gives a lot of references potentially for you to go off and do some of your own reading and maybe ask some clarifying questions afterwards. 

But it's actually really incredible that it gives you all these like different citations and processes the information for you a little bit. Now let's see if Perplexity finished. Okay Perplexity is still still researching and ChachiPT is also researching. 

So let's briefly pause the video and I'll come back when this is done. Okay so Perplexity finished and we can see some of the report that it wrote up. So there's some references here and some basically description. 

And then ChachiPT also finished and it also thought for five minutes, looked at 27 sources and produced a report. So here it talked about research in worms, Drosophila in mice and human trials that are ongoing. And then a proposed mechanism of action and some safety and potential concerns and references which you can dive deeper into. 

So usually in my own work right now I've only used this maybe for like 10 to 20 queries so far, something like that. Usually I find that the ChachiPT offering is currently the best. It is the most thorough. 

It reads the best. It is the longest. It makes most sense when I read it. 

And I think the Perplexity and the Grok are a little bit shorter and a little bit briefer and don't quite get into the same detail as the deep research from Google, from ChachiPT right now. I will say that everything that is given to you here, again keep in mind that even though it is doing research and it's pulling stuff in, there are no guarantees that there are no hallucinations here. Any of this can be hallucinated at any point in time. 

It can be made up, fabricated, misunderstood by the model. So that's why these citations are really important. Treat this as your first draft. 

Treat this as papers to look at. But don't take this as definitely true. So here what I would do now is I would actually go into these papers and I would try to understand is ChachiPT understanding it correctly? And maybe I have some follow-up questions, etc. 

So you can do all that. But still incredibly useful to see these reports once in a Okay. So just like before, I wanted to show a few brief examples of how I've used deep research.

So for example, I was trying to change a browser because Chrome upset me. And so it deleted all my tabs. So I was looking at either Brave or Arc and I was most interested in which one is more private. 

And basically ChachiPT compiled this report for me. And this was actually quite helpful. And I went into some of the sources and I sort of understood why Brave is basically TLDR significantly better. 

And that's why, for example, here I'm using Brave because I've switched to it now. And so this is an example of basically researching different kinds of products and comparing them. I think that's a good fit for deep research. 

Here I wanted to know about a life extension in mice. So it kind of gave me a very long reading, but basically mice are an animal model for longevity. And different labs have tried to extend it with various techniques.

And then here I wanted to explore LLM labs in the USA. And I wanted a table of how large they are, how much funding they've had, et cetera. So this is the table that it produced. 

Now this table is basically hit and miss, unfortunately. So I wanted to show it as an example of a failure. I think some of these numbers, I didn't fully check them, but they don't seem way too wrong.

Some of this looks wrong. But the big omission I definitely see is that XAI is not here, which I think is a really major omission. And then also conversely, Hugging Face should probably not be here because I asked specifically about LLM labs in the USA. 

And also Eleuther AI, I don't think should count as a major LLM lab due to mostly its resources. And so I think it's kind of a hit and miss. Things are missing. 

I don't fully trust these numbers. I'd have to actually look at them. And so again, use it as a first draft. 

Don't fully trust it. Still very helpful. That's it. 

So what's really happening here that is interesting is that we are providing the LLM with additional concrete documents that it can reference inside its context window. So the model is not just relying on the knowledge, the hazy knowledge of the world through its parameters and what it knows in its brain. We're actually giving it concrete documents. 

It's as if you and I reference specific documents like on the internet or something like that while we are kind of producing some answer for some question. Now we can do that through an internet search or like a tool like this, but we can also provide these LLMs with concrete documents ourselves through a file upload. And I find this functionality pretty helpful in many ways. 

So as an example, let's look at Cloud because they just released Cloud 3.7 while I was filming this video. So this is a new Cloud model that is now the state of the art. And notice here that we have thinking mode now as a 3.7. And so normal is what we looked at so far, but they just released extended best for math and coding challenges. 

And what they're not saying, but it's actually true under the hood, probably most likely is that this was trained with reinforcement learning in a similar way that all the other thinking models were produced. So what we can do now is we can upload the documents that we wanted to reference inside its context window. So as an example, there's this paper that came out that I was kind of interested in. 

It's from Arc Institute and it's basically a language model trained on DNA. And so I was kind of curious, I mean, I'm not from biology, but I was kind of curious what this is. And this is a perfect example of what LLMs are extremely good for because you can upload these documents to the LLM and you can load this PDF into the context window and then ask questions about it and basically read the documents together with an LLM and ask questions off it. 

So the way you do that is you basically just drag and drop. So we can take that PDF and just drop it here. This is about 30 megabytes. 

Now, when Cloud gets this document, it is very likely that they actually discard a lot of the images and that kind of information. I don't actually know exactly what they do under the hood and they don't really talk about it, but it's likely that the images are thrown away or if they are there, they may not be as well understood as you and I would understand them potentially. And it's very likely that what's happening under the hood is that this PDF is basically converted to a text file and that text file is loaded into the token window. 

And once it's in the token window, it's in the working memory and we can ask questions of it. So typically when I start reading papers together with any of these LLMs, I just ask for, can you give me a summary of this paper? Let's see what Cloud 3.7 says. Okay. 

I'm exceeding the length limit of this chat. Oh God. Really? Oh, damn. 

Okay. Well, let's try chat GPT. Can you summarize this paper? And we're using GPT 4.0 and we're not using thinking, which is okay. 

We can start by not thinking. Reading documents. Summary of the paper. 

Genome modeling and design across all domains of life. So this paper introduces Evo 2 large-scale biological foundation model and then key features and so on. So I personally find this pretty helpful. 

And then we can kind of go back and forth. And as I'm reading through the abstract and the introduction, et cetera, I am asking questions of the LLM and it's kind of like making it easier for me to understand the paper. Another way that I like to use this functionality extensively is when I'm reading books. 

It is rarely ever the case anymore that I read books just by myself. I always involve an LLM to help me read a book. So a good example of that recently is The Wealth of Nations, which I was reading recently. 

And it is a book from 1776 written by Adam Smith. And it's kind of like the foundation of classical economics. And it's a really good book. 

And it's kind of just very interesting to me that it was written so long ago, but it has a lot of modern day kind of like, it's just got a lot of insights that I think are very timely even today. So the way I read books now as an example is you basically pull up the book and you have to get access to like the raw content of that information. In the case of Wealth of Nations, this is easy because it is from 1776. 

So you can just find it on Wealth Project Gutenberg as an example. And then basically find the chapter that you are currently reading. So as an example, let's read this chapter from book one. 

And this chapter I was reading recently, and it kind of goes into the division of labor and how it is limited by the extent of the market. Roughly speaking, if your market is very small, then people can't specialize. And a specialization is what is basically huge. 

Specialization is extremely important for wealth creation because you can have experts who specialize in their simple little task. But you can only do that at scale because without the scale, you don't have a large enough market to sell to your specialization. So what we do is we copy paste this book, this chapter at least. 

This is how I like to do it. We go to say Claude. And we say something like, we are reading the Wealth of Nations. 

Now remember, Claude has knowledge of the Wealth of Nations, but probably doesn't remember exactly the content of this chapter. So it wouldn't make sense to ask Claude questions about this chapter directly because he probably doesn't remember what the chapter is about. But we can remind Claude by loading this into the context window. 

So we're reading the Wealth of Nations. Please summarize this chapter to start. And then what I do here is I copy paste. 

Now in Claude, when you copy paste, they don't actually show all the text inside the text box. They create a little text attachment when it is over some size. And so we can click enter. 

And we just kind of like start off. Usually I like to start off with a summary of what this chapter is about just so I have a rough idea. And then I go in and I start reading the chapter. 

And if at any point we have any questions, just come in and just ask our question. And I find that basically going hand in hand with LLMs dramatically increases my retention, my understanding of these chapters. And I find that this is especially the case when you're reading, for example, documents from other fields, like for example, biology, or for example, documents from a long time ago, like 1776, where you sort of need a little bit of help of even understanding the basics of the language.

Or for example, I would feel a lot more courage approaching a very old text that is outside of my area of expertise. Maybe I'm reading Shakespeare, or I'm reading things like that. I feel like LLMs make a lot of reading very dramatically more accessible than it used to be before, because you're not just right away confused, you can actually kind of go through it and figure it out together with the LLM in hand. 

So I use this extensively. And I think it's extremely helpful. I'm not aware of tools, unfortunately, that make this very easy for you. 

Today, I do this clunky back and forth. So literally, I will find the book somewhere. And I will copy paste stuff around.

And I'm going back and forth. And it's extremely awkward and clunky. And unfortunately, I'm not aware of a tool that makes this very easy for you. 

But obviously, what you want is as you're reading a book, you just want to highlight the passage and ask questions about it. This currently, as far as I know, does not exist. But this is extremely helpful. 

I encourage you to experiment with it. And don't read books alone. Okay, the next very powerful tool that I now want to turn to is the use of a Python interpreter, or basically giving the ability to the LLM to use and write computer programs. 

So instead of the LLM giving you an answer directly, it has the ability now to write a computer program, and to emit special tokens that the ChachiPT application recognizes as, hey, this is not for the human. This is basically saying that whatever I output it here is actually a computer program, please go off and run it and give me the result of running that computer program. So it is the integration of the language model with a programming language here, like Python. 

So this is extremely powerful. Let's see the simplest example of where this would be used and what this would look like. So if I go to ChachiPT and I give it some kind of a multiplication problem, let's say 30 times 9 or something like that.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Then this is a fairly simple multiplication, and you and I can probably do something like this in our head, right? Like 30 times nine, you can just come up with the result of 270, right? So let's see what happens. Okay, so LLM did exactly what I just did. It calculated the result of the multiplication to be 270, but it's actually not really doing math.

It's actually more like almost memory work, but it's easy enough to do in your head. So there was no tool use involved here. All that happened here was just the zip file doing next token prediction and gave the correct result here in its head.

The problem now is what if we want something more complicated? So what is this times this? And now, of course, this, if I asked you to calculate this, you would give up instantly because you know that you can't possibly do this in your head, and you would be looking for a calculator. And that's exactly what the LLM does now too. And OpenAI has trained ChatGPT to recognize problems that it cannot do in its head and to rely on tools instead.

So what I expect ChatGPT to do for this kind of a query is to turn to tool use. So let's see what it looks like. Okay, there we go.

So what's opened up here is what's called the Python interpreter. And Python is basically a little programming language. And instead of the LLM telling you directly what the result is, the LLM writes a program and then not shown here are special tokens that tell the ChatGPT application to please run the program.

And then the LLM pauses execution. Instead, the Python program runs, creates a result, and then passes this result back to the language model as text. And the language model takes over and tells you that the result of this is that.

So this is tool use, incredibly powerful. And OpenAI has trained ChatGPT to kind of like know in what situations to lean on tools. And they've taught it to do that by example.

So human labelers are involved in curating data sets that kind of tell the model by example in what kinds of situations it should lean on tools and how. But basically, we have a Python interpreter, and this is just an example of multiplication. But this is significantly more powerful.

So let's see what we can actually do inside programming languages. Before we move on, I just wanted to make the point that unfortunately you have to kind of keep track of which LLMs that you're talking to have different kinds of tools available to them. Because different LLMs might not have all the same tools.

And in particular, LLMs that do not have access to the Python interpreter or a programming language or are unwilling to use it might not give you correct results in some of these harder problems. So as an example, here we saw that ChatGPT correctly used a programming language and didn't do this in its head. Grok3 actually, I believe, does not have access to a programming language like a Python interpreter.

And here, it actually does this in its head and gets remarkably close. But if you actually look closely at it, it gets it wrong. This should be one, two, zero instead of zero, six, zero.

So Grok3 will just hallucinate through this multiplication and do it in its head and get it wrong, but actually remarkably close. Then I tried Clod and Clod actually wrote, in this case, not Python code, but it wrote JavaScript code. But JavaScript is also a programming language and gets the correct result.

Then I came to Gemini and I asked 2.0 Pro and Gemini did not seem to be using any tools. There's no indication of that. And yet it gave me what I think is the correct result, which actually kind of surprised me.

So Gemini, I think, actually calculated this in its head correctly. And the way we can tell that this is, which is kind of incredible, the way we can tell that it's not using tools is we can just try something harder. What is, we have to make it harder for it.

Okay, so it gives us some result and then I can use my calculator here and it's wrong. So this is using my MacBook Pro calculator. And two, it's not correct, but it's remarkably close, but it's not correct.

But it will just hallucinate the answer. So I guess like my point is, unfortunately, the state of the LLMs right now is such that different LLMs have different tools available to them and you kind of have to keep track of it. And if they don't have the tools available, they'll just do their best, which means that they might hallucinate a result for you.

So that's something to look out for. Okay, so one practical setting where this can be quite powerful is what's called ChatGPT Advanced Data Analysis. And as far as I know, this is quite unique to ChatGPT itself.

And it basically gets ChatGPT to be kind of like a junior data analyst who you can kind of collaborate with. So let me show you a concrete example without going into the full detail. So first we need to get some data that we can analyze and plot and chart, et cetera.

So here in this case, I said, let's research open AI evaluation as an example. And I explicitly asked ChatGPT to use the search tool because I know that under the hood, such a thing exists. And I don't want it to be hallucinating data to me.

I want it to actually look it up and back it up and create a table where each year we have the evaluation. So these are the open AI evaluations over time. Notice how in 2015, it's not applicable.

So the valuation is like unknown. Then I said, now plot this, use log scale for y-axis. And so this is where this gets powerful.

ChatGPT goes off and writes a program that plots the data over here. So it created a little figure for us and it sort of ran it and showed it to us. So this can be quite nice and valuable because it's very easy way to basically collect data, upload data in a spreadsheet, visualize it, et cetera.

I will note some of the things here. So as an example, notice that we had NA for 2015, but ChatGPT when it was writing the code, and again, I would always encourage you to scrutinize the code, it put in 0.1 for 2015. And so basically it implicitly assumed that it made the assumption here in code that the valuation at 2015 was 100 million and because it put in 0.1. And it's kind of like did it without telling us.

So it's a little bit sneaky and that's why you kind of have to pay attention a little bit to the code. So I'm familiar with the code and I always read it, but I think I would be hesitant to potentially recommend the use of these tools if people aren't able to like read it and verify it a little bit for themselves. Now, fit a trendline and extrapolate until the year 2030.

Mark the expected valuation in 2030. So it went off and it basically did a linear fit and it's using SciPy's curve fit. And it did this and came up with a plot and it told me that the valuation based on the trend in 2030 is approximately 1.7 trillion, which sounds amazing except here I became suspicious because I see that Chachapiti is telling me it's 1.7 trillion, but when I look here at 2030, it's printing 20271.7b. So it's extrapolation when it's printing the variable is inconsistent with 1.7 trillion.

This makes it look like that valuation should be about 20 trillion. And so that's what I said, print this variable directly by itself, what is it? And then it sort of like rewrote the code and gave me the variable itself. And as we see in the label here, it is indeed 20271.7b, et cetera.

So in 2030, the true exponential trend extrapolation would be a valuation of 20 trillion. So I was like, I was trying to confront Chachapiti and I was like, you lied to me, right? And it's like, yeah, sorry, I messed up. So I guess I like this example because number one, it shows the power of the tool and that it can create these figures for you.

And it's very nice. But I think number two, it shows the trickiness of it where, for example, here it made an implicit assumption and here it actually told me something. It told me just the wrong, it hallucinated 1.7 trillion.

So again, it is kind of like a very, very junior data analyst. It's amazing that it can plot figures, but you have to kind of still know what this code is doing and you have to be careful and scrutinize it and make sure that you are really watching very closely because your junior analyst is a little bit absent-minded and not quite right all the time. So really powerful, but also be careful with this.

I won't go into full details of advanced data analysis, but there were many videos made on this topic. So if you would like to use some of this in your work, then I encourage you to look at some of these videos. I'm not going to go into the full detail.

So a lot of promise, but be careful. Okay, so I've introduced you to Chats GPT and advanced data analysis, which is one powerful way to basically have LLMs interact with code and add some UI elements like showing a figures and things like that. I would now like to introduce you to one more related tool and that is specific to Cloud and it's called Artifacts.

So let me show you by example what this is. So you're having a conversation with Cloud and I'm asking generate 20 flashcards from the following text. And for the text itself, I just came to the Adam Smith Wikipedia page, for example, and I copy pasted this introduction here.

So I copy pasted this here and asked for flashcards and Cloud responds with 20 flashcards. So for example, when was Adam Smith baptized on June 16th, et cetera? When did he die? What was his nationality, et cetera? So once we have the flashcards, we actually want to practice these flashcards. And so this is where I continue to use the conversation and I say, now use the Artifacts feature to write a flashcards app to test these flashcards.

And so Cloud goes off and writes code for an app that basically formats all of this into flashcards. And that looks like this. So what Cloud wrote specifically was this code here.

So it uses a React library and then basically creates all these components. It hard codes the Q&A into this app and then all the other functionality of it. And then the Cloud interface basically is able to load these React components directly in your browser.

And so you end up with an app. So when was Adam Smith baptized? And you can click to reveal the answer. And then you can say whether you got it correct or not.

When did he die? What was his nationality, et cetera. So you can imagine doing this and then maybe we can reset the progress or shuffle the cards, et cetera. So what happened here is that Cloud wrote us a super duper custom app just for us right here.

And typically what we're used to is some software engineers write apps, they make them available, and then they give you maybe some way to customize them or maybe to upload flashcards. Like for example, the Anki app, you can import flashcards and all this kind of stuff. This is a very different paradigm because in this paradigm, Cloud just writes the app just for you and deploys it here in your browser.

Now, keep in mind that a lot of apps that you will find on the internet, they have entire backends, et cetera. There's none of that here. There's no database or anything like that, but these are like local apps that can run in your browser and they can get fairly sophisticated and useful in some cases.

So that's Cloud Artifacts. Now, to be honest, I'm not actually a daily user of Artifacts. I use it once in a while.

I do know that a large number of people are experimenting with it and you can find a lot of Artifacts showcases because they're easy to share. So these are a lot of things that people have developed, various timers and games and things like that. But the one use case that I did find very useful in my own work is basically the use of diagrams, diagram generation.

So as an example, let's go back to the book chapter of Adam Smith that we were looking at. What I do sometimes is, we are reading The Wealth of Nations by Adam Smith. I'm attaching chapter three and book one.

Please create a conceptual diagram of this chapter. And when Cloud hears conceptual diagram of this chapter, very often it will write a code that looks like this. And if you're not familiar with this, this is using the mermaid library to basically create or define a graph.

And then this is plotting that mermaid diagram. And so Cloud analyzed the chapter and figures out that, okay, the key principle that's being communicated here is as follows, that basically division of labor is related to the extent of the market, the size of it. And then these are the pieces of the chapter.

So there's the comparative example of trade and how much easier it is to do on land and on water and the specific example that's used. And that geographic factors actually make a huge difference here. And then the comparison of land transport versus water transport and how much easier water transport is.

And then here we have some early civilizations that have all benefited from basically the availability of water transport and have flourished as a result of it because they support specialization. So if you're a conceptual kind of like visual thinker, and I think I'm a little bit like that as well, I like to lay out information as like a tree like this. And it helps me remember what that chapter is about very easily.

And I just really enjoy these diagrams and like kind of getting a sense of like, okay, what is the layout of the argument? How is it arranged spatially? And so on. And so if you're like me, then you will definitely enjoy this. And you can make diagrams of anything, of books, of chapters, of source codes, of anything really.

And so I specifically find this fairly useful. Okay, so I've shown you that LLMs are quite good at writing code. So not only can they emit code, but a lot of the apps like ChatGPT and Cloud and so on have started to like partially run that code in the browser.

So ChatGPT will create figures and show them and Cloud Artifacts will actually like integrate your React component and allow you to use it right there in line in the browser. Now, actually majority of my time personally and professionally is spent writing code. But I don't actually go to ChatGPT and ask for snippets of code because that's way too slow.

Like ChatGPT just doesn't have the context to work with me professionally to create code. And the same goes for all the other LLMs. So instead of using features of these LLMs in a web browser, I use a specific app.

And I think a lot of people in the industry do as well. And this can be multiple apps by now, VS Code, Windsurf, Cursor, et cetera. So I like to use Cursor currently.

And this is a separate app you can get for your, for example, MacBook. And it works with the files on your file system. So this is not a web inter, this is not some kind of a webpage you go to.

This is a program you download and it references the files you have on your computer. And then it works with those files and edits them with you. So the way this looks is as follows.

Here I have a simple example of a React app that I built over a few minutes with Cursor. And under the hood, Cursor is using Cloud 3.7 Sonnet. So under the hood, it is calling the API of Anthropic and asking Cloud to do all of this stuff.

But I don't have to manually go to Cloud and copy paste chunks of code around. This program does that for me and has all of the context of the files in the directory and all this kind of stuff. So the app that I developed here is a very simple tic-tac-toe as an example.

And Cloud wrote this in a few, probably a minute. And we can just play. X can win.

Or we can tie. Oh, wait, sorry, I accidentally won. You can also tie.

And I'd just like to show you briefly, this is a whole separate video of how you would use Cursor to be efficient. I just want you to have a sense that I started from a completely new project and I asked the Composer app here, as it's called, the Composer feature, to basically set up a new React repository, delete a lot of the boilerplate, please make a simple tic-tac-toe app. And all of this stuff was done by Cursor.

I didn't actually really do anything except for write five sentences. And then it changed everything and wrote all the CSS, JavaScript, et cetera. And then I'm running it here and hosting it locally and interacting with it in my browser.

So that's Cursor. It has the context of your apps and it's using Cloud remotely through an API without having to access the webpage. And a lot of people, I think, develop in this way at this time.

And these tools have become more and more elaborate. So in the beginning, for example, you could only like say change, like, oh, Control-K, please change this line of code to do this or that. And then after that, there was a Control-L, Command-L, which is, oh, explain this chunk of code.

And you can see that there's gonna be an LLM explaining this chunk of code. And what's happening under the hood is it's calling the same API that you would have access to if you actually did enter here. But this program has access to all the files.

So it has all the context. And now what we're up to is not Command-K and Command-L. We're now up to Command-I, which is this tool called Composer.

And especially with the new agent integration, the Composer is like an autonomous agent on your code base. It will execute commands. It will change all the files as it needs to.

It can edit across multiple files. And so you're mostly just sitting back and you're giving commands. And the name for this is called Vibe Coding, a name with that I think I probably minted.

And Vibe Coding just refers to letting, giving in, giving control to Composer and just telling it what to do and hoping that it works. Now, worst comes to worst, you can always fall back to the good old programming because we have all the files here. We can go over all the CSS and we can inspect everything.

And if you're a programmer, then in principle you can change this arbitrarily. But now you have a very helpful system that can do a lot of the low-level programming for you. So let's take it for a spin briefly.

Let's say that when either X or O wins, I want confetti or something. And let's just see what it comes up with. Okay, I'll add a confetti effect when a player wins the game.

It wants me to run React Confetti, which apparently is a library that I didn't know about. So we'll just say, okay. It installed it, and now it's going to update the app.

So it's updating app.tsx, the TypeScript file, to add the confetti effect when a player wins. And it's currently writing the code, so it's generating. And we should see it in a bit.

Okay, so it basically added this chunk of code. And a chunk of code here, and a chunk of code here. And then we'll also add some additional styling to make the winning cells stand out.

Okay, still generating. Okay, and it's adding some CSS for the winning cells. So honestly, I'm not keeping full track of this.

It imported React Confetti. This all seems pretty straightforward and reasonable, but I'd have to actually really dig in. Okay, it wants to add a sound effect when a player wins, which is pretty ambitious, I think.

I'm not actually 100% sure how it's going to do that, because I don't know how it gains access to a sound file like that. I don't know where it's going to get the sound file from. But every time it saves a file, we actually are deploying it.

So we can actually try to refresh and just see what we have right now. So, oh, so it added a new effect. You see how it kind of like fades in, which is kind of cool.

And now we'll win. Whoa, okay. Didn't actually expect that to work.

This is really elaborate now. Let's play again. Whoa.

Okay. Oh, I see. So it actually paused and it's waiting for me.

So it wants me to confirm the command. So make public sounds. I had to confirm it explicitly.

Let's create a simple audio component to play Victory Sound. Sound slash Victory MP3. The problem with this will be the Victory.mp3 doesn't exist.

So I wonder what it's going to do. It's downloading it. It wants to download it from somewhere.

Let's just go along with it. Let's add a fallback in case the sound file doesn't exist. In this case, it actually does exist.

And yep, we can get add and we can basically create a git commit out of this. Okay, so the composer thinks that it is done. So let's try to take it for a spin.

Okay. So yeah, pretty impressive. I don't actually know where it got the sound file from.

I don't know where this URL comes from, but maybe this just appears in a lot of repositories and sort of cloud kind of like knows about it. But I'm pretty happy with this. So we can accept all and that's it.

And then as you can get a sense of, we could continue developing this app and worst comes to worst, if we can't debug anything, we can always fall back to a standard programming instead of a vibe coding. Okay, so now I would like to switch gears again. Everything we've talked about so far had to do with interacting with the model via text.

So we type text in and it gives us text back. What I'd like to talk about now is to talk about different modalities. That means we want to interact with these models in more native human formats.

So I want to speak to it and I want it to speak back to me and I want to give images or videos to it and vice versa. I want it to generate images and videos back. So it needs to handle the modalities of speech and audio and also of images and video.

So the first thing I want to cover is how can you very easily just talk to these models? So I would say roughly in my own use, 50% of the time I type stuff out on the keyboard and 50% of the time I'm actually too lazy to do that and I just prefer to speak to the model. And when I'm on mobile, on my phone, that's even more pronounced. So probably 80% of my queries are just speech because I'm too lazy to type it out on the phone.

Now on the phone, things are a little bit easy. So right now the Chatshubpt app looks like this. The first thing I want to cover is there are actually like two voice modes.

You see how there's a little microphone and then here there's like a little audio icon. These are two different modes and I will cover both of them. First, the audio icon, sorry, the microphone icon here is what will allow the app to listen to your voice and then transcribe it into text.

So you don't have to type out the text. It will take your audio and convert it into text. So on the app, it's very easy and I do this all the time.

As you open the app, create a new conversation and I just hit the button and why is the sky blue? Is it because it's reflecting the ocean or yeah, why is that? And I just click okay. And I don't know if this will come out but it basically converted my audio to text and I can just hit go and then I get a response. So that's pretty easy.

Now on desktop, things get a little bit more complicated for the following reason. When we're in the desktop app, you see how we have the audio icon and it says use voice mode. We'll cover that in a second but there's no microphone icon.

So I can't just speak to it and have it transcribed to text inside this app. So what I use all the time on my MacBook is I basically fall back on some of these apps that allow you that functionality but it's not specific to ChatGPT. It is a system wide functionality of taking your audio and transcribing it into text.

So some of the apps that people seem to be using are SuperWhisper, WhisperFlow, MacWhisper, et cetera. The one I'm currently using is called SuperWhisper and I would say it's quite good. So the way this looks is you download the app, you install it on your MacBook and then it's always ready to listen to you.

So you can bind a key that you wanna use for that. So for example, I use F5. So whenever I press F5, it will listen to me then I can say stuff and then I press F5 again and it will transcribe it into text.

So let me show you. I'll press F5. I have a question.

Why is the sky blue? Is it because it's reflecting the ocean? Okay, right there. Enter. I didn't have to type anything.

So I would say a lot of my queries, probably about half are like this because I don't want to actually type this out. Now, many of the queries will actually require me to say product names or specific like library names or like various things like that that don't often transcribe very well. In those cases, I will type it out to make sure it's correct.

But in very simple day-to-day use, very often I am able to just speak to the model. So, and then it will transcribe it correctly. So that's basically on the input side.

Now on the output side, usually with an app, you will have the option to read it back to you. So what that does is it will take this text and it will pass it to a model that does the inverse of taking text to speech. And in ChachiPT, there's this icon here that says Read Aloud.

So we can press it. No, the sky is not blue because it reflects the ocean. That's a common myth.

The real reason the sky is blue is due to Rayleigh scattering. Okay, so I'll stop it. So different apps like ChachiPT or Cloud or Gemini or whatever you are using may or may not have this functionality, but it's something you can definitely look for.

When you have the input be system-wide, you can, of course, turn speech into text in any of the apps. But for reading it back to you, different apps may or may not have the option. And, or you could consider downloading speech-to-text, sorry, a text-to-speech app that is system-wide like these ones and have it read out loud.

So those are the options available to you and something I wanted to mention. And basically the big takeaway here is don't type stuff out, use voice. It works quite well.

And I use this pervasively. And I would say roughly half of my queries, probably a bit more are just audio because I'm lazy and it's just so much faster. Okay, but what we've talked about so far is what I would describe as fake audio.

And it's fake audio because we're still interacting with the model via text. We're just making it faster because we're basically using either a speech-to-text or a text-to-speech model to pre-process from audio to text and from text to audio. So it's not really directly done inside the language model.

So, however, we do have the technology now to actually do this actually as true audio handled inside the language model. So what actually is being processed here was text tokens, if you remember. So what you can do is you can truncate different modalities like audio in a similar way as you would truncate text into tokens.

So typically what's done is you basically break down the audio into a spectrogram to see all the different frequencies present in the audio. And you go in little windows and you basically quantize them into tokens. So you can have a vocabulary of 100,000 possible little audio chunks.

And then you actually train the model with these audio chunks so that it can actually understand those little pieces of audio. And this gives the model a lot of capabilities that you would never get with this fake audio as we've talked about so far. And that is what this other button here is about.

This is what I call true audio, but sometimes people will call it by different names. So as an example, when you go to chat GPT pricing, you have to look for this thing called advanced voice mode. So advanced voice mode is referring to true audio.

What that means is that the voice is handled natively inside the language model. The model can understand audio chunks and predict audio chunks. So it can hear and speak directly in audio.

There's no text involved whatsoever. So it's quite magical. It's really nice.

And let me take it for a spin. Okay, so here's what this will look like. Hey, how's it going? Hey, Andre, I'm doing great.

How about you? I'm great. I'm wondering why the sky is blue. Is it because it's reflecting the ocean? Glad to hear you're great.

Actually, the sky is blue because of a phenomenon called Rayleigh scattering. Okay, okay. Instead of speaking to me normally, can you speak in the voice of Yoda so that you...

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)