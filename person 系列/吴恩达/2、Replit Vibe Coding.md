
五种氛围编程的技能：思考（Thinking）、框架（Frameworks）、检查点（checkpoints）、调试（debugging）、上下文（context）

我们将把这比作一盘国际象棋，

*Thinking：*

* 逻辑思维（logical）：提出问题，国际象棋是什么？
* 分析性思维（analytical）：分解问题并学习如何解决它们？
* 计算思维（computational）：国际象棋游戏背后的模式是什么？可能会引导你做一些事情，比如编写一个程序来执行国际象棋的规则
* 程序思维（procedural）：比如编写一个能下竞技象棋的计算机程序

*Frameworks：*

* 你不知道有什么是你不知道的
* 我想做的事情应该如何做
* 什么是框架
	* ...允许我做的事情？
	* ...用 LLMs 的最佳实践？
* 如果你不知道...只需要问！

*checkpoints/versions：*

* 系统总会出故障——这是事实
* 应该用版本控制
* 我们将把构建过程分解成小块，以短冲刺的方式快速推进。

*debugging：*

*context*：

* 上下文窗口：LLM 在给定时间内可以处理的标记数量。
* 上下文可以是我们提供给 LLM 的提示，但也可以是其他内容
	* 图像
	* 文档
	* 错误
	* 关于你的应用程序/环境/偏好的详细信息（！）

因为 LLM 可能有过时的训练数据（或缺乏我们实现的细节），我们需要提供额外的上下文。


----

**实现 MVP**

- 仅向 AI 提供与 MVP 相关的信息
- 从小处开始，逐步提升
- 提供基础的上下文和重要细节

**实现新功能**

- 提供与新功能相关的上下文
- 提到框架，提供详细的实现文档
- 进行增量更改（检查点）

**调试错误**

- 弄清楚事情如何运作
- 找出问题所在
- 弄清楚如何向 LLM 提供信息以解决问题
    - 弄清楚如何引导上下文

---

在本课程中，我们将构建一个网站性能分析器。您将学习如何检查HTML内容并理解网站结构，也就是要自动化的核心概念。在利用Replit的AI助手构建首个原型前，我们会先创建产品需求文档（PRD）和线框图。正如刚才所述，我们将开发一个SEO分析工具，您可以输入任意网站URL来查看页面加载时间和标签信息。我们还会详细解释这些指标的具体含义及其重要性。

这让我们能够检查网站是否正确实施，并深入了解我们的SEO最佳实践。因此，当我们讨论构建产品需求文档（PRD）或线框图时，这就是我们所指的内容。接下来，我将详细讲解我面前的这份资料。这里使用的是在线白板工具，我认为它非常重要。这是一个互动工具，允许你输入网站URL，查看网站加载时间和标签的可视化表示，以确保我们构建的内容配置正确，并检查网站是否正确实施。

作为延伸目标，我们可能会将最近分析的网站存储到数据库中，这里涉及的技能包括提示、处理和分析HTML内容，以及创建直观的数据可视化。同时还需要处理错误和边缘情况，以及存储问题。因此，在左侧我花了一些时间绘制了我们将要构建的模型草图。用户主要交互入口是一个简单的文本框，可以在其中输入URL。通过输入该URL，我们希望应用程序能返回我们输入网站的相关推荐。

例如：嘿，你没有给你的网站添加标题，或者没有为网站添加描述，这会让谷歌难以找到你的网站，或者让人们难以了解你的网站是关于什么的。这些元标签嵌入在每个网站中。因此，当你构建网站或部署应用程序时，考虑这些事情实际上非常重要。重要的是要考虑当人们搜索你的网站时，它的预览如何显示。而这正是我们的应用程序将要让我们实现的功能。它将允许我们分析任何网站并获取这些信息。

因此，当我们谈论氛围编码或顺其自然时，进行这样的练习通常更有帮助：明确我们想要什么，将其概念化、可视化。这样我们就不会在毫无概念的情况下使用工具，或者至少能更清楚地了解我们想要达到的目标。那么，让我们来谈谈这如何转化为Replit Agent的初始提示。我的提示是：帮我创建一个交互式应用程序，以交互和可视化的方式显示任何网站的SEO或元标签，以检查它们是否正确实施。

请注意，在第一句话中，这属于特定领域的知识。你可能不知道什么是SEO标签，也可能不了解元标签的含义。但AI了解这些概念，通过使用这些术语，并大致理解我们可能不熟悉的领域框架，我们能够精准引导Replit Agent完成构建任务。我们提示的下一部分内容是：该应用应能抓取网站的HTML代码，然后根据SEO优化的最佳实践提供关于SEO标签的反馈。此外，应用还应提供谷歌和社交媒体预览效果。

现在，我认为最后一点非常重要，因为我做过一些SEO优化的工作，这一点我特别想可视化展示，因为它可以非常直观。但请注意，我们要求应用程序获取网站的HTML，这实际上是在抓取网站。所以，对于任何网站，我们都会抓取它，然后获取其内容。我们希望我们的应用程序能返回反馈，对吧？我们希望在我们所做的事情上获得可操作的见解。

因此，我们正在仔细斟酌语言和措辞，以便为Agent提供能够获得这些结果的表达方式。需要注意的是，我知道SEO标签是嵌入在网站的HTML中的，任何人都可以访问它们。这就是为什么我知道这个应用程序有很大可能会成功。为了更直观地展示我们正在构建的内容，当我们在Google上搜索DeepLearning.AI时，返回的结果实际上是网站的SEO标题，而描述则列在下方。

因此，我们可以在控制台或页面的实际底层HTML中看到这一点。如果我点击页面，然后右键点击并选择“检查”，我们将获得所谓的浏览器控制台。这实际上让我们能够做一些相当酷的事情，或者更准确地说，是开发者工具。所以如果我点击页面，在那里右键点击并选择“检查”，我将访问我们的开发者工具。这可能看起来有点吓人。这只是网站的底层HTML。现在，如果你进入HTML的head部分，你会看到一大堆东西。



You're gonna see
actually things called meta tags. And one of those is the title. DeepLearning.AI start or advance your career in AI, as well as the description, join over 7 million people learning how to use
and build AI through our online courses. These are the tags
that we're going to extract to understand if our website is properly implemented. So we've talked about our prompt. We've gone over our wireframe in our PRD. We're going to type in our prompt
and just click start building on Replit. Now we're going to see some things
start happening. I'm going to walk through this
because it's our first course and discuss
how Agent builds these applications. Fundamentally Agents different
from other types of vibe coding apps because Agent produces a plan. The plan is going to help us understand
what we're building and confirm that we're taking the right actions, or we're moving
kind of closer to what we actually want. So Agent is going to analyze that prompt
and present a plan to us. Says it's going to build the initial prototype and ask if it wants
if we want any follow-up features. In the initial prototype, it'll also mention if it's using
any frameworks or integrations. I'm going to approve this plan and start
and then talk a little bit more about that. So it's important to mention that Agent
does come with a ton of integrations, So, if you'd said for example: build me an app
that uses the Anthropic API. Or: build me an app
that has Stripe payments. We support those integrations natively, and Agent
should mention that in the app plan. Now, what's happening on the right
is that in real time, Agent is designing a visual preview
of our app, and the goal there is to get you something
that you can use as fast as possible, or something that you can visualize
rather as fast as possible. It's important to note
this is not the actual application. And so we can see, you know,
kind of closer to our wireframe. Okay, the main entry point
is just this website URL. And from there
we're going to be able to analyze it. Now once Agent is done with that visual
preview, it actually is mentioning to us up here. It's creating a fully interactive version
of our app. And so underneath
the hood Agent is scaffolding this project, installing packages,
configuring your environment. And this is what we talked about
at the beginning. Right? Replit is more than just Agent
and assistant. It's actually an entire environment,
an entire workspace. So while we're getting this set up,
I'm going to walk through the workspace. If you go over here up into the top left, you can take a look at the files
that exist in the workspace. And as Agent continues building,
these are going to be populated, with the folders and the directories
that the project contains. Right? And so it's important really, again,
to drive home that this is an entire interactive
workspace in the cloud. And any time, if you're familiar
with programing concepts, if you want to open something like a shell
or a console, you can do that. And we have access to that. Now, you also don't have to do these things
if you're not familiar with programing. We're going to build entirely
using natural language. But it is important to mention
that through this new tab interface up top or the All Tools section on the left-hand
side, you can access all of the services and tools available in the workspace,
which include things like integrating git, installing dependencies,
viewing our output. We'll go over exactly what that is storing
secrets. Secrets are basically a very secure way
to store environment variables that require no configuration. And many more things. But going back to our app,
Agent is still working on the fully functional version,
which might take 5 to 10 minutes. And so that's another important thing
to call up as we're building this application. The way Agent is designed is to build
these apps from sort of start to MVP. And that's why we stayed high level. And we're keeping the prompts simple. These runs could take two minutes. It could take 5 or 10 minutes. But for the first implementation
you can expect a fully featured app. As you can see on the left there, Agent
just created a client, a server, and a whole host of other configuration
files. Now we see the Agent is actually writing
and creating files in this interactive pane here. And part of the reason that we do this
is because we want to get you to an MVP as fast as possible, and so there's less
back and forth required with Replit Agent. But along the way, Agent is going to show
you exactly what it's doing. So we can see that
it's writing this file server routes. It already wrote our client index
and some other things. Now, it's not important to dig into
how all of these things work, but what might be useful is to start to pay
attention to how Agent creates these apps. And start to reverse engineer
how the applications work, because it can teach you a lot
about programing. It can teach you a lot about building. For example, I see that there are
folders here client, server, and shared. If you're unfamiliar, a client is typically a front-end
and a server is typically a backend. So it makes sense that clicking
to the server exposes things like routes. If you're not familiar with APIs. This is a way, programmatic way
for applications to communicate. And that happens in our backend. But in the client,
we might see the source file and notice there's like an index HTML file,
things like app.tsx, that might be a main sort of entry point
for our app or index.css. CSS defines how the app is style. And so what I'm trying to drive home here
is that you really don't need to look at any of this. You know, like if this is intimidating for you
to just close that and don't look at it and you know,
like we'll just keep vibe coding. But if you're trying to learn and follow
along as you're building these things, you can start to poke around,
start to pay attention. Oh, hey, Agent is defining this thing,
this component called Social Media Preview. I bet if we go into our components folder
and we look at the stuff that's in here, the Google preview probably defines something that would show us a preview
of what it might look like on Google. So again, Agent is going to, on the first
run, create this app from scratch. Is going to implement the front-end
and the backend. This is a lot of talking. Yes. But what we're going to have
when Agent is done running is an MVP that we can start iterating on. It does that in order to deliver, use
something that just works and that we can kind
of keep punching through or we can keep iterating on
once our application is stood up. And so that's been our workspace overview. We're going to let the Agent
keep building. You'll be able to see
all of the things that it does. And then we'll come back
once we have our MVP and start iterating. This is often how I develop
my first prototypes when I'm building with Replit Agent. Cool. So what you'll notice on the right
is that we have an implemented version of our app. And I want to dig in for a second
to the Agent sort of history chat history
here to see what was happening. So as you saw when we were writing
the application, there are a bunch of file edits
that happened. Any time you can dig into these
and you can even click the drop-down to see exactly what was created
or exactly what changed. Then, Agent went along
and actually installed the dependencies that we needed for our application,
as well as some ancillary dependencies that I might have missed
on the first pass. Then it configured the run button up
top here, to execute our app and cycled through
and actually recognized, hey, we missed a package like this thing's
not going to work unless we install it. Then it did some final checks, made an edit to our storage
implementation, noticed an issue, restarted the application
and fixed that issue. And so one of the great things
about building with with Replit building with Agent
is that you don't have to worry about like some of these like hassles
and getting started. You can let Agent take the reins and kind
of implement some of this functionality. From there, it confirmed exactly
what it did and exactly what it created. Now, this is another good way to learn
about what we're building. Hey, we created a complete data model
for storing and analyzing these tags. We built back end API endpoints. We implemented a front end with a URL
input, results overview and visualization. And we did a bunch of other stuff. So, now we can see if it works. We can start debugging.
We can start testing our app. We have our web view on the right
which is going to show us a preview of our app. Now what's important to note
if you've ever written code, if you've ever built with another tool,
you've probably did that on localhost, which is basically running
something on your local machine. This WebView is going to look similar,
but fundamentally, this app,
this Replit app, is running at an URL, so you can actually access this
from anywhere. You can access it in your browser. If I click this and it would open
this WebView up in a new tab, you could scan it
and access it on your phone in real time as you're developing. Your friends
could access it while you're building. It's important to mention
that once you leave this app, once you close this page,
this URL will go to sleep. This is not a deployment, but it is a development environment
that is live on the internet technically. So fundamentally
we're working with like a real web page. So let's test
this out. Let's see how it works. I'm going to type in my blog
and see what's going on. So I click analyze. We get an SEO summary. 86 out of 100 I guess that's not bad. It might be like a b, a b plus here. I'm not complaining. It seems like we have some some scores. Let's see what's going on here. So Matt Palmer. That's right. Developer marketing simplified.
That's right. It looks like a warning. Our title tag is too short a description. The description is a bit too short. The details there. It looks like
we're getting some best practices, so this is pretty similar
to what we wanted. And we also have
some keyword recommendations. We're getting a kind of a pass there and some other recommendations. So what, our analyzer is telling us is,
hey, if we want our website to show up on search on Google, maybe even an indexing
for LLMs, I don't know. We could optimize our title,
we could improve the description and make it maybe a little bit longer. So this is really great. I'd say it's close to what we want. It looks like we also have some social media previews,
so let's go see what's going on there. So we have a Facebook preview
and it is loading the image properly. And we have a Twitter preview and it's
also loading the image properly there. So what we've done here is implemented
this tool that they can then check to make sure that our website
is displayed properly on social media, and that it's being
indexed properly by SEO. And this was one shot, right? We all we did
was type in that initial prompt. And let's try with another website. Let's say, DeepLearning.AI See what we get. Okay. So I just want to point out
my website had a higher score. I'm sure after I record this Andrew's
website, we'll have a higher score. I have no doubt he's
going to be on top of that. But, just to see that
everything's coming through, we do get our social media previews and we do get our Twitter card
previews and everything else. So this is kind of the part
we're talking about, right? Like now I'm just testing the app. I'm seeing if it does
what I expect it to do. Part of vibe coding, a large part of vibe
coding is having an attention to detail and having, really like dialed
in product sense, for example. I'm not crazy about the spacing here. 83 looks like it's on top of 100. That's like not good enough.
We're going to fix that. We're going to make that better. Some of the other elements down here this looks nice but these are misaligned
past isn't in the middle of this pill. I think a lot of the other sort of visual representations of this of this site
are really good. But, part of building with AI
is being very descriptive about what works, about what doesn't work,
and about what you want to change. Right? And so another example. This,
the padding could probably probably be more padding on this website
for it to look nice. One of the interesting things about web
development is that you want your apps to be responsive. So you can see there,
if we make this wider, the padding actually
looks a little bit better. So what's going on here is probably that
this is an optimized for small screens. We can see that by resizing the WebView
or by going to this handy little screen size toggle here and seeing
what it looks like on an iPhone. Maybe we want to build this for an iPhone. So it's important to check that out,
right? Like, hey, these pills, they're not
really expanding the way I want them to. These tabs,
they might look a little compressed. The icons might not be, aligned properly. And then we don't get the same aspect
ratio for these images. Again, this might seem trivial, but it's
really important to dig into the details. Right? The text is overlapping here. So all that considered we have an MVP. We have something that works. And I think it was pretty impressive
that Agent did it in one shot. So now we're going to follow up and we're going to fix
some of these characteristics. So I'm going to say
make my app fully responsive. This prompt is important. What is responsive mean? Well, what it actually means in web
development is, is that your app responds
to resizing the screen right? And so that will make it mobile friendly. And so saying make my app fully
responsive and mobile-friendly. We're using key terms that are synonymous
with web development techniques that AI is going to understand
and implement pretty well. So I'm going to say make my app fully
responsive and mobile friendly, and fix some of the alignment and padding issues with the buttons and elements. Specifically, fix the centering of the overall SEO score and remove the slash 100. Being very descriptive
about what I want edited here, you might notice
I'm asking for a couple things. As we talked about,
we want these edits to be concise, but in my experience building with AI,
we can do something like, hey, I want to make the app responsive
and mobile-friendly, and I want to fix
some of this other stuff, and then we can hit enter
and run that with Agent. Again,
we're going to kick off an Agent run. I think you can expect this one
to go a little bit faster, given that they're smaller edits. I'm going to talk a little bit
about what we're doing here. And then we'll let this finish up
and jump back once it's done. Again, building with AI, vibe coding,
it's like asking a junior developer or someone who's never seen your project
before to, make changes. You have to be really descriptive
about what you want. Make this fully responsive,
and mobile-friendly. Fix some of the alignment and padding issues,
specifically this one that I'm seeing. And if you can do that,
Agent has all the tools to access the files
in the context of your workspace to understand what the issues
are, and fix it. So you can see, hey, I'm looking at the home, component. I'm editing the URL form in, the client, and I'm going to start
updating the results overview to fix the SEO display
and make it more responsive. So just like that, we're targeting
some of these edits. And we can kind of hope. Right, that this is going to carry through
and do what we want it to do. Another important thing to mention is like
when we ask Agent to do these things, it has all the context of these files
and these directories. And so it's going to be able to search for
files is going to be able to understand exactly what's going on. A final point, these
and you can actually see in real time some of this stuff
being updated, which is cool. What we're doing
right now is we're in a chat with Agent. And so when I talked about context,
I talked about talking to AI and telling it
all of these different things. Everything in this chat is in the Agent's
context window. It has the history of all these things
that are done. As this starts to get longer, you might imagine AI being slower
or being less responsive. And so it's important when we're working
on different things, to create new chats, which will essentially clear the context or target the responses of Agent. And that's the same thing
is true of assistant. Assistant works in a very similar way. So what we're doing
is we're creating this initial chat. We got to a prototype,
and then we're iterating on a little bit. From there, we're going to switch to
maybe a new chat for our next feature. But that's all I wanted to talk about. We're going to let this run
and we'll jump back in. Once we have our results. So, it looks like we're done with that run
before we dig in, I just want to call out
that you might see this V2 early access a little pill if you're taking this course
relatively soon. We're
working on our new revision of Agent. By the time you're taking it,
hopefully this is already out to everyone. You might not see this. And you can just assume that you have
the latest and greatest version of Agent, and then everything's in a function
the same way. Now, what we're seeing here is a very
similar interface to what we just did. And that's it. Agent made improvements are kind of
returned with a report of what it did. Made some checkpoints along the way. So notice how these checkpoints are free. You should be seeing the same thing
on your side. And fix the SEO score display, improve
the mobile view with responsive layouts, added shadow effects, and rounded corners
to enhance the visual hierarchy. But generally don't ask for that. But you can know
that actually looks pretty nice. I like that a lot. And so, you know,
as we think to frameworks and we think through ways of designing
beautiful, engaging, fun applications, we can take that as a note of,
hey, like this looks really cool. And, made all UI components
adapt to different screen sizes. So what do we do
now? Well, we're going to test it out. So let's do the same thing
and enter our website. Okay. First thing, this is good. It didn't add Https to the
beginning of this and it wants a website. So follow up right away. Make it so I don't have to type https every time. That's a really great follow-up. We'll send that in a second as a part
of our next prompt. For the minute, for now, I'm going to just type in my URL,
see what comes through. This is better aligned. I think it could still be centered. Also, I got a nice little toast there
that popped up in the bottom right. These pills are actually now aligned and centered,
so it's cool that we fixed that. I like really the way
most of this looks and it looks like everything's coming through. So I check the social media previews. These let's see these look better. What happens if I resize the screen here. So, if I resize this
like these are the response. The widths are coming through
a bit better. If you recall these icons
were a bit kind of crunched. If we go back wide here, we'll just do
like iPhone 14 Max like a go old school. We're on the 60 now,
but this looks better, right? Like we have a more responsive layout. This is a bit more friendly
if I'm doing this on my iPhone, and, I'm starting to get an application that looks better
designed and mobile-friendly. So this is really cool. We're building tools. We're building tools that anyone can use, that
we can use to analyze our own websites and that work, which is really fun. So we're
going to call this V1 of our application. And what we're going to do in the next
lesson, is add some polish to this. We're going to go through, we're
going to make some edits in Assistant. And we're going to really,
take our apps to the next level, maybe add some advanced functionality
and then deploy. So I'll catch you in the next lesson.


Now we'll enhance our search engine
optimization or SEO analyzer prototype. We'll use Agent to add core functionality, then switch to Assistant for customization
and feature expansion. Finally, we'll deploy our application
so others can access it online. Now let's get back to it. So, in this lesson
we're going to enhance the functionality of our application and start adding
a lot of polish to exactly what we built. So, I'm going to go ahead to the top
and create a new chat with Agent to clear out the context
like we discussed. And start off by saying,
that we want to make a more visually appealing
and visually representative app. We have all this data about our website,
about the tags that we built. But the only real visual summary that
we're getting is an overall SEO score. So I'm wondering if we could get something
that more, better captures
the entirety of these tags. Maybe something that tells us
if our images, a summary of our images, a summary of our titles and descriptions, as well as this
kind of high-level information. So what we're going to do is we're going
to consolidate that into a prompt and ask Agent, we're also going to ask again
for a little bit more polish on the UI. So what I'm gonna say here
is make the application more visual. So I'm starting high level. And then I'm going to drill down
with just some quick bullet points for Agent that are all kind
of feeding into the same functionality. Again, we're keeping these very targeted. We're not getting too crazy with what
we're doing in one prompt. So, create summaries for each, category of meta tag that you will display visually to the user, similar to the overall, overall all score. If you remember from our wireframe, right,
I had in mind this, concept of multiple scores that we could inspect
each kind of category in our website. And it's okay that, you know, our MVP,
what we're building here, it doesn't look exactly like my wireframe. The wireframe is really more just to guide what we're building
and keep that in the front of my mind, but I am going to drill down here and try to tune those things
in a little bit more. Make the app overall more visual, and allow me to get a summary of the SEO results at a glance. So we'll add one more point,
which is exactly that. And, you know, a lot of times
I see when people are building with AI, they'll come to me and say, hey, Matt, I'm
trying to get AI to do this thing because XYZ and I want ABC outcome, right? And I'm like, just tell the model that. So exactly what I just told you,
we're going to tell to Agent. Help, overall, let's make the app more, visual and user-friendly for folks that might be new to SEO. So we can get a high-level view of how well our page is implemented. Now it's kind of like,
you know, giving wishes to a genie. You know, if you're not very explicit,
the genie might just do some crazy stuff. Right. So what we have to do here,
if we're asking for simplicity, is follow up that we want all the
functionality that our app has right now. We don't want you to take anything away. So that's a very important thing
that I've noticed does wonders. Do not remove any functionality just to make,
it easier to see a high-level summary or drill down into the details. And,
I think we kind of know the drill by now. So we're just gonna let this one run. We'll come back once
Agent is done implementing these features, and we'll talk about what it did. Okay. So we're through that round of edits. Let's see what we've gotten here. We're just going to go back
with our old example. We're going to analyze that. And now it looks like we get a bit more
of a visual representation of our checks. There appears to be,
let's see what's going on here. Much better overview. So we're generating kind of key findings
and what's good meta keywords. It's good. These are present, what needs attention
as well as some priority recommendations. So optimizing the title tag
and improving the description. We also get some category breakdowns. So SEO and social media optimization. So it looks like
there are some checks going on there. It's checking both for titles and the way that things
are going to show up on social media. We're checking how robots index
our sites, what's known as a robots.txt. Typically determines how websites
can be scraped on social media. We also have open graph tags and Twitter
card checks. Now, we can still drill down
into technical details and actually get a Google search preview,
which is nice. So you can see like this
is maybe a little too simple for Google though, as our social media previews
and this is a bit more clean. Nicer display
both for Facebook and Twitter. And then if we want to drop down into our SEO recommendations,
we get a much more descriptive view here. We can still see our raw tags. There's also some like nice hover effects. So, you know this is like 3 or 4 prompts. But what I want to point out is that, more than anything,
we had attention to detail and we had attention
to exactly what we wanted changed. And so we drilled into our app. There's still some things
that aren't perfect, right? Like this is not, ideal. The spacing here, we could really kind of like nerd out with getting this
to look the way we want it to. But overall, we have something
that's fundamentally more polished. Now, there were still a couple points. So we're going to open up Assistant here. To make some like finer
grain changes to what we just built. And when you open up Assistant
you might notice that your app restarts. You might notice that, there's a refresh. And that's because we're kind of like
living Agent mode, so to speak. And if you'll remember,
I didn't really like that I couldn't type in that I had to type in Https. So I'm going to say to Assistant because
it actually has a lot of the same context. Right? Into our application
that Agent has, I'm going to say, can you make the website entry form such that https is automatically populated and the user doesn't have to enter any other info? And because this is a quicker,
tool assistance, more lightweight tool, you're going to notice
that the responses are much quicker. And so Assistant
is going to read the relevant files. So this is our first kind of intro
to Assistant. It's behaving a lot like Agent
but in a similar manner. It's going to make the change and then
it's going to make a check checkpoint. So we made a checkpoint Https is prefixed theoretically,
we'll do DeepLearning.AI this time. Deep learning, we just type in DeepLearning.AI. We get our result. So what you might notice
is that our, our sort of, globe element is on top of the Https.
To provide context for Assistant. I'm going to actually open
up, our screenshot tool here. This is just using Mac OS. So if you're on windows
it might be different. I'm going to copy this to my clipboard. And then I'm going to paste
that screenshot in and say, now the globe icon overlaps with Https. Can you fix it? When we talked about context right? We talked about providing
additional details to AI. I giving it the information
it needs to fix errors. And that's what we're doing here. Hey, now
Assistant can see the overlapping problem. And it's going to present us
with a solution. And ask if we want to apply those changes. Looks like
we're getting a bit of an error here. Let's preview and see what's happening. It's actually just removing
it looks like removing the globe here. But the commit didn't quite work. That's okay. We're actually you know, sometimes stuff like that happens,
you might get similar errors. I'm just gonna literally paste
this in. Let me copy this run on it again. Let's see what happens. There. So it looks like it just took another
command really quick to run through. And actually looks like
the original changes were applied. We just might have gotten
a, a bit of a rerun getting a little error
there, it's all good. And when Assistant makes a new checkpoint
it's going to restart our app. So now we have our app. We have the Https. I can just type in DeepLearning.AI. I can click analyze. But just like that
we flipped over to Assistant. We made some more lightweight details. This is kind of like what we wanted
to build for our application. So we're going to walk
through the deploy pattern. This is something we'll do
for both of our applications. Note that deployments are limited
to Replit core users. So you might not see this experience. But at any time, if you want to subscribe
or upgrade, you can deploy that way. Clicking deploy, we're actually just going
to configure the build for you. You don't really need
to worry about these details or really even what Autoscale means. We're going to approve
and configure the settings, and then we're going to come up
with a name for our app. So we call an SEO tag inspector. Note that we're configuring the build
command and the run command. I don't really need to worry about any of this. Agent is going to take care of it
for me. All right. Click deployment. And you know it's important to mention
what's going on. This typically takes 2 or 3 minutes. We're taking this application. We're bundling it up. We're taking this entire environment
that we've built with Agent. And then we're just putting that
on the cloud basically, You don't really have to worry about the
technical complexity of what's going on. You don't have to worry
about any of the nuance or details. What you do have to know
is that everything that you just built will look the same on the web page
that we're deploying now. If you went back
and made more changes this application, they will not automatically flow through. You'd have to go through
and click redeploy. So we're going a little bit slower,
in this lesson. But that's because we want to cover all
the basics and understand how things work. So when we build our application,
when we click deploy, we're taking a snapshot of everything
that we just built with Agent, with Assistant, all these features that we added
and then promoting them to their own web page,
which we just defined. If at any time you want to learn more
about the different types of deployments that we have
or how deployments work or any of the technical details,
you can learn more in our documentation, or check out some videos on YouTube
that I built that have some really in-depth content. But at a high level, we're going to select
the right deployment type for you. And we can be very confident
that that's the right type to deploy. So in the meantime, you can watch this cool
loading screen. All deployments also come with logs. So you can see exactly
what's happening in your deployment. They come with analytics. So you'll be able to see who's visiting
your deployment, how many users you have, as well as some other settings
that you can drill into. Those will be visible
once our deployment is done. In the meantime, we're going to sit back. I'll join you once this,
this is done deploying. So, we're back. You can see the deployment
took about two minutes, which is great. And we're going to give you the domain
you just deploy to. So if I go to this URL we're going to get the same app. This is now deployed on the internet. Note it's at its own URL. You can visit this site if you want. I actually just entered a site and it had Https and the form
automatically removed that for me. So that's really nice.
I can click analyze. It's going to work exactly
the same as what we had before. So, these are the basics. We just built end to end an application
with Agent with Assistant that scrapes web pages, performs an analysis on that web
page to tell us if there's anything that we can improve
or maybe optimize for SEO Google Results, or social posts
and then deployed it. We built a tool completely that
other people can use that you can use, and that's how simple it is. And so, this lesson was really more about
polishing, flipping over to Assistant. In the next lesson,
we'll be able to speed through a bit more of the basics,
since we've already covered those and get into more of a complex topic,
so I'm really excited for that one. I'll catch you in the next lesson.



In this lesson, we'll build
a national parks ranking app with voting capabilities. Just like in the last lesson,
we'll develop a PRD and wireframe to outline our requirements. Crafting effective prompt, and use Agent
to create our initial application with our sample park data. So now let's get
started on our next project. And it's going to be a bit
more of a complex project using persistent storage and some,
more complete interactive functionality. So what we're going to build
is a head-to-head ranking app that uses a custom data set to store votes
and rankings in a database. So the idea is we're going to pull
in some national parks data for the United States,
and you'll be able to go to our app, and vote between two parks
that are presented to you, the user. Once you create a vote, we will, in real time, sort of tally up a ranking
for that park and return the relevant ranked parks. So we're going to go through
a similar process. We'll start with a PRD. We'll start with a whiteboard. And then we're going to discuss some more advanced building techniques
once we get into Replit for our app. So again starting with our whiteboard here
we're building our head to head voting app. And I want to go ahead and sketch
this one out as well. Again, I'm thinking we have two parks
that we're comparing head to head. Once we vote on this park we're going to
calculate a dynamic ranking. Now the parks are ranked according
to what's known as a ELO system. So the goal here is to have our rankings
displayed in real time, to store both our parks and our rankings
in a database, to use a persistent storage method like a database and display
overall rankings and recent votes. This brings me to an important point,
which is that by default, if you're creating apps on another platform or on Replit,
that data has to be stored somewhere so that we can access it and we can have
it available to us at any time. And so what we're going to do is initialize this application
and then move that data to persistent storage to a database
such that we can access it. We can update those values
and have it stored in our application. And that's a really cool feature
about Replit. We have databases built
in. We'll walk through that. So key skills for this one
prompting persistent data deployments debugging and handling
errors and edge cases. So let's look at our prompt. Help me build an interactive app
for voting and ranking the best national parks. The app should allow users
to vote on parks head to head, then calculate a ranking for the parks
based on the Chess ELO system. The app prominently displayed the matchup along
with overall rankings and recent votes. Again, this is really straightforward. We're emphasizing
the ease of our frank work and ElO system. And you might be saying,
hey, what about the data is that I'm just going to know
about all these national parks? Well, probably not. And even if it does, it's
not gonna have images for those parks. It's going to be hard to find the,
the information. So what we're gonna do from this app is we're actually going to pull data
in from an external source. Let me show you how to do this in Replit. It'll be really straightforward. But first, let's just take our prompt
to the Replit homepage. So just like the last lesson, we're
getting started here with our prompt. We've entered everything in. And now what we're going to do is enhance
this prompt with some additional context, like we talked about in the introduction
to this course. So it looks like
the US has 63 national parks. And this is the Wikipedia page
for national parks. And it actually we actually just list
all of the national parks here. So this is really cool. Something we haven't talked about is that Agent can actually just scrape
this web page for us. And if we're thinking about the context
that we provide to LLMs, well, if we say build this ranker
with all these national parks, and then we just give AI a list of all the national parks and potentially
the image URLs for those parks, we can be pretty comfortable
that it'll be able to use that data to integrate it into our application. So what we're just going to do is copy
the URL and bring that over to Agent. And so what we're going to do we're gonna
hit shift enter to create a new line. And we're going to paste
in a Wikipedia link. What you're going to see up top
is this little modal here. It's going to prompt us if we want to take a screenshot
of that page or get the text content. Obviously a screenshot of a Wikipedia
page, probably not what we want so we can fetch the text content. Our tools are just going to link to scrape that web page
and return us the text content. You can actually see it in real time
so you'll know exactly what it completes. It typically takes 1 or 2 seconds here. And once you see something populated
you can see the content. So now we know, hey, we're asking
AI to build this interactive voting app. Theoretically
it should have all of the content that it needs, all of the data,
which is the list of parks, the 63 parks. It'll also have access to the URL. So we'll see if Agent can implement
those properly. That might take some additional prompting. And the next piece we can actually use
that PRD a little bit more directly. We can use that wireframe that we drew
a little bit more directly in this app than we did in the last one. So we're
going to head back over to, our wireframe. And so now what we're going to do for
this one is we're going to actually just take
a take a screenshot of this wireframe. Because I like the design here. I want AI to also have some context
into what we're doing. And we're going to paste this into Agent. So now I've pasted the wireframe
and our prompt is just a little bit more complete. We not only have, really descriptive
but concise, prompt that tells Agent exactly what we want, but we also have
the text content of the Wikipedia page that explains what national parks are
or what parks exists in the United States and has a complete list of the parks,
along with really relevant data. And we've attached our screenshot. So let's see what Agent
does with this information. Just like the last
time Agent is going to come up with a plan and present us info for what
it's going to build before we jump in. Since we've been through this process
since, you know, really the first lesson was getting comfortable
with what we're building. We're going to confirm the plan here,
and then I'll catch back up with you. Once Agent is done with the initial MVP
build. One thing important thing to note. These additional features
here are just that, additional features. So regardless Agent's going to work
towards the initial prototype. We're actually going to work towards
implementing things like a Postgres database or maybe some of these
other functionalities. So you don't have to worry about that. You can just approve the plan and start. Also note that rate results may vary, so if you're following this tutorial
AI does different things. Sometimes your preview
might look different, the final result might look different, but
we're going to work through it together. So I'll catch you
once this preview is done. Okay, so we're back. And if you're following along,
will the app was being built,
you might have seen some errors come up. And we still have some errors,
so that's okay. We're going to we're going to work
through these one at a time. And you probably saw actually that Agent
fix some errors. And that was because Agent has the ability to both read the output
of what's coming from the application. So what's being written to the console. If we open a new tab and type in console,
you can actually see what's happening
as the application is working. And it also has the ability to take
screenshots and fix things dynamically. So we have our interface. There's a lot of stuff going on here. There's not a lot of things
that are working, but that's okay. We expected to get errors. On the whole, this looks nice. The images aren't coming through. I don't think we have all of the parts
here. I don't think it's scraped the data
like we asked it to. And we have some, like nonfunctional
account buttons up top, which is fine. But this looks like this
looks like our wireframe. So I think that's good.
And we're going to try and fix this. And we're not going to worry too much
about the broken images or the some of the other stuff
that's going on here. We're just going to focus on
seeing if we can make this work. But let's check
actually if the voting works. So I click vote and actually in these ones
the images do work. So the Everglades got a vote. It is tallied in my recent votes
which is cool. It doesn't look like it flowed through to
my rankings, so that's something to note. The rankings don't appear to be working,
but the recent votes and the images in some instances
do seem to work. This is where it might be helpful
to dig in to the code and kind of see what's going on. So I'm going to poke around, in
the components, see what's in hooks, lib. Okay. So park data, it looks like, you know, it just hardcoded
a bunch of parks in here. These do not appear
to be all of the parks. And it assigned an icon type
for these parks, which is probably
what's flowing through the front end. And then Replit has access to Unsplash. So it looks like
it just pulled these images from Unsplash. And then my guess would be for some parks
that didn't have images. So I think what our goal here
is going to be to see if we can get it to use Wikipedia
like we originally asked for. It's possible we just gave it a little bit
too much information. This is building with AI, right? We're going to take a look at this
as we talked about. What we're doing
now is we're testing the application seeing what works, seeing
what doesn't work. What doesn't work is that our rankings aren't pulling
from our persistent data storage. We are recording recent votes,
which is cool. But we also didn't really pull in parts
the way we want it to. So what I'm gonna tell Agent,
we're going to try something different. We're going to say the parks
data are listed on the Wikipedia page. Inside a table in the HTML. Please fetch the page. Download it and extract all the parks from the source. There should be 63, if I remember correctly. Each park has an image in the table. You should use the externally hosted image as the park image in our app. And then I'm replacing the Wikipedia page.
This time, we're not actually not going to get
the text content. The goal here is to see if maybe through an alternative prompting method,
we can get Agent to just integrate this directly. And I'm going to run that. So the goal here is I'm thinking
about the biggest problems with our app. The first one is that like
we just don't have accurate data. So there are other ways
to go about this, right? We could process the data ourselves. We could upload it. But I just prompted Agent, you can see what it did was it ran
a curl command on that Wikipedia page. And curl
is just a fancy way of getting HTML. And now it's using a couple libraries, beautifulsoup
and requests and writing a Python script to extract the part data
from the Wikipedia page. So this is pretty complicated, but I think it's important to
to walk through because we're kind of there, these tools that, the
AI can do for us, that Agent can do for us, and sometimes it works
and sometimes it doesn't. But this would save us time,
and it would let us build something really cool, given that we have
all the resources that we need. It's kind of like what we built
previously in the SEO, analyzer. We were able to scrape
and access that data. So it looks like it
just ran the script to extract the parks, but now it's trying out
a different approach. So it actually looks like
Agent is writing some scripts here to analyze the structure of that HTML
that it just downloaded. And hopefully if this is running,
we can kind of see the scripts that it's reading and writing in the in the file structure here. So we have analyze parks
updating these scripts to actually kind of introspect
that website that it downloaded. And I guess our goal or
my expectation would be that we'd come out with something that has all of our park
data implemented. Pretty complete. I'm actually pretty impressed. The Agent was able to do this. So,
theoretically it understands the structure of what we just extracted,
and now it's executing that Python script and then going to get the contents there. Okay. So now I want to talk. What just happened. Because there was a lot of stuff
going on in that run. What Agent did
was it wrote Python scripts both to fetch and analyze in the parks. And if you've been paying attention here, if you've been writing programing
or if you've been, you know, doing more technical work,
you might have noticed this is a JavaScript
project, is a TypeScript project. But we also now have Python files. And you might be saying, well,
how do we set that up? And that is because Replit as a platform
allows you to install languages in just seconds. And so Agent actually configured
this entire environment to run Python, ran these files to analyze the Wikipedia entry, pulled down
the Wikipedia page, wrote the data to parks data JSON, verified
there were 63 entries in the list, and we can look through this list
and be sure that this is the same. It's actually using the same images. These images are much better
than the ones we had before, and if we want to double check,
we can flip on over to Wikipedia. So now I'm in Wikipedia and I'm going to search for Virgin Islands. And we can verify that this is in fact the same image that we had previously. So what do we just do? We were able to extract this data without actually providing
or cleaning any data ourselves. Agent just did all of that for us. This is really cool. And we're making some good progress because my guess would be voting
probably still works exactly the same. We get new matchups,
we can skip the match-up, we get our recent votes, the rankings probably don't work. They definitely don't work. But again,
this is what we talked about before. We're adding context. We're fetching data. We're doing things
one step at a time. And so now
we have a description of our park. We have some data we pulled in
and we can provide those votes. Sign in, sign up. Not functional. Votes are not functional. We're going to walk through this one
step at a time. So let's say the rankings currently aren't working. The recent votes flow through. But I don't see any updates for these scores. And again we're going to trust that Agent is going to be able
to analyze this and make the fix. So we have a storage implementation. Agents
can open that up and take a look at it. Now you might be saying Matt
why don't we just start with a database. You talked about implementing a database. We know we want a database eventually. In my experience, it actually works better to make sure
that the data is being fetched properly and then make sure that the app
actually works with in-memory storage. So storing in memory works
for this application. But it means that like every time it's
restarted, we probably lose our rankings. So making sure that the app works first,
and then asking Agent to migrate that data over to a database.
And that's what we're doing. Other things to note. We can also kind of rely on Agent
to kind of solve these problems. So what do we do before? Well we just specified
where the image lived. We gave it a URL. We specified how we wanted the problem
to be approached. I want you to go to this URL.
There's a table. You have to analyze the table, extract
the data. Agent was able to do that and actually
take care of that entire thing for us. So again, as these models get better,
we can think about the smallest sort of building blocks
that AI is able to solve. Maybe like a year ago, you know, six months ago it was writing
a function or tab autocomplete or like doing something
that wasn't even that impressive. Now we can solve much more higher level
problems. Hey, we don't have a data source. Hey, we need to fix a ranking system. Hey, we need to update how these things are flowing through. And so now if you recall, we just voted
for the Virgin Islands, right? So, Virgin Islands beat Hawaii volcanoes. And our ranking now reflects,
a 1516 score. We can also check
let's make sure all of our parts are being pulled through here. That we do
in fact, have 63 national parks. So we have all of those. And again, so we have Death Valley
up against Virgin Islands. The way that an ELO system works
is that if the score is higher, for the competitor
and a lower rank competitor beats that competitor,
it should be higher in the rankings. So if we vote for Death Valley, you can
now see Death Valley is number one, and it has a 1517 score
rather than a 1516. It looks like let's check rank 30. I have no idea how to pronounce this,
so please don't hold me to that. We're going to check and make sure
that this actually is rank 30. Go in the page. It is. We can confirm that works again. What are we doing?
We're just testing our application. And that's that loop I was talking about
in the very first lesson. So what can we do now? Well, we have our voting system.
We have our rankings. They're updating in real time
and we have our recent votes. This is all in memory though. So what we want to do
is add this to a database. Now if you've built with other tools,
you know that adding databases are pretty complicated. One of the really great features
about Agent, about vibe coding with Replit Agent is that we can say, okay, let's now make our storage persistent, store all data in a database so it persists across sessions and users. Really? That's it. We can kind of
just trust that if we send this prompt, that Agent is going to be able to, number one, understand what we would like
and have the access to tools. You can see there
we just created a database. We created a server database file. And now it's going to take that in-memory storage and it's
going to migrate it over to our database. Now, while Agent is moving
through our sort of database steps, we're going to open up a new tab, just because I want to show you
what's going on. And we have a Postgres database. If you haven't used the database before
or you don't even really know what a database
is, it's kind of just like a CSV that relates to other sort of tables. So we're going to have a table structure. And then what we're going to see is that Agent is going to start
populating this database with data. You can actually see on the left
exactly what it's doing. It's going to start running some scripts. We have included a database viewer where you can kind of interact
with all of the tables. We'll come back to this once
there are tables. But what I want to point out
is fundamentally what's going on here. It's important
to understand how our databases work. So then if we go to our tools
and we go to secrets, we have some really great partners. We work with Neon for databases. And so what you can think about
is we're spinning up this like Postgres database on the back
end through our partners at Neon. And all we're doing is dropping in these
connection secrets into our secrets pane. So our secrets pane. If you've ever built an application before
this is like an environment. These are environment variables. We can pull in secrets from our account or just have these secrets
that live in our app. Let's see what's going on over here. Looks like Agent is still working
through the database connection. It's debugging and fixing some type errors and then executing some database migration scripts. Again don't worry about this too much,
but if you're interested in this and you're trying to learn a bit more about how databases work
with full stack web applications, hey, you know, like understanding
that it's using Postgres to understand what Postgres
is, understanding how the connection works and the packages work. And these are all really great ways
to learn more about databases, what database
migrations are, and how applications work. And this is how I've learned. So that's
why I'm going through this. Again, Agent's going to take a minute here. We're just going to let it work. And then we'll come back
once we have some data in our database. Okay. We're back. And we tried the database implementation. We're going to walk through
what happened because Agent went completely off the rails. For about 15, 15 minutes, but it's okay. So what happened? It did a lot of stuff,
and it set up our Postgres database, created a schema,
used consistent naming. It says it loaded Parks into the database. But like, I'm here, I'm refreshing. Okay, well, we have match ups,
we have parks users and votes. But it then kind of started working on an API server integration with database storage that's separate
from our actual integration. So let's let's see
what's going on in the web view. If we run the application, fails to run. So we're getting errors. So it looks like it did most of this. Something went wrong. This is where we're going to use what we
talked about before which are rollbacks. And you know the last change we made
was up here. You can see
we had a checkpoint about 15 minutes ago. And this is backed by git. So if you really wanted to see
what was going on here, with our checkpoints,
if you go to the git pane, we'll actually get a summary of everything
Agent has ever done. This is really useful. And you can actually create
a GitHub repository from here. Push it to your GitHub account
if that's what you're into. But the cool thing about checkpoints,
we hit a really like we just hit a roadblock.
I'm gonna roll this back. So what you're going to see is
that, Agent is rolling back. It completed that rollback. Now, if I run our application, we are back
precisely to where we were before. Agent is going to ask us. Hey, like,
what should I do differently? I'll tell you what you should do
differently in a second. I'm gonna refresh this
and see what's going on. It initialized our app. So now we're back to that state. Stuff breaks. This is kind of what I was talking about
before, right? It's okay. That stuff breaks. The cool thing is that we're rolling it
back and we're going to try again. So we're going to think through
what we did and we're going to try
something differently. So I think I was pretty basic. I'm going to create a new chat and we're
going to be really descriptive this time. We're going to say: our app currently uses parks data hardcoded in parks data dot Json. We'd like to move this to a Postgres database. Again, Agents going to know
how have the tools to access that. But I think what we need to do
is just be very explicit about the type of data we're using
and how we want that data inserted. You should analyze the structure of the data and create a schema for rapid, rapid import. Be sure to check the data types and perform all necessary migrations. So what are we doing here? We're trying again. Oh. We hit. We ran into a roadblock. We reset the app. We're trying one more time with new data. What you probably notice in the app
is that we lost all our scores and votes. That's expected because we don't have
persistent data just yet. So, we kind of took another approach. We added a little bit more description
to our prompts. We started a new chat session just
to make sure that the context was clear. Since, you know,
we've been doing a lot in this application and now we're running the app
to rerun that transformation. So we're gonna let Agent Roll here for,
hopefully a short amount of time. We'll come back, we'll see what happened. And, we'll keep building. All right. Amazing. So it looks like our updated prompt
got us the result we wanted. So Agent went through,
pulled that data, created our database, and then performed a migration,
importing a data from the JSON file
that we had into the database. So now we're getting the parks
and we go to our database tab should be able to refresh this. We see votes which is empty. We see parks
which has all of the parts that we wanted. Now in a form of persistent storage. We have a users table as well,
which could be useful in the future. We have matchups. So it's recording every matchup
park A ID, park B ID, which I would assume corresponds to Carlsbad Caverns and Bryce Canyon, both in Utah, if I'm not mistaken. But if we check those out, 11 Carlsbad
Caverns might not actually be in Utah. And Bryce Canyon is eight,
so looks like we have that. We have this really cool, database explorer here where you can kind of poke
around and look at the data, see what we're working with. So very cool stuff. We have some other interesting rows, but this is kind of what we're going for. So let's see if it works. We're going to vote for Bryce. Recording vote. Vote recorded. We get the matchup to change and Bryce
Canyon gets a better ranking. What we would expect
to see in our database right, is a vote in the votes table. So we have eight versus 11. We have the winner/loser
and the updates populating through. And then we also have a new matchup
in the matchups table. So that was just a couple prompts. We migrated the data
over into persistent storage. So now, all right
we have our ranking for Bryce Canyon. If I stop this app well we won't stop it. But if I refresh it or if I restart it
there it of course will all persist. We have persistent data. We've done a really good job. This is, you know, something that has been
historically really hard to do, integrated a database and really Agent
did most of the heavy lifting. So we're going to spin it down
for this lesson here. We're going to pick it up. We're going to learn a bit more
about our app. Maybe implement
one more feature, deploy it, and have a real functioning
as an NPS rank application deployed in just a couple minutes. So stick around. We'll wrap this one up. And you guys are going to be expert
vibe coders by the end of the next lesson.



In this lesson,
we'll refine our National Parks voting app by integrating our complete data set. We'll also add final features with Agent and Assistant
to ensure everything works correctly. Then we'll deploy the application
for public use. In the last lesson, we built out our park
ranking head-to-head voting app. Here we got all the functionality
implemented. We have our parks
being stored in a database. We kind of got this nice little ranking table here, along with our recent votes
that are being stored in Postgres. Now we're going to add
some additional functionality. We're going to understand how this works
and then deploy our app. But on that last stage
and running the dial I kind of zoned out. And so I kind of missed I missed
exactly how things were being implemented. So what I'm gonna do now
is I'm gonna head to Assistant, and I'm just going to ask it to help me
understand how my project's working. Again, you're going to see the
the preview refresh as we switch to Assistant mode. But I see client, I see server and shared. But like,
I don't know much about databases or how we implemented our database. So I'm just going to ask Assistant: help me understand what frameworks we used for our database. How does it work? And how are we managing it? And again
I think frameworks are important. Something I've been emphasizing
throughout this course. And understanding
what we're building is important because as we get into these debugging issues, you need to understand
how to solve these problems and how things work in order
to solve those problems. So, the system basically just gave me
a recap of everything Agent just built. We're using Postgres,
which is a main database configured through a connection URL
in our environment, variables, which we talked about. Cool. We're using an ORM framework
called Drizzle ORM, which is a TypeScript First ORM. I have no idea what that means. We're going to dig into that in a second. We're also doing some schema management,
so we're defining that. I can actually look through here
shared schema. Okay cool. Like this is defining, the parks table
and all of the columns there. It's defining a votes table. Neat. Actually walking through our tables. And the code base has two storage
implementations an in-memory implementation. So this was like
our old dev implementation. We could clean that up if we wanted to. So good call out
and explains how these things work. Now what was that?
They just walk this through. I don't really know what an ORM
framework is. So I'm going to say
what is an ORM framework and why would I use drizzle ORM. This is exactly how I've learned to code
or how to build with AI and basically everything
that I'm showing you today. I just asked a bunch of questions and, you
know, Assistant uses the latest models. We're just asking questions,
and understanding kind of what's going on. So this isn't going
to actually tell us here exactly what our ORM is. Object relational mapping is a framework that lets you interact with your database
using programing language objects. Instead of writing real SQL queries. Okay. So we're using objects instead of writing
SQL or SQL, which is a language
for interacting with databases. So it's actually telling us
some of the benefits type safety schema management,
query building, schema validation. Okay, maybe I don't know exactly what these things are,
but I could keep digging into them. So really the goal here is just to help
you understand, hey, I don't actually know what this thing is. I don't really know, what's going on, but I can learn
and I can ask AI and I can use that as a way to kind of reinforce the things
that I'm building and to keep, iterate it. So just like that, since then, it helped
us understand how our application worked and maybe even learn a few things
about Drizzle, about Postgres, and some of the tools
that we're building with. But I've been really impressed.
I like how this works. We have an application
where we can vote for the parks that are recorded in our database,
and we get our ranking system. This is exactly what I wanted,
and it's really cool. It's really interactive. We actually pulled the data
and these images from Wikipedia. So pretty complicated app here. Some data manipulation visualization kind of extract transform load type stuff. So now we're going to go ahead. We're going to deploy this
just like last time, we're gonna use Autoscale. We're going to approve those settings. And this time you'll notice
that we have secrets for our database. Right. So if we're going to deploy with APIs or external services,
we'd pull those secrets through as well. We'll call this one Park Challenge. I think that's available. And we're going to go down and deploy it. And just like last time, Replit is going to build this
deployment, it's going to configure everything and promote
that to, our production environment. There's really not much we need to worry
about since we built the entire thing out. And typically
in development, it's really hard not only to recreate your environment
in the cloud, but then to add
in those external services, to add in databases and manage
those databases in a separate environment, or do a bunch of other stuff. Right. And so what we have here is a database
that's directly integrated into the environment. As we talked about before, our app
kind of installed a bunch of stuff for us that ran some Python scripts
to import data. You know, Agent configured this database
that embedded into our application. And we didn't have to go anywhere. We didn't have to go to another tab. We didn't have to set up another account
or input an API key. We just did it all from writing Replit. So the power of building the Agent,
the power of building with Assistant, is that we're able
to make these applications, the full stack applications that have databases
that have front ends and have back ends and really, build complex tooling from one pane that is vibe coding, right? We did a lot of planning. We did a lot of discussion
about tools and frameworks, but for the most part, this looked a lot
different from, coding, or building otherwise. There's a lot of vibes. We're just kind of going
by feel with the debugging. Were going by feel
with like how we managed context. So I'm gonna let this deployment
in wrap up. We'll come back, we'll finish it
up, make sure our application works and we'll move on from there. Cool. So we deployed our application,
we get a URL, and it's deployed as an auto-scale deployment. Again,
if you want to learn more about that, you can check out our videos
or the documentation. Opening the app. We get exactly what we had before
I can vote on Mammoth Cave. And actually, it's important to note that the same score
is pulled through from before. That's
because we're using the same database. So we have our voting system,
we have some recent votes, and we have our rankings that are
being stored persistently in our database. As a kind of talked about,
this is the power of Replit. We just deployed this live on the internet
for anybody to access. If you want to share with your family
or friends, you can do just that. And yeah, you know, end-to-end full-stack
application, persistent storage,
vibe coded all the way. So in our next lesson
we're going to talk about next steps. We're going to talk about where to go
from here. But thanks for joining along. Thanks for building some cool apps. Let's let's wrap it up.



And that's it. Congratulations on completing the course. In this lesson,
we'll recap what we've built, discuss best practices, and explore
next steps for continuing your journey in Vibe Coding. So we've just built some pretty amazing applications
without writing any code at all. And the amazing thing, and what we did
was that we built two production ready apps, not just toy demos
that were full stack with persistent storage and deployed five
to their own URL for anyone to access. Again, we wrote little or really no code
and it was all logic. It was all the concepts that we discussed
at the beginning of this course. So what were those concepts? Well, it has to do with thinking. It had to do with frameworks,
and debugging ,and checkpoints, and context. So let's do a quick review
of everything we learned before we move on to some next steps in your vibe
coding journey. So in our very first lesson
we talked about five skills. Those were thinking framework,
checkpoints, debugging and context. In thinking we discussed
a logical hierarchy for thought. And that goes from everything
from logical thought. Maybe asking
what is the game that I'm playing? If you're playing a game of chess
to procedural thought, how do I excel at that game
and how do I implement the functionality or instruct
a computer to excel at that game? And that's kind of what we did
with our apps, right? We asked ourselves, what is SEO? What is the features? What are the features
we're trying to implement? Understand. What is our natural, our natural park system
kind of rank are trying to do? What are we trying to implement? And that started with frameworks. It started with understanding what we didn't know
and trying to figure out more about the things we wanted
to, and understanding what frameworks allowed us to do that thing and work
best with LLMs. In some cases, that meant asking questions to our tools,
to agent and assistant to learn. So even when we were building, stuff broke
and we kind of knew that going in. But we used checkpoints, we used versioning to minimize
the impact of what we were building. And a really great thing is that
Agent and Assistant came with those checkpoints, came
with that version control out of the box. And so what that meant is that we could
chunk up our builds into MVP's minimum viable products and features
and move really quickly. Now debugging.
There was a good bit of debugging. I think we had fun,
but we were methodical. We were thorough. We understood how our apps worked,
sometimes asking Assistant and we got to the root of the
issues and fix our problems. Finally, context. Obviously, context is important. It's been something I've emphasized
throughout the course, but we got that context by providing images,
providing links to what we were building and in one case, providing actual data
to a web page because we provided context, because we gave a thorough explanation
to Agent of what we were trying to do. It's actually able to extract that data
and implement it into our application. When Agent made a mistake
with implementing our database, we provided additional context. That is, additional details
about our app in order to get around it. So remember always thinking
about the context that you're supplying or not supplying to your LLM. And finally, we used our framework
for iterative building iterative vibe coding if you want to call it that,
for creating features with AI testing, those features, getting at an error
or maybe not an error at all, debugging that error to get to a checkpoint
and then moving on to the next feature. And along the way,
we built two applications that worked. But if you follow this pattern,
you'll be able to build much more advanced applications
and much cooler things. So what are some next steps? If you're new to vibe coding
or this is the start of your journey, I think the biggest step
is to just keep building. I've found that I learn best by doing
and I would encourage you to keep doing the same. And most importantly, keep having fun. I do this
by finding common problems in my life, or finding things that I want to try
to automate or recreate or improve, and then implementing the skills
that we just did together. So you can follow along on social,
you can connect with Replit, you can connect with me. I would love to see the things
that you build, and showcase them on our Replit accounts. And finally, if you're a Replit
core member, we do have Replit community. So once you join Core,
you'll be eligible to join our community where you can post and interact
with other members. But again, I'm Matt in developer relations
with Replit. This has been Vibe Coding 101 on Replit. Thanks for joining.


