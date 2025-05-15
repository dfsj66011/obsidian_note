
大家好，我是 Pieter Abbeel，非常高兴能和大家分享关于深度强化学习基础的系列讲座，这个领域让我感到无比兴奋。本视频是第一讲，我们将涵盖 MDPs 及其精确求解方法。当然，这是六讲系列课程中的一部分。

第二讲我们将探讨深度 Q 学习。第三讲介绍策略梯度和优势估计。第四讲涵盖 TRPO 和 PPO。第五讲讨论 DDPG 和 soft actor critic 算法。第六讲我们会以基于模型的强化学习作为收尾。我认为这六讲内容将帮助你们打下坚实基础，既能理解当今强化学习领域的所有研究工作，或许还能开展一些自己的研究。

那么我们就从这第一讲开始，首先我会为大家介绍整个系列的背景动机——为什么需要关注深度强化学习？深度强化学习让我如此兴奋的原因是什么？之后我们将探讨基本框架马尔可夫决策过程，接着研究几种精确求解 MDP 问题的方法。

你可能会想，既然我们要精确求解这些问题，那为什么这门系列课程在第一讲之后还有后续内容？事实上，精确方法虽然为后续学习奠定了良好基础，但它们并不适用于大规模问题。因此我们将看到，精确方法能很好地解决小规模马尔可夫决策过程，而对于大规模马尔可夫决策过程，我们将在后续课程中基于此处所学的基础知识，学习其他适用的方法。

在本讲座的最后，我们将实际探讨一种称为 *“最大熵框架”* 的表述。这一框架将为我们后续课程奠定更坚实的基础，因为最大熵正逐渐成为强化学习领域最受欢迎的框架之一。正如你将看到的，它能够为智能体提供基础探索和鲁棒性方面的帮助，而这些可能是更传统、标准化的框架所无法实现的。

首先来点动力吧，我认为至少对我来说，深度强化学习的真正兴奋点始于 2013 年。2013 年发生了一件大事——DeepMind 展示了通过深度强化学习，神经网络智能体能够学会玩多种 Atari 游戏。这是一项重大突破，因为在此之前，强化学习的成果通常仅限于相对较小、较简单的玩具级问题。

突然间，这里出现了能够理解屏幕内容的智能体——它们可以处理视觉输入，并且你可以在不同游戏上运行强化学习算法，让它学会玩各种游戏。这是个非常激动人心的成果。与此同时，实际上，我在伯克利的实验室也一直在研究深度强化学习在机器人领域的应用，最初主要集中在仿真机器人上。

大家现在看到的右侧是一段机器人学习成果的精彩视频集锦。这些都是通过强化学习实现的——游泳机器人学会了游泳，跳跃机器人学会了蹦跳，二维步行机器人学会了奔跑。再次强调，这些能力并非通过人工精心编程控制获得，而是完全由机器自主学习掌握的。

这是通过机器人进行试错学习、强化学习实现的。实际上，这个过程是这样的：在最初的迭代中，机器人表现不佳，但它们从这些经验中学习，随着时间的推移变得越来越好，最终熟练掌握你设定它们要掌握的技能。

从那以后，另一个重大成果是 2015 年 DeepMind 推出的 AlphaGo，它证明了在强化学习的帮助下，计算机有可能在围棋领域击败最优秀的人类选手——世界冠军。这是一个重大突破。许多人曾认为这还需要很多年才能实现。

就在那一刻，强化学习成为了核心。与此同时，我在伯克利的实验室里持续推进模拟机器人的学习能力——从之前视频中展示的二维机器人（它们只能在垂直或水平平面上运动）发展到如今完整的三维机器人学习奔跑。

那么接下来我们将看到的，当我播放这段视频时，就是观察这个机器人如何学习。我们将亲眼见证强化学习的实际应用。起初它只会摔倒。然后它坚持得稍久一些才倒下——这已经是进步。随着时间推移，在不断迭代的过程中，它通过试错学习、从自身经验中汲取教训，表现会变得越来越好。

妙处在于，同一段代码随后可在另一台机器人上重新运行。它可以学会操控——比方说一台四足机器人，也能在同一台机器人上执行新任务，比如学习起身动作。同样无需任何特定编程，你唯一需要编写的就是一个强化学习算法。

然后，强化学习算法本质上就是用来训练机器人并使其掌握这些技能的。事实上，这里使用的算法——TRPO（信赖域策略优化）加上广义优势估计——你们会在后续几节课中完全理解。这个算法同样可以运行在 Atari 游戏上，并学会玩这些游戏。

然后，强化学习算法本质上就是用来训练机器人并使其掌握这些技能的。事实上，这里使用的算法——TRPO（信任域策略优化）加上广义优势估计——你们在本系列课程后续几讲中会完全理解。这个算法同样可以运行在 Atari 游戏上，并学会玩这些游戏。

接着我们还研究了如何将这一技术应用到真实机器人上。现在大家看到的是Brett——伯克利大学研发的用于消除重复性任务的机器人。Brett能够学习将积木块放入对应的孔槽中。嗯，我看到这里正在快进演示...这个机器人通过不断试错练习将积木放入匹配孔槽的能力，经过持续训练后，Brett最终掌握了将积木精准插入对应孔槽的技能。

当时实际上是我实验室的博士后Sergey Levine（他现在已经是伯克利大学的教授了）与Brett共同领导了这个项目。Sergey后来在谷歌待了一年，得以扩大这项研究的规模，让一整组机器人运行类似的强化学习算法。这里的妙处在于，对机器人来说，数量越多学习速度反而越快。

为什么会这样？因为机器人越多，它们就能共享数据。所有机器人都在并行收集数据、学习并分享这些数据。他展示的结果表明，通过试错学习来掌握抓取物体的能力，虽然远非已解决的问题，但实际效果却出人意料地好。

这就是你在这里看到的实际场景。事实上，有大量机器人在一起训练。到了2017年，OpenAI证明在热门游戏《Dota 2》中，他们开发的AI在一对一比赛中击败了最强人类玩家。后来这个AI在五对五比赛中也表现优异。这表明通过深度强化学习，我们不仅能掌握简单的雅达利游戏，还能攻克复杂得多的电子游戏。

在此我要重点介绍我们在伯克利完成的更多工作。我们现已证明，这种模拟机器人能够掌握极其广泛的技能，几乎可以学会任何杂技动作。它们能通过自主试错进行学习。这种方法同样适用于非人形机器人——比如这只正在奔跑的模拟狮子，它的步态就是通过强化学习训练出的策略实现的。

当然，这真的很棒，因为现在有了这种能力，比如当你设计一款电子游戏或制作电影时，不必再逐帧手动调整狮子的运动轨迹，只需告诉它从A点移动到B点，它就能自行计算出如何从A跑到B。

2019年，Deepmind展示了AlphaStar——这是一款通过学习掌握《星际争霸》游戏的AI机器人。作为比雅达利游戏复杂得多的另一款极受欢迎的视频游戏，他们通过结合模仿学习与强化学习，成功证明了该AI能接近人类顶尖玩家的水平。

接下来的一大亮点，或许你们之前已经见过，但依然令人振奋——OpenAI展示了通过强化学习让机器人学会解魔方的可能性。现在大家看到的这段视频相当长，我会直接快进播放。

这只机械手经过大量强化学习后，已经掌握了破解魔方的技巧。我们现在看到的正是它的实战表现。短短几分钟内，它就能成功复原魔方。以上只是部分亮点展示，实际上该领域每天都有大量研究工作在持续推进。

但我想，这能让你很好地理解强化学习已开始实现的各种可能性——这确实非常令人振奋，因为不仅最终成果本身往往极具吸引力，更关键的是这些智能体和机器人是通过自身的试错学习来掌握这些技能的。

由此可见，它们在某种意义上具备极强的学习能力，这种能力未来可能延伸至掌握其他任务。这一切的核心在于强化学习，而强化学习所采用的框架正是马尔可夫决策过程。接下来让我们深入这一框架，为其建立形式化的理论结构，在此基础上方能构建真正实现强化学习的算法。

So our agent will be in an environment, gets to choose an action, after choosing that action, the environment will change. And then the agent gets to observe the changed environment gets to choose an action again, and this process repeats over and over and over. And with the current situation of the environment, there is a reward associated with that, scoring how good that situation is.
              
                  10:28
                  For example, maybe in a video game, the score of the video game could actually be the reward or the incremental score you achieved in the last step could be the reward. Um, if a robot's supposed to run, maybe the amount of forth progress made could be how the reward is measured and so forth. And the goal is for the agent to repeatedly interact with his environment, and over time, figure out the right action for each situation to maximize reward.
              
                  10:55
                  We're making an assumption here in this framework, that the agent gets to observe the state. There are extensions called POMDPs where the agent doesn't get to see the full state, but w we'll focus on the straight-up MDP setting at this time. So what does formally define an MDP? If you have a problem you want to solve, if you can map it to an MDP, it means you can run a reinforcement algorithm on it.
              
                  11:18
                  But what does it mean to map it to an MDP? Well, there's a set of states, then a set of actions the agent gets to choose from. There's a transition function that defines the probability of ending up in state S prime at the next time, given at the current time, the agent is in state s and took action a.
              
                  11:37
                  There's a reward function that assigns reward for that transition. When you were in a state s, took action a, landed in state s prime. There is a start state, or sometimes it starts state distribution if  it's not always in the same state things start. A discount factor gamma. And I'll say a bit more about that in a moment, but essentially it captures that things that are further in the future, we might care less about.
              
                  12:01
                  We care more about getting reward today than getting reward, let's say a year from now. And so the discounting says we should discount future rewards. One justification for that is, let's say maybe your reward is money. If you had money today, you could invest it. And it would become worth more if you invested it.
              
                  12:22
                  Uh, or if you did a very simple investment, put it in the bank, it would earn interest. And you can think of this discount factor gamma as discounting future money, counting it for less value, because if you only get it in the future, you can not earn the interest on it in the intervening time. And then there's a horizon H which says for how long we're going to be acting.
              
                  12:43
                  So if you can map your problem onto this framework, then actually you can feed it to a reinforcement algorithm that can try to solve it for you. The goal then of course for that reinforcement algorithm will be to find a policy to maximize the expected, discounted reward over time. So gamma to the power T says that things further in the future will be discounted more.
              
                  13:06
                  Because this count factor gamma will be something between zero and one, maybe 0.9 or 0.99. And so that signals that if you have 0.9, you roughly care about a horizon of 10. Uh, if you have a discount factor, 0.99 you roughly care about a horizon of a hundred. So it might depend on your problem setting, what you want to use.
              
                  13:28
                  We'll also later see that some algorithms might want to have gamma as a hyper parameter to help them function better. So what's an example of what fits this paradigm? Cleaning robots, let's say you have a cleaning robot, the state what's the state, it's where everything is in your house, as well as where the robot is.
              
                  13:48
                  And then the actions is, well, the robot might be able to move somewhere, um, or might be able to pick something up. The transition model describes if a robot takes a certain action, maybe move forward, what's the chance that actually moves forward, how far will it move forward? Or if it does a pickup, what will be the result and so forth? The reward function could be something, well, let's say it's a vacuum cleaning robot, it could be you get rewarded if dirt is removed from the floor.
              
                  14:18
                  Or maybe you get rewarded based on whether all the objects are in the location they're supposed to be, and everything's been organized. So you would get assigned a reward to making progress on those tasks or having fully achieved those tasks. Gamma is this discount factor, and maybe with a cleaning robot would care about a pretty long horizon, maybe  0.99 or 0.999.
              
                  14:41
                  And then the horizon H is how many steps we're going to be considering. For your cleaning robot in your house, you might want it to be working well for a several years. So it might have a horizon of several years of walking robots, maybe you're designing a walking robot and then the state could be the configuration of all the joints of the robot, but not just to the angle, the joint angles, but also the velocities because physical state is not just in pose but also in the derivative of pose.
              
                  15:09
                  And then maybe there could also be things about the environment in the state, like the configuration of the ground in front of the robot, if the robot gets to observe that. Actions could be the motor torques applied to each of the motors, a transition model captures the physical dynamics of this robot interacting with the world.
              
                  15:27
                  The reward function would depend on what you want as a designer. If you want the robot to maybe stand still, you would give a reward for, staying in place. If you want it to run you would reward it for making forward progress. Again, you might pick a discount factor gamma based on over what horizon, you kind of think that you can measure good behavior.
              
                  15:46
                  And say, if I see the robot active for five seconds, that's indicative of how well it's doing. Then maybe you have this kind of factor gamma that relates to that. Pole balancing. Uh, this is maybe not something you spend a lot of time thinking about in your daily life, unless you're an RL researcher, but can you bounce a pole maybe in the palm of your hand, or maybe that's sitting on a cart, which is a very common kind of task as a kind of basic test, a low dimensional problem, that's often looked at.
              
                  16:14
                  Games fall into this framework, maybe tetris, backgammon, atari games, go and so forth. Server management, very different from the examples we've seen so far. But let's say if a lot of requests are coming into your servers, you want to manage where each request go. Maybe you're running a shopping site or maybe you're running a server for compute jobs, and there's different researchers sending in compute jobs and these jobs have different properties.
              
                  16:39
                  And you want to see which job you want to prioritize onto which server and then maybe your reward function relates to the throughput of jobs. And if you assign jobs to the right service, maybe the throughput will be better than if you assign them to the wrong service. But your your reward function could also be something much harder to optimize for, but maybe you want to try such as, you know, the number of research breakthroughs coming out of your lab, when you're running a specific server management policy.
              
                  17:06
                  Shortest path problems are examples of Markov decision processes, maps onto this whole framework. Often MDPs are also used as models for animals or people. So you might say, oh, maybe I'm building a self-driving car and now there's other people driving. And I want to model them and maybe I'll model them as behaving, according to trying to optimize their own policy within their own MDP.
              
                  17:28
                  And that could be an interesting way to capture what's happening around you. In fact, there's some intuition there, right? Cause probably the other drivers are optimizing something they're probably optimizing to get to their destination, to be safe, maybe some other things that they might care about, like cell phone coverage, or a good view where they're driving.
              
                  17:48
                  So our MDPs are defined by the list here on the left. And we've seen quite a few examples and you can probably come up with more examples of your own. And in some sense, if we have a working reinforcement learning algorithm implementation, that's kind of all you need to do. You just need to map your problem onto an MDP, and then you can run the RL algorithm, and hopefully it'll solve your problem.
              
                  18:14
                  So in practice, unless you're doing, research and development on the algorithm themselves, being able to define your problem as an MDP is really key. And then you run something. But in this course, of course, I want to introduce you to all the specifics and foundations of reinforcement and algorithms.
              
                  18:32
                  So we're going to not just define problems this way. We're going to now start understanding what they are and how we can start solving them. So we'll have a very canonical example throughout this lecture, which is grid world. Grid world is not something you probably interact much with in real world, but it's a really simple environment that allows us to build intuition about MDPs and reinforcement learning algorithms.
              
                  18:56
                  And also when you start running experiments of your own, it allows it to run experiments really, really fast because it's a very small problem formulation. So here's how our grid world, what do we have? We have an agent, and this agent is this little robot here, and the agent is currently in square one comma three, so first row, third column.
              
                  19:22
                  And from there, it can move to any of the neighboring squares. So the actions correspond to moving to a neighboring square. This here has a big rock blocking this square, so you can not pass through this one. And then the rewards here, there's a plus one, if you'd get through the diamond square on top, and there's a negative one, if by mistake you were to go into the fire pit over here.
              
                  19:44
                  And here's a start state on the bottom left. There is effectively 11 locations, the agent can be at. And clearly the goal is for the agent to move to that top right square and collect the plus one reward. Formally the goal is to maximize the expected, discounted sum of rewards accumulated over time. And we want to find a policy that doest that.
              
                  20:09
                  Now, the dynamics of this world can vary. Um, you could set this up  in a way where the actions are deterministically successful, or it could be that the actions are noisy, when the robot tries to move in a certain direction, there's a potential for an other consequence. Maybe 80% chance of success and 20% chance to randomly move off to the side instead of to the direction the robot want it to move.
              
                  20:33
                  And that will allow us to also reason about stochastic environments, even within this simple, simple grid world. The beauty of course, with this grid world is that you can pretty much eyeball the optimal solutions. And we'll do some of that very soon, which then in turn could give intuition, when you run an algorithm: is it doing it right, does it match with our intuition of what should be happening? So the goal is to end up with a policy that tells you what to do.
              
                  20:57
                  In this case, the policy is shown the bottom here might be a pretty good policy to maximize expected reward. Note that there's one subtle thing here with the discount factor. What it does is, by having this discount factor that makes future rewards less valuable, you get the incentive to follow the shortest path to the square with the reward.
              
                  21:15
                  Because if you don't follow the shortest path, you're wasting time, which means your future reward will be discounted more heavily and you won't get as much. All right. So we've seen the framework of MDPs. Let's look at a first solution method, value iteration, which will be the foundation for many things we'll see in the future.
              
                  21:35
                  So what's the concept of a value function? A value function V*(s) is the maximum we can achieve in expected discounted sum of rewards. So when we use the best possible policy that maximize this, how much discounted sum of rewards are we're going to get? To be able to think about this value function, we have to think about what is the optimal strategy and what will it get us.
              
                  21:59
                  So let's go to our grid world example. So agent is in our gridworld. It can move in any of four directions. Of course, not down where it is right now, because that's up against the boundary. And then if it lands in the minus one square, it gets minus one and everything's over. It lands in the plus one, it gets a plus one reward everything's over.
              
                  22:20
                  So let's now think about this. Assuming everything's deterministically successful. Our discount factor, gamma is one, so no discounting is happening effectively. Our horizon is a hundred, so we have a hundred steps for our agent. The agent starts in the (4,3) square, which is up here. Well then it'll right away collect that reward of one and be done.
              
                  22:39
                  So that's easy. That's the best it can do. Then, what if it starts in (3,3) over here. The best it can do is move to the right, collect that reward, and then it's done. And since there's no discounting and only takes one step, so it's within the horizon of a hundred easily achievable. No discounting, so that reward of one really is worth one.
              
                  23:00
                  And we also have a value of one here. How about (2,3) over here. Well, it'll take two steps to get to the plus one reward, but there's no discounting. So it'll still be a value of one. How about starting at (1,1)? Well, it'll take 1, 2, 3, 4, 5 steps to then collect that reward, but there's no discounting so still optimal value is one.
              
                  23:21
                  How about if it starts in (4,2), well, over here, it's trapped in the firepit and it'll get the reward of negative one. And so the optimal value here is not great, but that's just what it is when you were there the helpful value happens to be negative one because you have no options in life. When you're here, you're just in the fire pit getting a negative one.
              
                  23:40
                  Okay. Let's make this a little more complicated. What if we have a discount factor gamma of 0.9. And actions are still the domestically successful. And let's say we start over here (4,3). Well, what happens? We grab the diamond once we're there. That's our action. We get a plus one reward and it's done.
              
                  24:00
                  That happens right away. So there's no discounting yet. We get a value of one. What if we now start in the (3,3) square next to it. We first have to step towards the square with the diamond. And then the next action is to get the plus one reward. So it's one delayed. So we have a discount of 0.9 and that's the best we can do.
              
                  24:21
                  So the value is 0.9. How about starting in (2,3)? Now it takes two steps. So we're discounting twice. 0.9 times 0.9 times the 1 is 0.81. What if we start in the (1,1) square? We have 1, 2, 3, 4, 5 steps before we can grab the diamond to get the reward. Discounting 0.9 to the power five. And so the optimal value starting in (1,1) is 0.59.
              
                  24:52
                  And then how about if we start in (4,2) the fire pit again? Well, that's not where we want to start obviously. There's nothing we can do there except for essentially the agent uh, is stuck in the fire, it gets a reward of negative one. Now let's make it even more complicated. Now the action success probability goes down to 0.8.
              
                  25:13
                  So when the agent, for example, where it is right now wants to go up there is a 80% chance it will land in the square above it, but a 10% chance it'll veer  off to the right, 10% chance it'll veer off to the left. So the dynamics is more complex now. Discount factor still 0.9 and horizon 100. So plenty of time.
              
                  25:35
                  What if we start square with the diamond. Here there is no navigation actions, just the grabbing action with actually always succeeds. So we just get the reward of one right away. What if we start next to it? Well, if we start next to it, we want to move towards the diamond and then grab it. And so that would be, we'll get to diamond one step later, 0.
              
                  25:55
                  9 times one is what would have. That only happens if our action moving to the right was successful. And that happens with 80% chance. So with 80% chance, so it's 0.8 times discount 0.9, we get then the value of being in the square (4,3), which is one. But we also have a 0.
              
                  26:21
                  1 chance when we tried to move to the right that we actually ended up moving down. Or move up, but up bounces against the wall stuff to stay in place. So is there a 0.1 chance of Len staying in place in the (3,3). So we have 0.1 chance of staying in place where we would get the value of square (3,3), but it's discounted by 0.
              
                  26:43
                  9 because the valid square (3,3) we'll only start getting one step later. And then same if we were to move down the value of square (3,2), and so something interesting is happening here to know the value of the square (3,3), we actually get something recursive in terms of the values of the neighboring squares. So you'll again, have to compute it in terms of values of the neighboring squares that you might transition into.
              
                  27:07
                  This is a general principle here that we're going to explore, that's called value iteration. So simplest is the initialization of our value function. We're going to have now value functions, not just V star, but V star indexed also by how many time steps are still in the future. Let's say there are zero time steps in the future for us, there's nothing left.
              
                  27:29
                  Well, then the value is  zero, obviously because there's nothing left. And so what can initialize the values for all states as zero for when there is zero time steps left to go. Then when there is one time step left to go. That's the optimal value for the state, when we have one time step V one star S.
              
                  27:51
                  Well, how do we get that? Well, we look at all actions available in that state and then some over all future states, the probability of landing in that future state S prime given we're currently in state S took action a, then we multiply with the reward we get from that transition, plus a discount factor the value we'll be getting from then onwards.
              
                  28:16
                  Then onwards has zero time steps left. So V zero star S, which is actually going to be a zero. This is really the key idea behind valley iteration is that you decompose the problem of having a certain number of time steps left into the immediate thing and one less time step left. Same thing happens for two times steps left.
              
                  28:38
                  What is the optimal value for state S when horizon is two? Well, we'll look at all possible actions available to us. For each action we check what is the expected reward we would get right away, so average reward averaged by the probability of going to that state S prime and also averaged what we'll get in the future weighted by the transition dynamics.
              
                  29:01
                  And in this case, what we'll get in the future, of course, it's again discounted and it's with one time step left. Then same for k  time steps, you have a general iteration here. It says, what is the value for being a state S with K time steps left. well,  the best action we can take at that time as measured by the distribution over possible states we weigh by the probability of landing in a future state s', the reward from the transition, as well as the value we would get in that next state discounted by gamma, with K minus one steps left.
              
                  29:36
                  Thinking back to the previous slide here, that iteration is effectively, what we were writing down here, we're writing down how the valid state three, three is what you get immediately, in this case immediately the reward is zero and then the distribution of what you get also in the future, depending on which state you land in.
              
                  29:57
                  Okay. So now, let's take a look at the full algorithm. We start with setting the value function, which is just some kind of array or vector,  equal to zero for when there is zero time steps left for all states. Then we're going to iterate. If we have a total horizon of H then we're going to have to iterate H times.
              
                  30:20
                  For each horizon, we essentially work our way from zero time steps left, one time step left, and so forth. We're going to have to look at all states and for every state we're going to have to compute this maximum, the exact equation we've been seen in seeing on the previous slide. Find the best action by looking for each action: what is the expected immediate reward plus expected future value from the state you landed in.
              
                  30:46
                  And we might as well also catalog what the action is that is prescribed by all this. And then the result is that as we run these value updates or Bellman updates or a Bellman backups we get the optimal value for every state, for anywhere from zero through H time steps to go. Let's take a look at this in action.
              
                  31:09
                  Okay. Zero iterations in. Well, all the values are zero. Now, we'll iterate through all states to find V one of s. What do you think we're going to see? Well, when there's one time step to go, we know for S at the diamond state, we'll be able to grab that reward of one. If we're in the fire-pit state, we will end up having to eat the fire effectively and get a reward of negative one.
              
                  31:33
                  And so when K goes to one, we expect a one on the top, right. Then a minus one below it. And how about all the other states? They'll remain zero because within one time step, there is no reward to be accumulated. So we applied this equation to all those states and here as a result. Then we can go again.
              
                  31:55
                  What if we are curious about V2? We have k=1  shown below. V2 will do one iteration on top of that. Well, what do we expect? Well, the square neighboring to the plus one square will become non-zero because we'll be able to transition into that. And then at the next time, get the value of that top right square.
              
                  32:15
                  How about next to the fire pit? Will that go to non-zero? Well, the optimal action would be not to move into the fire pit. And so the optimal action will actually be able to keep this value at zero by explicitly avoiding the fire pit. So this will actually stay  zero. Indeed, what we see here. How do we get to 0.
              
                  32:37
                  72 while we had an 80% chance of successfully moving to the right, if we move to the right, we would get a value of one, but it's discounted by 0.9. So we have 0.9 value by getting there times that 0.8 probability of success of moving there. And so as we move through these iterations, what we see is this kind of fanning out of the value from where the good values are to states that can reach these good states with reward.
              
                  33:02
                  And so as this iterates over time, we see that all these states start getting non-zero values,  reflecting that now in nine times steps, you can actually get to the plus one reward with a pretty high chance and get the corresponding discounted value. And as we keep iterating this, we'll see that at some point, these things stop changing.
              
                  33:24
                  So let's see, going back here at 10, 11, 12, not much change to a hundred, no change from 10, at least with two digits behind the point. And so when we see her as a convergence behavior, and this is actually quite typical for value iteration. If you run it for long enough, it'll converge and you get what is a stationary, optimal policy, as well as the effectively infinite horizon value is shown to you.
              
                  33:51
                  Even though he didn't run it for infinitely many iterations, it converges to that, or very close to that much sooner. And ,the speed of convergence often relates to the discount factor. The closer the discount factor is to zero, the faster things will converge. The closer it is to one, the longer it might take for things to converge.
              
                  34:10
                  Okay. So there's a theory behind that. Value iteration converges. At convergence, we have found the optimal value function V star for the discounted infinite horizon problem, which satisfies the Bellman equation. So at convergence, we have the solution to this equation. Now, so we know how to act for infinite horizon with discounted rewards, which is nice.
              
                  34:33
                  You just run valid trust and feel convergence, and this produces V star. And then once we have the star, we can extract the optimal action using the Bellmanequation yet one more time. Or we might have just stored the optimal action from our Bellman updates during running the algorithm. Note that the infinite horizon policy is stationary, that is the optimal action at the state S as the same action at all times.
              
                  34:55
                  So it's very efficient to store. We don't need to store an action for each state for each time. It's just sufficient to store an action for each state. Which is really convenient and that will be our policy that prescribes for each state, the optimal action PI star. Okay, what's the intuition behind this convergence? Why do we know this is going to converge not just in this example in the grid wall, but more generally? Well, V*(s) is the expected sum of rewards accumulated starting from state s, if you
              
                  35:29
                  act optimally for infinitely many steps. Okay. V*_H(s) is the expected sum of rewards accumulated starting  from state S acting optimally for H steps. The additional reward that's collected over time steps, H plus one, H plus two, and so forth. We can write out what that is. It's gamma to the power H plus one, because it's discounted what you get at time H plus one, times the reward at time H plus one.
              
                  35:58
                  Then gamma to the power H plus two times the reward at time H+2 for being in state s H plus two and so forth. This is a geometric series. We can bound this from above. And so this is smaller than gamma to the power H plus one divided by one minus gamma times the maximum reward you can possibly get anywhere.
              
                  36:19
                  Well, look at this. This quantity R max over one minus gamma is fixed, will not change as we change our horizon H. But as we increase our horizon, that is the number of iterations we run our value iteration algorithm, this gamma to the H plus one at the top, gamma is below one, so that will keep shrinking.
              
                  36:42
                  And so the difference between optimal value V star for infinite horizon and optimal value V star H for finite horizon H is bounded, as H becomes larger by a smaller and smaller and smaller number. And that's why this is going to converge. As age becomes large enough, the optimal value become very, very close to V star of s.
              
                  37:07
                  So this goes to zero as goes to infinity, and that's exactly what we just talked about. For simplicity of notation, I just assume that all the rewards are positive here. If some rewards could be negative, then you kind of have to redo the same derivation where you use the absolute value of rewards rather than just rewards.
              
                  37:29
                  And it's the maximum absolute value of reward that will be in this equation rather than just the maximum reward. All right. Now there's another way that people have shown value iteration converges, uh, through contractions. And this is a mechanism that sometimes it can be helpful in proving other things.
              
                  37:50
                  So let me give you just the main intuition here. The idea here is that we first gonna define a norm. In this case, the max norm. So the max norm of a vector is the maximum absolute value among all entries in that vector. An update operation is a gamma contraction in the max norm, if only if we have two vectors, so UI and VI, and then we apply an update to both of them becoming UI plus one and VI plus one, that those updated vectors UI plus one and VI plus one are closer together by a factor gamma than the original factors UI and VI.
              
                  38:25
                  There's a theorem that says a contraction converges to a unique fixed point, no matter the initialization. Why would that be the case? Intuitively essentially, no matter where you start, no matter which two vectors you look at, that you could have started from, they get pulled closer together, over time.
              
                  38:42
                  And if they always get pulled closer and closer together, and that's true for all pairs, they must end up all in the same point. Okay. Fact: the value iteration update is a gamma contraction in the max norm meaning that when we do a Bellman update and we have two different starting points for our update two different value functions, we start from, we do a one-step update.
              
                  39:04
                  They will be brought closer together, which means then because it's a contraction. Uh, if we do it as long enough, it'll actually converge no matter where we started from to the optimal value function. So converges to unique fixed point. And there is an additional fact, as you go along and can actually even bound the error that is still there by looking at the size of the update.
              
                  39:25
                  If your update is small, it actually means that also in the future only small changes can happen because it tends to be changes that propagate throughout the space. And so you can have a bound on your error, even for finite number of iterations. So once the update is small, we're close to convergence.
              
                  39:43
                  And so sometimes you can run it that way. Instead of saying, I'm going to run this many iterations, you can say I'm going to run it until the update is small enough, which means I'm close enough to the optimum and I'll abort my value iteration. All right. So that was some theory, now let's take a look at what's some of the environment parameters do to the optimal policy.
              
                  40:03
                  So here we again have a grid world. Agent starts in yellow square and it can again, move up down left, right. We have a bunch of fire pits here, bad fire pits, rewards of negative 10 if you land in there. Then there's a reward of one over here, which is somewhat close by. And a reward of plus 10 over there.
              
                  40:23
                  And keep in mind, once you land in a reward square, the way the environment works is at that point, the episode is over. So if you go to the one, you'll get the one and it's over. You can not later also go to the 10. So whichever non-zero award square you are on, that's where you end up and it's over once you've collected that reward.
              
                  40:43
                  So now let's think about it. What if we want the agent to prefer to close exit the plus one while being okay with risking the cliff. So maybe along the bottom, go to the plus one. What choice do you think we need to make for gamma to discount factor and for the noise on the dynamics? Four different choices here.
              
                  41:07
                  What if we want the agent to prefer to close exit, the plus one, so close again, but avoid the cliff. So go the roundabout way to the plus one. When would that be optimal? What kind of discount factor would favor that? And what kind of noise would favor that? What if we preferred the distant exit, the plus 10 while risking the cliff.
              
                  41:26
                  So we want to go along the bottom, keep going along the bottom all the way till reach the plus 10. And what if we want to go to the distant exit, but take no risk, go along the top. So there's a little exercise for you to think about. So there are four different agent behaviors that we might want to get out.
              
                  41:43
                  And what I'm telling you is that actually in this grid world, in this MDP, by choosing the discount factor gamma in a specific way, and choosing the noise on the dynamics the probability of success of the action, and when it's not successful, it'll veer off to the left or the right from the direction it wanted to move.
              
                  42:01
                  By choosing that noise factor and the discount factor in a specific way, different behaviors will be optimal. And I've given you four choices here for discount factor and noise. I've given you four scenarios here for what I would like to get out as optimal policies. How do they match up? And just to be clear, they don't match up line by line.
              
                  42:23
                  The exercise is to figure out for a, is it 1, 2, 3, or 4 that will result in a? For b, is it 1, 2, 3 or 4 that will result in b. Why don't you give it some thought. Okay. So if we want the agent to prefer the close exit plus one while risking the cliff. So going along the bottom, we need the discount factor, gamma of 0.1 and noise zero.
              
                  42:59
                  Noise zero, that should be easy to understand because at noise zero means that actually, even though we say risking the cliff, there was no risk of being errantly going into the cliff because zero chance of your action not being successful. So you can safely navigate close to the cliff because there was no noise.
              
                  43:17
                  And then how do we ensure the agent prefers to exit with a one instead of keep going to go to the 10? Well, we need to make sure we discount enough. The 10 takes two extra steps to get to. So we need to make sure the discount factor is such that two extra steps makes it so much discounted it's not worth that much anymore.
              
                  43:39
                  With a discount factor of 0.1, those two extra steps will cost us 0.1 to the power two. So 0.01 multiplied into 10. That 10 will only be worth 0.1, if it comes two steps later. And so we prefer a one now over a 10, two times steps later. What if we want it to prefer to close exit still while avoiding the cliff? Well, if we introduce noise, well then the agent, if it goes along the bottom, it would likely drop into the fire pits, and get very negative rewards.
              
                  44:13
                  So the optimal policy would be to go around, be far away from the fire pits. And then since our discount factor is still 0.1, since it would take two extra steps to get to the 10 compared to the one, the discounting of that two extra steps is 0.
              
                  44:33
                  1 to the power to, it's not a good choice to keep going for this two extra steps to get the 10. You want to just get to the one where you can get to sooner. If we preferred the distant exit which is a plus 10 while risking the cliff. Well, if we make things deterministic again, no noise, it's not really a big risk to run along the cliff because you're not going to fall into it.
              
                  44:53
                  And the discount factor 0.99 means that the discounting of the two extra steps, 0.99 to the power two, which has almost no discounting for the two extra steps. And so we're definitely going to favor taking two extra steps to get to the 10 over two steps sooner getting the one. What if we want the distant exit and avoid the cliff? Well, then we introduce noise again.
              
                  45:15
                  When there is noise and our agent is not guaranteed to be successful in its actions, well it shouldn't be close to fire pit,  it's dangerous. It will want to run around along the top. It is willing to take the extra two steps to get to the far away 10, because 0.99 is very little discounting. So it's worth the extra two steps to get to 10.
              
                  45:34
                  All right, now we've seen values of states. Turns out there's another concept, which is Q values, and there'll be important too. They are very related. Q-Star for a state as and action a is the  expected discounted sum of rewards, if you start in state s, take action a, and thereafter act optimally.
              
                  45:55
                  So it's essentially, it's like a value, but not for a state, but for being in a state and having committed to a specific action in that state. We'll call those Q values. Okay. We're gonna actually have a Bellman equation for Q values also. And some of the algorithms that we'll later see, we'll use this Q value Bellman equation.
              
                  46:16
                  What is the optimal Q value for being in state s, action a? What is the optimal value? Well, what happens after we committed to that action, we'll have a transition to the next state s'. So we need to say, okay, what's the distribution over next states s'? Well, it's the reward we get for that transition, just here.
              
                  46:35
                  So the first term here measures expected reward on the immediate transition. And then the second term here measures expected value at the future state s'. This here really is V*(s'), but now expressed with Q values. What is V*(s')? It's the best you can do from state s'. Well, that means we need to pick the optimal action, a' in that state s'.
              
                  47:05
                  So let's stare at this just a little longer, because this is one of the most important concepts: the Q value iteration equation. Q value is expected sum of rewards plus expected future Q value, which of course is discounted because it's one step in the future. So Q value iteration does the same thing as value iteration, but  with Q values.
              
                  47:26
                  You now have Q  subindexed by k and k+1, and exact same equation iterating through these Bellman updates to start from initialization of Q values, which could be all zeros for Q zero star. And then from there work our way up to Q H for whatever horizon we want. And of course, for the same reason value iteration converges with V values, it also converges with Q values.
              
                  47:51
                  And so we can get a optimal Q value for infinite horizon that we can then use for our agent. Okay. So what does it look like on this kind of environment after a hundred iterations? The important thing I want to point out here is that, well, in these navigation squares, we now see four values because each of the four actions has a different value.
              
                  48:13
                  For example, here, the square next to the diamond square, where we'll exit with a reward of one. When we take the action to the right, we have a high chance of success of landing into that square with a reward. And so there's a high value. And when we go up well, can't really move up, we stay in place.
              
                  48:27
                  We actually wasted a cycle. We moved to the left we wasted more cycles, we've got to work our way back. It moved down, we not only waste a cycle, but we also risk maybe with a noisy action ending up in the fire pit later. So that's even worse. So you can read of from this Q values, what is the right action to take.
              
                  48:44
                  And that's one of the nice things about Q values as you end up with your Q values, you can read off a optimal action in each state by seeing which one achieves the highest Q value. Okay. So at this point we have covered value iteration. In fact, two types of value iteration, based on V values and based on Q values.
              
                  49:06
                  Both will be important. And they already allow us to solve small mDPs. We can loop through all the states and actions repeatedly. We can solve them this way. So if you have a small MDP, you can just implement this value iteration thing. And there you are. You get optimal values for the infinite horizon problem,optimal policy, and you can then deploy that policy and have your agent do really well in your environment.
              
                  49:32
                  There's actually another method. And, since we already know how to solve small-scale MDPs with what we saw so far by iterating over all states and actions, you might wonder, well, why do we still need another method? It turns out that some of the approximate methods we'll see later. Some of them will build directly on the value iteration approach.
              
                  49:48
                  Others will build directly on the policy iteration approach. And so it's good to understand both because the methods that we'll see in the future, there's different methods, that are relevant for different kinds of situations. And so we'll want to have both foundations. Okay. How does policy iteration operate? A first step is policy evaluation.
              
                  50:08
                  In policy evaluation, what you do is you say I'm going to, well, I'll start from value iteration to ground this. So this is what we had in value iteration. In policy evaluation, we fix the policy. If we don't get to max over actions, we're going to fix the policy and then we can do something similar, the same equation, but there's no max anymore.
              
                  50:28
                  Once we fixed the policy, we can run value iteration for that fixed policy. And that's called policy evaluation. And that'll over time give us the values for different number of time steps left for that policy pi. So if somebody gives you a policy, you run this policy evaluation sequence of updates.
              
                  50:47
                  At convergence, we'll find the value for that specific policy for each state for infinite horizon. You might wonder, how do we know this is going to converge? Well, it's just special case of value iteration because you're running value iteration update equations. Just the max is very restricted.
              
                  51:06
                  You don't get to choose. But that doesn't fundamentally change things in terms of convergence. All right, so let's do another exercise. You have to familiarize ourselves a bit more with policy evaluation and let's now consider a stochastic policy. pi(a|s) is the probability of taking action a in state s.
              
                  51:29
                  Okay. Which of the following is the correct update to perform policy evaluation for this stochastic policy? Is it equation one, two or three? I'll give you a moment to think about it. Okay, let's take a look at this. So first equation we have a max over actions, but we know that in policy evaluation, the policy chooses the action.
              
                  52:09
                  So if we're giving the updates the freedom to choose the action, we might end up with actions that are not matched with the policy we're prescribed to evaluate. So if this is not going to work. Second one, value for going for K plus one steps. We land in state s' we sum over possible actions as described by the probability of taking that action here.
              
                  52:35
                  Probability of landing in the next state s', and times reward for that transition, plus, the value when we use that policy with k steps left in the next state s'. Well, this looks right to me because this is exactly what we have in our value iteration equations, except that the action cannot be chosen with a max it's done based on what the policy probabilistically prescribes, and there is a corresponding averaging happening here.
              
                  53:03
                  This is an equation 2 is the answer. This is how we compute the value of a  policy. How about equation three? Um, well, that's not right because you can not choose your next state s'. And so when we compute values here, it assumes that part of the process that we would get to choose our next state s', but that's not available.
              
                  53:25
                  We only get to act according to the policy, not choose next states directly. So again, three is wrong. Only two  is a correct answer. All right. So now we know how to do policy evaluation, both for deterministic policies and stochastic policies. Now we can build an algorithm around this. We can at a high level, have an algorithm where we have a policy we start with, we evaluate it.
              
                  53:53
                  And after we evaluated it, we'll use that evaluation to make an improvement to the policy. Then we'll evaluate the new policy, use that evaluation to make an improvement again. And that's called policy iteration. What does one of these iterations look like? So at first evaluate the current policy pi_k, and then we'll improve.
              
                  54:11
                  Evaluation we've already seen, essentially repeat those value iteration equations, but without a max, you have to follow the action the policy prescribes, can't take the max. But other than that, this is just value iteration type updates. You find the values of each state when using that policy, and then we can make an improvement.
              
                  54:29
                  Find the best action by looking one step ahead. Here's the idea. The new policy looks at what is essentially again, a value iteration update. Where again, we'll get to max over actions actually now, and we store whatever action is the best action we store in pi_{k+1}(s). Let's look at this a bit more closely.
              
                  54:52
                  What we're doing here, we're looking at the action in state s that maximizes, averaged over future states s', if we were to take that action a, the reward we'd get immediately and the discounted future value we'd get if we were to follow our current policy pi_k, which we just evaluated in the other step.
              
                  55:17
                  So, you get to choose your first step and after that, you're going to use the policy pi_k because we know the value of each state for policy pi_k so we can use that to evaluate the future. All right. And then we go back. We evaluate now this policy pi_{k+1}, improve upon that with a one-step look ahead approach and keep repeating.
              
                  55:38
                  And over time, this will actually converge. At convergence we'll actually find the optimal policy and actually often converges in less overall iterations than value iteration. But of course there's a trade-off because inside this big iteration here, there are smaller iterations happening here that are like value iteration.
              
                  55:56
                  And so even though the overall outer iterations might be less, there was more work on the inside. And so there are some trade-offs involved. Okay. No, I want to give you a little bit of extra intuition as to why we have these guarantees. So here, repeating the  policy iteration algorithm: is policy evaluation, followed by policy improvement, which is done by one step look ahead.
              
                  56:19
                  And I told you there's a theorem that guarantees this converges and at convergence you find the optimal policy and the optimal value function. What's the intuition for this? Here's a proof sketch. First of all, I'm going to tell you why it's guaranteed to converge. Then I'm going to tell you why it's optimal once it's converged.
              
                  56:36
                  So why is this guaranteed to converge? Well in every step the policy improves. This means that a given policy can be encountered at most once, because if you're improving and improving and improving, well, the next policy can not be a previous one because the next one is better. This means that after if it traded as many times as there are different policies, which you can bound in a finite environment, there's only so many policies we must be done and have converged.
              
                  57:11
                  Well, we now know that we're converging this hinges on one little thing, is that in every step the policy improves. What's the intuition behind that? When you do this one step look ahead, you're choosing an action, that is the best action. So you are choosing the best action right now, if you later applied the existing policy.
              
                  57:34
                  Well, by taking the best action now fall by layer the current policy. That's better than just applying the current policy right now fall about a current policy. So you aren't improving your action in that first step. Now the new policy that you're using will actually do this in every step. It will then the next step again, do the same thing.
              
                  57:57
                  It'll say in this new state I'm in, I'm going to look one step ahead. And by doing that every step you're actually doing even better than a one step improvement. And it's, it's not just that you have a one step in, but you actually have a beyond one step improvement under the hood when you get your new policy pi_{k+1}.
              
                  58:13
                  Now, why are we optimal at convergence? Well, by definition of convergence pi_{k+1} must be able to pi_k for all states. This means that when we compute pi_{k+1} is equal to pi_k, meaning that the argmax is equal to what pi_k already prescribes. But we know that's the value iteration equation. And in value iteration, when we have convergence we're at the optimum.
              
                  58:38
                  And so same thing here with policy iteration, we improve the policy. When we have convergence, we satisfy the Bellman equation, the value iteration equation, and we're at the optimal value function V star. Alright, that was a little bit of a theoretical aside. But hopefully it helps you build a bit more intuition for this algorithm.
              
                  58:58
                  So at this point we've covered two methods to solve MDPs: value iteratoin and policy duration. And these will be the foundation for many of the algorithms we'll see in the later lectures to solve larger scale problems. But before we go to that I want to introduce to you also another formulation for solving MDPs or at least framing MDPs, which is the maximum entropy formulation.
              
                  59:27
                  As you'll see many of the areas we'll see in the future actually will borrow ideas from this. So the question we're asking here is what if we could find a distribution over near optimal solutions? Instead of finding a single optimal policy, what if instead, we try to find a distribution over possible behaviors that are all pretty good or put higher probability on good behaviors, lower probability on not so good behaviors, can we do that? Well, we will momentarily see how to do that.
              
                  59:58
                  But let's first think about what would that give us if we have such a distribution. First of all, it gives us a more robust policy effectively because by having this distribution over possible behaviors, if then we deploy a policy and something changed in the world, we can rely on other things in the distribution.
              
                  1:00:16
                  Maybe the optimal policy, somehow the path is blocked, but because if it is a distribution over solutions, we can rely on one of the others and still do quite well. Another thing that could give us is more robust learning. And this is a little subtle for now, maybe, but at the high level, essentially for now, we've looked at solution methods where we directly solve MDPs.
              
                  1:00:40
                  And that's possibly in these smaller scale problems. But in the future larger scale problems we're going to solve, there'll be an iterative process where learning happens into leaving of data collection in the world with improving a policy or a value function, collecting more data, improving the policy value function and so forth.
              
                  1:00:57
                  And so as we go through that data collection process. Well, how do we collect data? Well a typical thing is to use current policy to collect it. If your policy is very deterministic, highly optimized, then the data collection might not be as interesting. And so by looking at the maximum entropy approach, we'll end up with policies that introduce more variation in how the data is collected, which can provide better exploration, which can lead us to find better optima in the long run as we're training this policy.
              
                  1:01:27
                  Okay. So what is entropy? Entropy is a measure of uncertainty over a random variable X, let's say. It's the number of bits required to encode it. If you go through information theory, it's a very precise measure of what it takes to encode random variable X on average. So mathematically the entropy of a random variable X with  a distribution P, is a sum over all values, x_i this random variable can take on, the probability it takes on that value.
              
                  1:01:55
                  So it's a weighted sum times a two log because we're going to be measuring in bits in this case. Though, it's just a scale factor, of course, if you use a natural log. And so intuitively what's going on here, but I don't want to dive into too much information theory. You can think of this as if I have a distribution over values my random variable can take on.
              
                  1:02:14
                  If I want to encode it, values that are very likely, I'm going to encode with a small number of bits. And values that are less likely I'm going to use more bits for. Well, why, why even use more bits? Because I, of course, I need to distinctly encode them. I can not use the same encoding for different values because then I cannot distinguish.
              
                  1:02:32
                  So I need, let's say if my variable x_i can take on 10 different values, I'll need 10 different bits sequences to be able to transmit that variable one different bits sequence for each value from one through 10, let's say. And so instead of uniformly assigning bits sequences, I might assign a very short bit sequence to a value that's very likely, and then a longer one to a value that's less likely.
              
                  1:02:55
                  And it turns out that the way this optimally can be done is that you end up with essentially a number of bits log_2 1/p(x_i)  for each value x_i. Okay. So let's look at some examples. If we have a binary random variable X that can take on two values, zero or one, let's say. And, we look at the probability of X being equal to one, which is a number between zero and one.
              
                  1:03:23
                  And we'll look at the entropy of the distribution. It peaks at a half, because that's where it is maximum uncertainty about what the value is going to be. Let's look at some other examples. Here are two histograms, which of these two histograms do you think as a higher entropy? Um, one on the left or the one on the right? Well, the one on the left has a higher entropy.
              
                  1:03:47
                  Why is that? Well, entropy measures the amount of uncertainty over what value to random variables going to take on. The one on the right, there's not much uncertainty. It's almost always going to be this first value. Whereas the one on the left, well, there's a lot of uncertainty about which one value it might take on.
              
                  1:04:03
                  More mathematically, you can compute this. The entropy for the distribution on the left, which is 0.25, 0.25, 0.25, 0.125, 0.125 is this equation over here, which is 2.25. Calculate for the one on the right: 1.3. So the one on the left has almost twice the entropy compared to the one on the right. So now that we understand a little bit about entropy, it's this thing that measures how much variance or uncertainty, there is in the outcome of a sample from a distribution.
              
                  1:04:39
                  We can now start bringing it into MDPs. So let's say we have a regular MDP, we try to maximize expected sum of rewards. What does a maximum entropy MDP do? Well, instead of maximizing just the sum of rewards, and by the way, you can also discount this, but I left the discounting out here for just simplifying the equations a bit, less symbols for us to stare at.
              
                  1:05:02
                  It adds an extra term. What is this extra term? It's looking at the entropy of the policy. So the policy describes the action to take in each state. But of course if we make the policy deterministic that entropy will be zero and that's a very low entropy as low as possible entropy. We're not going to want that.
              
                  1:05:22
                  So we now have a policy that describes a distribution over actions in each state. And what this says is that in maximum entropy MDPs we're not just maximizing reward we're also, with a trade-off factor beta, maximizing the entropy of the policy in each state encountered. Now there will be a trade-off, of course, because the more you can control your actions, the more precisely you can control what your agent achieves and the reward it's going to get.
              
                  1:05:53
                  And so the larger we make beta, the more we optimize for entropy, likely the less reward we're going to be collecting. And vice versa, the smaller we make beta, in the limit if we make beta equal to zero, we'll be able to collect more rewards because we don't have to worry about entropy. So at first you might say why even bother if this is going to be a trade-off on performance? But that kind of goes back to what I said earlier on when we pull that slide back up, even though it looks like a trade-off when you're actually training with some
              
                  1:06:24
                  of these more advanced algorithms, the learning will often be more robust if you have entropy in your objective. Because for a longer time to policy will stay more sarcastic and which will lead to exploratory data collection and better trial and error learning. Another reason you might care is that you could end up with a more robust policy.
              
                  1:06:41
                  If the environment changes you'll have other things to policy knows to do, rather than that one action for each state. Okay. So these are all good reasons to care about this. But now the question is, can we also solve this kind of MDP? To do this, we actually need to make a little side sidetrack. Um, instead of right away solving this MDP, the way we've done for regular MDPs with value iteration, we're going to do something called maxent value iteration.
              
                  1:07:13
                  But we need to first look at constrained optimization. So in constrained optimization, obviously, I mean, you can take an entire course in constrained optimization. I'm going to cover this here in literally a couple of minutes. So, if you haven't seen constrained optimization before, this might move a bit fast, but if you've seen it before, hopefully this is a good refresher.
              
                  1:07:32
                  If you've not seen it before, hopefully at least it gives you some of the intuition. In constrained optimization we're maximizing some objective subject to a constraint. So we have a function f(x) we maximize while g(x) has to be equal to zero, which limits our choices of x not every x is a valid choice.
              
                  1:07:48
                  We can only choose x where g(x) is equal to zero. And among those x where g(x) is equal to zero, we want the one that maximizes f(x). Okay. There's a concept called Lagrangian which is often used, where we maximize over x still. And then, instead of putting f(x) on the inside, on the inside, we put f(x) + lambda g(x).
              
                  1:08:09
                  So we're now gonna do a max min you might say, wow, we start with a max with a constraint. Now we have a max min. Turns out it'll help us. So we now solve a max min problem for f(x) + lambda g(x). And I'm not deriving this here, but there is a result in constrained optimization that says that at the optimum of this original problem, the derivative, or the gradient if it's a multidimensional x, of the Lagrangian script L with respect to x is equal to zero.
              
                  1:08:41
                  The gradient with respect to lambda, if there's multiple lambdas or derivative with respect to lambda if there's one lambda, is also equal to zero. Okay. So to solve the original problem, we formed this Lagrangian and then we can find a solution by setting these two derivatives equal to zero and solving for x and lambda that make this zero, it gives us a solution to the original problem.
              
                  1:09:04
                  It gives us the x we want. In addition also it gives us a lambda, in the process. Okay. Now I'm not going to step in great detail through the math here, but we can actually do this for max-ent. What I show on the slides here is for one step problem. In a one-step problem we maximize the expected reward in that one step.
              
                  1:09:25
                  And we use a policy which is a distribution over possible actions, plus beta times entropy. So we don't want to just pick the action that maximizes reward, because we also want high entropy. So our result here will be a distribution over actions. A policy pi(a) which assigns non-zero probability to all actions, a bit more probability to the optimal action, but also non zero probability to the others to have a good entropy.
              
                  1:09:48
                  So this is the problem here. Now, pi(a) is a vector with entries between zero and one for each possible action. So this is our objective here. Now pi(a) can not take on any values. The entries needs to sum to one because it's a probability distribution. So we form a Lagrangian. This is our Lagrangian. Set derivative with respect to our x variable, which is pi(a) and with respect to lambda equal to zero.
              
                  1:10:14
                  Work through some math. We'll get that our optimal max-ent policy pi(a) is equal to one over a normalization, constant Z. So let's forget about Z for now. It's equal to exp(1/beta r(a)). So let's also ignore  beta for now. Let's assume it's equal to one. Then we're saying the probability of taking an action is the exponentiated reward associated with that action.
              
                  1:10:38
                  So actions with high reward will have higher probability. Actions with low reward, or a negative reward will have lower probability. Now, of course, the probability has to sum to one. And that's why we have the one over Z over here, which is the normalization constant. So solution to this original problem, does kind of what we want.
              
                  1:11:00
                  This maxent formulation results in assigning higher probability to actions that have higher reward, lower probably actions with lower reward. And the extent to which we do this depends on beta. So if our variable beta, which is a trade off between entropy and reward, if you make beta very, very high, we really favor entropy.
              
                  1:11:22
                  Then this beta here will be very, very high and the larger we make beta, it'll shrink effectively the rewards that we have here, bringing them all closer to zero, make it more uniform, how much reward, we're exponentiating and as a consequence all the actions will have a similar probability. In the other limit, if beta goes to zero, we don't care about entropy.
              
                  1:11:46
                  Then this here will be dividing by a number close to zero, not exactly zero, but a very small number that will scale up their awards, push everything further apart. Higher rewards will be further away from medium rewards, further away from small rewards and we get heavy favoring of the high reward actions.
              
                  1:12:05
                  Now this is the optimal policy. What is the optimal value? Well, we can fill this back in, compute the optimal value. We see that the optimal value is the log, let's ignore beta for now so let's assume beta is one, the log of the sum over actions exponentiated reward of each action. So this is a soft max actually.
              
                  1:12:26
                  The value under this max-ent value iteration problem is the softmax of the values, rather than just picking the direct max. And how sharp we take the soft max depends on beta. And beta close to infinity favors entropy. It means we take a very soft, soft max and beta going closer to zero, makes this into a very sharp, soft max.
              
                  1:12:48
                  And so it is very intuitive in some way. When we think about value iteration, we take max. We think about max-ent value iteration, we'll be taking soft maxes. And how soft depends on the temperature parameter beta that we choose. And so max-ent value iteration. We had it for one step before now. Let's look at what we have for multi-step.
              
                  1:13:08
                  , We have this overall objective on the left one step update will look at this here, right. The immediate reward plus entropy, but then also we need to look at future awards. We have the decomposition here with V_{k-1}, 1 less step to go. This is the new Bellman equation, the max-ent Bellman equation right over here.
              
                  1:13:28
                  Um, well, let's see, reward plus then value that we can call Q. And then the other part is the entropy. And then if we look at this, well, we actually have the same problem as we had before. If we think about rewarding QS interchangeable, at least in notation here, then we know what the solution is to this problem.
              
                  1:13:49
                  Our optimal value will be a soft max of these Q values. And indeed, if you work through the math, you'll get that V_k will be a soft max of these Q values, sharpness of the soft max determined by the temperature parameter, beta. And then the policy will be similarly this softmax policy over the Q values.
              
                  1:14:12
                  And so what we see is we started from this idea that we want to introduce a distribution over possible solutions. And we saw it could bring an incentive for not just looking at the max by adding an entropy term in the objective. We worked through the math, we'll get this very intuitive solution that the policy becomes the softmax over Q values.
              
                  1:14:34
                  And as we do value iteration updates, the value for V_k is a softmax over the Q values. Q values defined above here are reward plus value with one less step to go. We now have also covered max-ent value iteration,    which intuitively lines up very well with the original value iteration, but of course introduces this stochasticity in the policies, which later we'll see, can help us in things like exploration.
              
                  1:15:03
                  All right. So that's it for lecture one, we looked at the motivation, why Deep RL is such an exciting space to be working in and to learn about. We looked at Markov decision processes, which is the formal framework, underlying reinforcement learning methods. We looked at two exact solution methods for the traditional MDP formulation, value iteration and policy iteration.
              
                  1:15:24
                  And then we looked at the max-ent formulation, which effectively turns it into a soft max iteration under the hood compared to what we had in regular value iteration and policy iteration. Now, of course, what we saw so far is great for small mDPs where we can have a loop over all states, all actions and do the bellman updates, either hard maxes or soft maxes.
              
                  1:15:49
                  But most practical MDPs will have a very large number of states where it's completely impractical to loop over all states and actions. And so the remainder of these lectures in this series will have us look at how we can deal with that, building on the foundations we already covered.
              


                英语 (自动生成)英语（美国）
                
                  00:01
                  Hi, this is lecture two in a six part series, um, on the foundations of deep reinforcement learning. Lecture two fits right here within all the others. In lecture one, we covered the foundations of MDPs and exact solution methods. We're going to build on that now to, uh, try to understand deep Q-learning.
              
                  00:25
                  So quick one slide recap of what we did in lecture one. We looked at optimal planning or control, which is given an MDP, which consists of a set of states, actions, often probabilistic transition model, a reward function, discount factor gamma, and a horizon capital H find the optimal policy pi star. We saw some exact methods to do this.
              
                  00:51
                  We saw value iteration. We saw policy iteration. They are exact solution methods. So in some sense you could think  aren't we done? But we're not. And the reason we're not done is because they have some limitations. First of all, they require access to the dynamics model or a transition model. Indeed, in the update equations, it has right there at the probability of next state, given current state and action, you need to have access to that, to run those equations.
              
                  01:18
                  That's not often the case that when an agent has to learn to do something in a new environment, that they already has access to exactly how that world works. And so we'll look at how to address that through sampling- based approximations, where the agent collects their own experience. And based on these sample experiences still is able to learn how to do well in these environments.
              
                  01:40
                  And then a second thing that was a limitation of the exact approaches is that they had a loop, a loop that loops over all states and actions and for any kind of reasonably interesting, um, situation, the number of states and often also actions will be really, really large. It'll be impractical to have a loop that looks at all of them.
              
                  02:00
                  And so we'll look at Q function and value function fitting, in fact, also policy fitting, uh, in later lectures to address that. Where instead of having a table, with the value or action for each state, because we can not have such a table there's too many states we'll have a function often in the form of a neural network that can take in a state, an output the corresponding value or action that you might want to take.
              
                  02:32
                  Okay. So that's effectively what the remainder, including this, the remainder five lectures will be about, is how to go to sampling based approximations and how to use function fitting rather than tabular approaches to solve MDPS. All right. So for this lecture the focus will be on a specific type of algorithms called Q learning.
              
                  02:53
                  We'll first look at you learning in still the tabular simpler setting to build intuition, and introduce these sampling based aspects. And then to introduce the function fitting aspects, we'll do a quick refresher on neural networks slash deep learning. And then we'll use them inside Q learning, which we'll call  deep Q networks, and which is actually the approach, and we'll specifically look at that approach that was used by DeepMind for the 2013 breakthrough reinforcement learning for Atari games.
              
                  03:26
                  So quick recap from the previous lecture, what are Q values? Q-Star S a is the expected utility as the expected sum of discounted rewards that the agent will accumulate when starting in state s, committing to action a in that state s at this time, and then onwards acting optimally. We had a bellman equation define those Q values.
              
                  03:53
                  So Q star s, a, we can decompose it into a contribution from the immediate reward and contribution of future awards summarized in the Q value at the next state. So specifically the Q value for state s and action a is the expectation, so the sum, weighted by probability, so the weighted sum, probability of state s prime, given when in state s took action a,, of the reward we got on the immediate transition plus gamma or discount factor, which makes things in the future, less valuable than things now, times what we'll get from state S prime onwards.
              
                  04:31
                  And what will we get from state as prime onwards? Well, how much value do we get there? Well, that's a recursive thing, really. Q star tells us. In state s prime, if we're taking the optimal action that maximizes this, then we'll get the max over q-Star s prime, a prime from that next state s prime onwards.
              
                  04:51
                  Now Q value iteration came down to effectively recursively computing, the values in this Belmont equation. So we initialized Q zero, let's say all zeros. And then from Q zero, we can find Q1. From Q1, we can find Q2, and so forth. And we also found that this will converge. We run this long enough and we'll have the Q star that we're looking for.
              
                  05:13
                  Okay. Now in tabular Q learning, q value iteration is great. We have to then do these updates, multiple iterations of these updates and in every iteration visit every state action and compute an updated value. This assumes access to the transition model and be able to iterate. Both of these we're going to address.
              
                  05:37
                  But for now, we're going to focus on not requiring access to the transition model. So we're going to rewrite this as an expectation. Q K plus one is an expected value of instantaneous reward plus future rewards summarized in the Q value at the next state. Once we have an expectation, well, expectations can be approximated by sampling and that's what we're going to do.
              
                  06:07
                  So instead of using the exact expectation, which we might not be able to do because in practical situations, we might not have that P S prime given S comma a that's, you know, the dynamics of the world. We might not really have them available, but we can experience transitions. We can experience samples.
              
                  06:25
                  Agent will actually collect those samples and we can then use those samples to estimate the value on the right-hand side, rather than, uh, computing it exactly. So for a state action pair (s,a) we receive a next state s' coming from the distribution. Then we can consider the old estimate and we can say, we have a new target, now, if that one sample was fully representative, then this should be the new estimate.
              
                  06:50
                  Our target that we use for the right-hand side, immediate reward, plus gamma times value from the next state onwards. Now this is a one sample estimate, so it might not be very precise. But it's an approximation and we'll see that in the algorithms we run we'll use many samples over time and there will be an averaging effect that will get us closer and closer to a more precise right-hand side.
              
                  07:15
                  And so we can incorporate this new estimate into a running average. So the Q k+1  the update that we had on the top on the left side, here was exact. Now we're going to say, well, it's going to be an inexact update because. And we, we only have one sample, so we should actually keep around what we had before with some weighting and then mix it with our new target.
              
                  07:37
                  And what is the effect here? This new target comes from one sample, but by doing this exponentially running average here, every time we get a new sample from state s, we can mix in new targets and we'll get this running average of straps. That's accumulated into our Q and we'll still get something close to the actual expectation.
              
                  08:00
                  Okay. So what does the algorithm look like? we start with the Q zero initialization for all states and actions. Then we'll get an initial state s that our agent is in. And then we're gonna run this. We sample an action a, agent will act, get the next state s'. If that's a terminal state, it's the end of an episode.
              
                  08:20
                  Well, then we've got to account for that. Our target is just to reward achieved in transition, and we'll have to reset our agent. Then, else, if it's not a terminal state s prime. Well, then we have the immediate reward R(s,a,s') plus gamma, our discont factor times the value from the next state s' onward.
              
                  08:41
                  That's  our target based on this one sample. Then, we will use this one sample to update our Q value for S comma a by mixing the old Q value with this new target value. And then our agent is next state s prime, and we'll keep running this. And as so, as we run this, our agent will keep experiencing state action, transitions and rewards, and the Q values will be updated everywhere the agent visits, and we'll get these updates.
              
                  09:08
                  We get an averaging effect over time that is similar to running the actual Bellman equation updates with the exact model. Okay. Now how do we sample actions? Because in what we just saw in the algorithm here, we have a sampling of actions. How do we choose those actions? Okay, well, you could choose the action that maximizes the Q value in the current state.
              
                  09:34
                  It's a greedy approach, kind of saying whatever I think is best, I'm just going to do it, um, That can work, to some extent that can be okay. But more popular is to not be fully greedy because if you're greedy, if something looks good, you keep doing it, but you have not had a chance to learn about anything else that might be even better.
              
                  09:53
                  And so epsilon greedy says randomly choose a action with probability, Epsilon, and otherwise choose the action greedily the one that maximizes the Q-value. So yes, you're often doing the thing that looks best, but you're also mixing in other things that's exploration. And that's important because by experiencing these other things, if we go back to the algorithm, we'll get updates for other states and actions that we wouldn't get if we just stick to the greedy one.
              
                  10:20
                  And so we can learn about other things and these other things could be better. So it is important to have this exploration. Okay. So what are some properties of Q learning? First of all, the amazing result is that Q-learning converges to the optimal policy, even if you're acting suboptimally, as you collect your data.
              
                  10:39
                  It's very, very interesting. It's called off policy learning. Uh, you know, you're doing this Epsilon greedy stuff. You're mixing in suboptimal actions, but these Q updates, nevertheless, will converge to the optimal Q values. And you can find the optimal Q values and associated optimal policy.
              
                  10:56
                  What are some caveats? Um, you have to explore enough for this to be true. So, your exploration mechanism was epsilong greedy or something else has to be present, and it has to have enough opportunity to explore enough. You have to eventually make the learning read small enough, so it doesn't keep hopping around.
              
                  11:16
                  So going back to the algorithm, the learning rate here, the alpha the learning rate is how much you go towards the new target. You need to decay that over time. Otherwise the latest experience, the one sample experience will make you hop around too much with every update. Then you have to also not decrease it too quickly because otherwise you cannot update enough to make up for new information that might have to correct past information that you incorporate early on.
              
                  11:48
                  So technical requirements, all states and actions are visited infinitely often, basically the limit doesn't matter how you select actions. That's just the thing you need to do , visit every state action infinitely often. And so, the learning rate, to make that more precise here is a mathematical condition.
              
                  12:05
                  The sum of the learning rates that you use over time has to sum to infinity, meaning there's enough power left on any given time, because if this sums to infinity, that means if you start at any future time past zero, it'll still sum to infinity, because what comes before will be finite. So you always have enough juice left effectively in your learning updates to correct for maybe unlucky past experiences that misled you.
              
                  12:30
                  But then also to make sure that the variance is bounded in this whole process, the sum of the squares of the learning rates has to be bounded. And so there are some papers listed below that give the theory behind this. Let's look at some demos. So here's the. Very simple robot. It's a box with a two link robot arm, and it can actuate a motor in here, a shoulder motor, and an elbow motor, or maybe think of as a hip motor and a knee motor.
              
                  12:58
                  We have a 2d state, two angles, and so it's not too large, but it is continuous. There's a continuum of states. So we're going to discretize it to be able to run our tabular Q learning based on samples, and let's see what happens. The reward is for moving off to the right in the forward direction.
              
                  13:17
                  And actions are, each angle here could be increased or decreased. Let's take a look at this. What we see is this robot initially is kind of just kind of in place, not doing much. And that's the exploration side of things. It's just kinda, hasn't learned much yet a lot has to happen before this really kicks into action starts moving.
              
                  13:40
                  Our Epsilon by the way is quite high. It's 0.8. So most of the time random actions are being taken. But it still starts drifting to the right a little bit. Cause sometimes the greedy action is starting to learn from all the experience collected. Um, keeps collecting data, keep collecting data and it slowly gets better.
              
                  13:57
                  And now we're going to change the Epsilon and bring it down. What will that do? Well, it means that when it chooses actions, now we're going to get to focus on the ones that maximize the Q value rather than being random. Let's see what happens. See actually now, as we scoot forward through this, the robot actually is moving forward pretty consistently.
              
                  14:18
                  So it's actually learned something, even though during the initial training was interleaving a lot of random actions and was great for exploration. You wouldn't be able to tell that it was learning. Once you've reduced Epsilon down to zero, you see that it actually learned a lot, and it's doing really, really well.
              
                  14:34
                  Okay. Then here is another video of the crawler action. What are we looking at here? We have to crawler on top and then here we have on the left values and on the right Q values. So it's a 2d grid,  because it's a two dimensional state space that 's discretized. And what would you expect to appear right now? Everything's initialized, uh, zero.
              
                  14:57
                  So it's red, not high reward, but as it's collecting experience, we expect as it has a positive experience where it gets good reward, that reward will go into the Q value over there. And that then will propagate from neighboring states visits that one, it'll propagate through. And we'll see some kind of fanning out of good rewards and it'll figure out what is the right things to do.
              
                  15:18
                  So, what we see here is indeed as it's collecting data, we see the values. There's an region where it's very green, which is good. Let's see updates happening in various parts of the space. We also see a clear region, emerge where it is better than other parts of the space. And so as we go through the learning and as we went fast, forward many steps here, we see a clear green region where it's really good to be.
              
                  15:42
                  And once you're there the agent will move very very fast. And then from the Q values, of course, it can read out the optimal action. So now the question is: can tabular methods scale? Let's think about discreet environments, grid world 10 states. We can definitely represent a table over 10 states.
              
                  16:01
                  Easy enough to do. But Tetris, if you do the count here there is 10 to the 60 states. We cannot store tables of that size and work with them. Atari, number of states is even higher, ten to 200, 300 or even 10,000 plus if working from pixels. These are very large state spaces and we don't want to store tables of this size.
              
                  16:27
                  We need to do something else. Continuous environment, even with crude discretization the crawler had a hundred states, but then a hopper would have ten to the ten, which is already a very large, it's 10 billion states. Humanoid with 10 to the 100. So it's not really practical to work with tables, with one entry for each of those, we need to do something else.
              
                  16:50
                  What can we do? What can we do instead of storing a tabular entry for each state? Well, in approximate Q learning, what we'll do is we'll instead of a table, have a parameterized Q function. Q used to be represented in what we talked about so far as a table with entries for each state and action. Now it'll be a function.
              
                  17:12
                  So Q will be a function that takes in a state s and an action a and output a value, the Q value for that state s an action a. Now we don't want to hardcode the function we want to have it learn the right function. So there is a parameter vector theta, and by changing the parameter vector theta, it'll represent a different Q function and have different values for different states and actions, depending on the choice of theta.
              
                  17:36
                  What are some ways of parameterizing the Q function? Well, it could be a linear function in features. Traditionally, this was very popular. These days, neural nets are more popular, but traditionally people would say, hey, let my Q function be a weighted sum of features. And maybe a feature could be something like, I don't know in Tetris, it could be how tall is my tallest column or how many gaps are there in the game board so far, and so forth.
              
                  18:05
                  But these days what's popular typically it's using a neural net. And so the neural network has many weights, theta is the parameter vector of the neural network. And you can put in as input state and action out comes  a Q value. Remember when we do Q learning, we have a target value: reward at the current time plus gamma times expected future rewards that we can look up in the Q function now.
              
                  18:29
                  And so when we want to compute that target value, we could now go look at that neural network and see what it says is the action that has the highest Q value. And then now with the neural network, we can't just say, okay, now make this state and action have this value because it's not explicitly keeping entries for state and action.
              
                  18:50
                  What we can do is we can say, hey our neural network representing Q theta, needs to nudge the value of the outputs for (s,a) closer to this new target that comes from our sample. And so we'll have this loss function here that says Q theta S a needs to be close to its target based on the recent transition into s'.
              
                  19:12
                  And, then we let's say do gradient based optimization on this to bring the parameter vector theta closer, well,  in a spot where this difference here becomes small. Okay, so clearly we're going to be working with neural nets. Let me do a quick recap of neural nets as a refresher. Neural nets are these networks where you have inputs, they get passed on from layer to layer to layer as they're being processed.
              
                  19:40
                  Each layer here has multiple units. So you have some numbers going in. Then this is one unit. It takes a weighted sum of the inputs, then squishes it through a non-linearity, passes on to the next layer. And this repeats repeats, repeats and outputs let's say a Q value. For example, if the inputs here or something about state and action, then it might output the Q value for that state and action.
              
                  20:03
                  Where they've been very popular of course, is image recognition as one example. And the canonical, well, the place where really known that's came of age, the modern era of neural nets. It's in the image net competition, where it was shown that with traditional computer vision, 2010, 30% error rate, 2011, not much better, 2012, not much better.
              
                  20:22
                  That was traditional computer vision without neural nets. Then Geoff Hinton and his students came in with a neural net approach, AlexNet that did way better. And then people switched to the deep learning approach, all across. And a lot of progress was made. And this is still the dominant method today for computer vision and essentially all other machine learning domains.
              
                  20:43
                  So this multi-layer perception type setup. Let's look at it a little more detail what's in it. There's a linear function that it starts with typically. So a single unit will say my F of X is a matrix w weighting matrix times the previous layer X. Then a multi-layer perceptron will stack this linear functions and nonlinearities in between.
              
                  21:03
                  So a two layer network will look something like: my output for an input X is equal to first, I do a matrix multiply with my input X. Then it is non-linearity where for each entry and this resulting vector, if they're below zero, set them equal to zero, if you're above zero, they'll stay the same, multiply it with another matrix.
              
                  21:23
                  A three layer network repeats this one more time. And it doesn't have to be this max non-linearity. There's others, though the max one is pretty popular. Could be a sigmoid non-linearity. A leaky ReLU. Tanh. ReLU is the one that's the maximum we've been seeing in the previous slide. So many variations.
              
                  21:42
                  Um, theReLU, tanh, sigmoid, leaky ReLU are probably the four most popular ones these days. And then you can build a multi-layer network that way. And again, keep in mind we're going to be doing here in this lecture, this network's going to represent a Q function. But presenting it in a more general way because these neural networks are used for representing many other things.
              
                  22:03
                  And in fact, in future lectures, we'll see how they might represent a value function rather than a Q function, or they might represent a policy, or they might even represent a learned dynamics model for the world the agent is acting in. Okay. So in this case, classifications of what's in an image with convolutional neural networks applied to images, let's say.
              
                  22:26
                  Okay, how do you optimize these neural networks? How do you find the right parameter setting? It's actually a non-convex problem, which for many years, people were actually scared of. That's one of the reasons a lot of people stayed away from neural networks for a long time, because it's non-convex, and so what if you get stuck in a local optimum? What can you really guarantee? But gradient-based methods actually are surprisingly effective and mini-batch stochastic gradients instead of full gradients uh, is
              
                  22:52
                  often used to speed things up. Gradient calculations. Is it, where do we get the gradient? Well, there is auto-diff frameworks like pyTorch,  TensorFlow, Chainer, and so forth. Most common methods these days are SGD, Stochastic Gradient Descent, plus momentum, plus some preconditioning in the form of RMS prop or Adam or Adamax.
              
                  23:11
                  All this things are things that should. If you were to implement this likely, you know, he's worked with tens of flora, PI tourists these days, and you would have it all available to you. You wouldn't have to implement the details of SGD, momentum or backpropagation, it's all taken care of for you.
              
                  23:25
                  You just choose the structure of your neural net and then you feed some data, and it then optimizes for you, finding the parameters of the neural net that fit your data best. Okay. If you want to learn more about this,  here are a couple of pointers where you can go learn a lot about neural nets in general.
              
                  23:41
                  We'll want to do here is see how they fit into reinforcement learning and specifically deep Q networks where actually in 2013, the first big breakthrough happened for deep reinforcement learning. So let's now go back to approximate Q learning. Instead of a table, we have a parameterized Q function, Q theta..
              
                  24:01
                  Often these days a neural network,  Q theta (s,a). What's the learning rule? Well, remember that, we're gonna get samples and each sample will generate a target. So by transitioning to state s', the agent generates a target, because it was in state s,  took action a landed in state s prime, there is reward associated with this.
              
                  24:19
                  And then there is the future reward summarized by the Q function in the next state and taking the best action in the next state. In tabular Q learning this target can be what we mix with the current value we have for that state and action (s,a), and that way gradually, we 're doing the right averaging.
              
                  24:37
                  When we're doing approximate Q learning with a neural network, we can not just do that kind of averaging, and we need to update theta. And so how are we gonna update theta? Well, Q theta is the function we're learning. And we have a target and we're going to say, well, we need to get close to that target.
              
                  24:56
                  And so we have an objective here, a squared loss objective in this case, and we could say, let's drive the error of this to zero by optimizing this objective. That might be a little over fitting because that's just really focusing on the last, very last experience. So more likely you take this objective, you do a few gradient updates and then you bring in a new experience and do a few gradient updates on the new objective that comes from that new experience, and keep repeating.
              
                  25:21
                  What does it look like in a full reinforcement learning algorithm that people might use? So this is the DQN algorithm directly taken from the paper that DeepMind paper on learning to play Atari games. So let's step through this in detail. Deep Q learning with experience replay. So there is some extra things here.
              
                  25:39
                  You initialize the replay memory. So there's a replay memory. So remember on this slide, I said, we have an experience from the agent (s,a,s'). We're going to use it as a target to update our Q function. Instead of just using it once, there's going to be a replay memory D where we're going to store past experiences, and then we can use those experiences multiple times in our Q function learning updates.
              
                  26:03
                  Then we initialize the Q function with some random weights theta. then we initialize the target action value of Q hat with some other weights. So there is two Q functions here. It turns out that,  by keeping track of two Q functions effectively, we're learning the same thing, but, a little bit out, out of phase, we'll stabilize the learning.
              
                  26:23
                  And a bit more about that in a moment. So we'll have two Q functions that we're tracking. Initialize our sequence, S one equal X one, and pre post process sequence five one. So there's some notation from the paper here, but essentially what they're saying is, as our agent is acting yet, in the Atari game it is getting observations, it's actually getting a sequence of frames as observations.
              
                  26:47
                  And that sequence of frames is preprocessed into a stacked frame, phi1 for time 1. And so everything that will be working with will be these phi's, which are the stacked frames the agent is working with because a single frame doesn't always have enough information because there's velocities involved as they move in this world.
              
                  27:05
                  So we were going to be working with these stacked frames. With probability epsilon, select a random action. That's our Epsilon greedy action selection mechanism that we already talked about. Otherwise it's like an action occurring to our Q function. So we looked at the current situation, encoded by phi.
              
                  27:23
                  We look at all actions we have available. So there's a bunch of joystick actions available. We see which one generates the higher Q value and take that action, if we're not doing the Epsilon thing. Execute the resulting action in the Atari emulator and observe a reward associated with that. The score might go up, which would give us a reward and observe the new image.
              
                  27:43
                  Then we do some processing on the image to, again, to those stack frames, to also understand the velocities. And we store the transition from effectively state phi_t, action a_t,  reward r_t,, and next state phi_t+1,  into the replay buffer. That's an experience that we can use to generate a target, right? And we sample random mini batch of transitions.
              
                  28:11
                  So we have just a new experience. We don't just use that new experience. We sample a bunch of past experiences. And then for each of these past experiences, state action reward state, we're going compute the target value. If it's a final termination state, with just the reward experienced. If it's not a termination state, then it's going to be the reward experienced  plus discount factor times the value at the next state.
              
                  28:39
                  And we use the Q hat here. So it's interesting. Cause there's two Q functions that play two known that's being trained effectively. Q hat is the one we use for the target Q values. But other than there being two this is exactly what we've been talking about. And then we perform a gradient step to bring the Q function that we're learning, Q theta closer to the target values, which are called y's here.
              
                  29:04
                  And then we repeat, and then periodically we set our Q hat equal to the Q that we're learning. So just Q had is something that's lagging behind. It's lagging behind on the Q that we're learning and choosing actions with. And the reason that's done is to stabilize because if the Q that we use for generating targets, changes too much, it's easy to introduce instabilities.
              
                  29:25
                  So we want to have our targets to come from this stabilized lagged Q function. So this is the DQN and algorithm. there's one more detail that they use to make sure you don't overfit to  specific target values. They use a Huber loss instead of a squared loss. A squared loss, when you're away from zero it grows, like, it's parabola, squared loss.
              
                  29:47
                  The Huber loss is a parabola at the center, but then at some point becomes linear. And what that means is that any single example, any single target can only contribute so much to how you're gonna update the weights of your neural network. And so you have more averaging happening rather than outliers potentially dominating your updates.
              
                  30:11
                  And then there's some annealing of the exploration rate. Initially there is a lot of exploration, and Epsilon goes closer to zero as we go along. And they don't use just a standard gradient update, but they use RMS prop, which is essentially a rescaling of the gradient updates, that is generally found to work better than just gradient updates in this case.
              
                  30:31
                  If you do that, actually the results they showed is, it can do really well on a range of Atari games. It learned the neural network, takes in pixels, and knows the Q values for all actions in that situation. And that allows you then to select the optimal action. They used a 3 million parameter network, some hyper parameters related to Huber loss and learning rates and Epsilon greedy.
              
                  30:55
                  But roughly achieves human level performance on 29 out of 49 games with the approach we just described. Under the hood, the neural net itself had a convolutional architecture much like the ones used in computer vision at the time that won image net competitions. And the results here: on the horizontal axis is a listing of all the games.
              
                  31:14
                  Vertical axis is performance scaled by human level performance, a hundred percent means human level performance. And we see that about two thirds of the games, the ones on the left here, have human level or better than human level performance. This was back in 2015, the Nature version of the DQN paper came out at that time.
              
                  31:32
                  Since then there's been improvements on this, but this is the big breakthrough result at the time. So what are some improvements on top that have happened? Double DQN says, well, when we take the max over actions in our target calculation, well, there might be some kind of upward bias. Our Q functions become overestimates because if randomly some action was initialized at a high Q value, it will be favored in this max.
              
                  31:58
                  How do we counterbalance that? Well, you're actually gonna use since we have two Q networks already anyway, we're gonna use one of them to see which action achieves the max and then use the other one to see what value it has for that action. And so that way there is some independence between how the action is chosen, the effect of the random.
              
                  32:21
                  So the randomness might not carry over between the two. And this helps a lot in terms of stabilizing the learning and learning actually a lot faster. And, all DQN implementations that I know if today are all double DQN implementations, actually using this idea to split the arg-max. Take the argmax on one Q function, and then the max gets taken using that action from the other Q function.
              
                  32:48
                  Another idea that's often used is prioritized experience replay. Well, you have this buffer of past experiences. Is all this data equally valuable? Maybe some data you can learn more from than other data. And so in prioritized experience replay, you keep track of the Bellman error, how much the target value for a state, action, reward, state, quadruple how much the target value is different from what the Q function currently thinks.
              
                  33:11
                  And so if the target value is very different than what the Q function would have predicted, then there is a lot to be learned here. And so you get a higher priority. A uniform dQN in gray is the bottom learning curve in both cases. And we see the different versions of experience replay that are prioritized based on Bellman error actually does better.
              
                  33:32
                  There are more things people have done. Something called dueling architectures; distributional DQN where you don't just try to have the Q function learn the expected rewards, but actually learn a distribution over rewards you might experience in the future. And then there is a noisy versions of DQN, which is another way to introduce randomness in the actions you choose to have better exploration.
              
                  33:55
                  And actually there is a paper called a rainbow DQN, which combines all of these and achieves still today some of the best performance on Atari. For Atari  rainbow DQN is the natural starting point for anything you would do.
              


                英语 (自动生成)英语（美国）
                
                  00:00
                  Welcome to lecture three in this six lecture series on the foundations of Deep RL. In lecture three, we're going to look at policy gradients and advantage estimation. Um, how does it fit in the bigger picture of our lecture series? Well, we already looked at what are MDPs and exact solution methods, which of course only apply to smaller MDP.
              
                  00:24
                  Then we looked at deep Q learning, which was our first method for solving larger MDPs. And now we're going to look at some alternative methods for solving larger MDPs starting with policy gradients and advantage estimation. First you might say, well, why, if we already saw deep Q learning, why do we need yet another method? Well, today's state of affairs in deep reinforcement learning is such that there are multiple methods out there, multiple types of methods out there that are all, uh, good and have, you know, their preferred use cases.
              
                  01:01
                  And to give a little bit of extra perspective on that deep Q learning time methods are very data efficient, which is nice, but are often not as stable as some of the other methods we'll look at. And also if data efficiency is not your bottleneck because maybe all your data is generated inside a simulator and so data really means compute.
              
                  01:26
                  Then maybe you want to, you know, how do you think about compute? Do you want to simulate more or they want to do more Q learning updates and so if your data can be collected very fast, um, often what are so-called on policy methods, which we'll look at in this lecture, uh, can be more effective in terms of your wall clock time, then, uh, something like a deep Q learning method.
              
                  01:48
                  Okay. So what are we going to look at this lecture? First policy gradient derivation,  pretty mathematical the whole of this lecture, actually going to be pretty mathematical. This is really about, uh, getting through the foundations of policy gradient methods. Uh, we'll look at the basic derivation, then we'll take advantage of temporal decomposition to be more data efficient then we'll look at baseline subtraction, which will reduce the variance of our policy gradient method.
              
                  02:16
                  Then we'll look at the value and function estimation, which can further reduce variance. And then we'll look at advantage estimation as a way to further improve our policy gradient method and actually start bringing it quite close to actor critic methods. And it's kind of a fuzzy boundary. Uh, you might already call them actor critic methods once we do that.
              
                  02:37
                  So reinforcement learning, uh, with agent interacting with the world, taking decisions after taking an action, the world changes call environment here. Um, just a new situation. The engine encounters as a consequence, takes an action again. This process repeats over and over and over. In policy gradient methods under the hood,  this agent is really going to be a policy PI theta, which chooses an action.
              
                  03:00
                  You based on the current input as the state or image observation of the world. And so for us, this will be a neural network taking in the current state or observation, and then generating a action. And it's known that work is parameterized by a parameter of vector theta, which is the way it's in the network.
              
                  03:18
                  And we're trying to learn the right setting of the weights. To maximize expected reward. That's the goal here we wanna maximize expected reward. Um, typically this neural network will output a distribution over actions. So PI theta for a given state input will just output a distribution over  actions.
              
                  03:38
                  You might say why? Haven't we earlier on seen that the domestic policies can be optimal? So why also consider stochastic policies? Well, in practice, It smooths out the optimization landscape is the way I like to look at it. If policies are deterministic, two policies are always kind of two different policies are in some sense, far apart.
              
                  03:58
                  There's no continuum between them making the optimization harder, but by allowing for stochasticity, there's this stochastic interpolation between them. That's all part of the policy space and we'll get a smoother optimization. There's another reason you might prefer stochastic policies.
              
                  04:17
                  As the agent is learning it needs to collect the data to learn from. That data collection needs to somehow explore the world. And stochasticity can also help with that. So under the hood, our neural network will represent a distribution over possible actionsgiven the state that's its current input. So why a policy optimization? Um, well often a policy can be simpler than a Q function or a value function.
              
                  04:47
                  For example, a robotic grasp. You want to pick something up, that's about moving the gripper to the object and closing, but do you know exactly how long it's going to take you? Do you know exactly the grasp quality metric you can assign to what you did that might be harder to calculate, but the right strategy might be simpler to represent and so faster to learn.
              
                  05:10
                  A value function also doesn't prescribe actions. If all we learn is a value function, we still don't know what to do. You need maybe a dynamics model to do look ahead against the value of function, but then you also need to learn a dynamics model. It's not impossible, certainly some people, uh, do do that.
              
                  05:25
                  Um, but you know, it's another method and it has its own downsides there by needing all that extra machinery. How about a Q function? Isn't that enough? We can read off the correct action from the Q function, um, well it's not always efficient, uh, to solve for the arg max over actions in a current state, because that's an optimization problem in itself.
              
                  05:45
                  And so maybe when we learn a policy, we right away can read off the solution, which could be quite convenient. So how are we going to compute our policy gradients? The methodology is called the likelihood policy likelihood ratio policy gradient. And this is really the core of what we're going to cover in this lecture.
              
                  06:09
                  So a little bit of notation that's new for this lecture to keep things compact and readable. We let tau denote a state action sequence. So, state at time zero, action u at time zero and so on till state at time H and action at time H. And so tau will denote an entire such state action sequence. And then we'll say the reward for our trajectory tau is a sum of the rewards for each of the state action pairs, because we know reward is typically associated with state action pairs or state action, next state triples.
              
                  06:47
                  And so we're going to have some new notation here reward for an entire trajectory. Under the hood, it'll still be associated with state and actions, but just to simplify our notation. Then what we're optimizing is the utility. So every choice of our parameter vector theta of our neural network will achieve a different expected sum of rewards when we use that policy.
              
                  07:08
                  And we can write this as the sum over all possible trajectories we could encounter, the probability of experiencing that trajectory under the policy parameterized by theta,  and then of course that's what we weight the reward by. So this is our objective here. And so in our new notation, our goal is to find a theta, the parameters in the neural network that will maximize U of theta, which is a weighted sum of rewards associated with each possible trajectory weighted by the probability of that trajectory.
              
                  07:43
                  And by changing the parameters of the neural network, we're changing the probability distribution over trajectories. Some settings will favor certain trajectories. Other settings will favor other trajectories. And so we want to find a setting of the parameters in the network, such that we favor the trajectories that have high reward and don't favor the trajectories with lower rewards so much.
              
                  08:06
                  Good. So how do we do this? We're going to again, do a gradient-based optimization as is quite common in everything we've talked about and generally done by others. So you U of theta is our objective, and we want to take the gradient with respect to theta. So theta lives in here. Right there, our gradient is put inside the summation.
              
                  08:29
                  Now, um, well, R tau the trajectory, it doesn't have theater in it, it's just a specific trajectory, but theta is over here. So here's where our gradient is going to act. But when we look at this, we can maybe think about how to compute this gradient, we'll see later how to do that. We have a, sum over all trajectories, but then it's not weighted by the probability.
              
                  08:56
                  And so what we really want is a weighted sum, because then we can sample from the distribution to compute this thing. So we're going to multiply and divide by the probability of a trajectory. And now reorganizing, we now have a weighted sum. And so we have now an expectation here where we have a gradient that we'll compute in the back here, but then it's an expectation with respect to our trajectories collected from our current policy.
              
                  09:22
                  So what we can do, we  can actually compete a sample based approximation. We don't have to enumerate all possible trajectories, which would be completely impossible in realistic, uh, problems. We're going to just take a sample based estimate. And so also typically what's done is this is rewritten that grad of P over P is the grad of log P so grad log P here, but then we can pick an empirical estimate of this thing.
              
                  09:46
                  And so our gradient estimate will be an empirical estimate of an essential, an average of the gradient in the back here, multiplied with reward. So this is interesting, um, to get our gradient estimate, we're gonna use our current setting of the parameters to a bunch of roll-outs, and then for each rollout, we're going to compute grad log probability of the trajectory under the current parameter setting times the reward collected along that trajectory.
              
                  10:17
                  Of course, this is still a, um, a lot of work to be done here, so we're not done, but that's the general structure. Now what's interesting when you look at this is that this can be used no matter what our reward function is. What I mean with that is we don't need to be able to take a derivative of our reward function for this to work, the only derivatives taken are with respect to the neural network.
              
                  10:40
                  Well, with respect to the distribution of our trajectories, which is induced by the neural network that encodes our policy, we don't need a gradient with respect to the reward function. And so you can have a reward function that is for example, one when you achieve the goal, zero everywhere else, and this will work.
              
                  10:57
                  We're computing gradients with respect to expected rewards. And it's because of this probability distribution distribution over trajectories, that's inducing smoothness and we can still take gradients. Okay. So, um, now what's some intuition behind this equation. Let's say, we look at this gradient here, we collected some trajectories, we look at the grad log probability trajectories and the reward.
              
                  11:25
                  Well, what's going to happen if the trajectory has very high reward, then the gradient here, so the log probability of that trajectory will be increased. So trajectories with high reward will get an increase in their probability. If now we have a trajectory with a very negative reward we'll move the parameter vector theta by moving in the gradient direction to decrease the probability of that trajectory.
              
                  11:55
                  So we're really kind of pushing up for rewards that are positive high and pushing down the probability for trajectories, uh, where the reward was negative low. So when you think about this,  it's saying shift probability mass away from trajectories with bad rewards, and shift probability mass towards trajectories with high reward.
              
                  12:19
                  It's a bit more subtle than that because probabilities have to integrate to one. And so by pushing probability up in some places you're implicitly also pushing it down in other places and the other way around. So there's more going on than just pushing it up on the ones that are good and down on the ones that are bad , there is  more happening because of the normalization.
              
                  12:39
                  Okay. Now so far, we've talked about entire paths and considered entire paths, but often rewards are more localized. So we might not want to do this kind of shift based on an entire path. So let's decompose the path into states and actions. So we have this grad log probability of trajectory under the current parameter vector theta of the policy.
              
                  13:10
                  What is it under the hood? It's a product of probabilities, the probability of next state, given current state and action and probably of action given state. And we have this grad log product of probabilities,  log of a product is some of the logs so we can change the thing with some of logs. Yeah. The gradient of a sum is a sum of the gradients.
              
                  13:32
                  So it can move that great in inside the sums. Um, now what happened here? Well, I could drop this first one cause there's no theta in it. So nothing in his first term will have a gradient contribution. We're just left with this second term here, so grad log of pi theta, the probability of policy assigns to the action taken given the current state.
              
                  13:56
                  Um, and so what's very interesting here is that no dynamics model is required. All we need is looking at the policy PI theta. Interesting, right? Cause we started with wanting to increase or decrease the log probability of a trajectory that was experienced and that trajectory consists of dynamics and policy.
              
                  14:14
                  But because the parameter theta does not appear in the dynamics model and it's part of the policy and because this is a log of a product and becomes a sum of the logs, and because gradient of sum is sum of the gradients, all we're left with is this part over here. What we see is to increase the probability of a trajectory, we would increase the log probabilities of actions along that trajectory.
              
                  14:37
                  And to decrease the probability of a trajectory, we'll decrease the log probabilities of actions along that trajectory actions in specific states. Okay. So we had a great end estimate that looked like this grad log probability of trajectory, times reward along the trajectory. And we now know that this first part here, we can actually compute as a sum of grad law probability action, given states along the trajectory, no dynamics model required.
              
                  15:04
                  We can do this directly based on a neural network that represents our policy. And so now we can start computing gradients, you can roll out the current policy, then, you know the rewards along each trajectory, and then you also know the state and actions along this trajectory. For a neural network do a backpropagation for each state and action experienced computer grad log probability of the action given state, and then accumulate the grad log probability of  the trajectory multiply with your reward along that trajectory, and you have your gradient.
              
                  15:35
                  So at this point, we can actually do this. What can do a likely to ratio policy gradient. Now as formulated thus far. Sure it computes the on expection the correct policy gradient, but it's very noisy. It's sample based. And if you don't have enough samples, not going to be very precise, um, what are some fixes that will lead to real-world practicality? First of all, we'll introduce something called a baseline.
              
                  15:56
                  We'll leverage even more temporal structure than we've done so far. And in the next lecture, we'll look at trust regions and natural gradients to further improve our updates. Um, so what was the intuition again? We had this grad log probability of the trajectory times reward, and we said, we're going to increase the probabilities of trajectories with high award, decrease probability of trajectories with low reward.
              
                  16:21
                  But actually really we want something a little more subtle, right? What we really want is we want to increase the log probability of trajectories that are above average, and decrease the probability of trajectories that are below average. Because if something's below average, you want to do less of it as above average, we want to do more of it.
              
                  16:40
                  But here, if for example, you have an MDP where the rewards are always positive for every state. The rewards are, let's say between zero and one, then no matter what you experienced, you're always going to have a contribution here that says let's increase the law of probability of what I did.
              
                  16:54
                  Um, sometimes by a little bit, sometimes by a lot, and you're gonna have to have a lot of averaging effects where the things are in, you know, have  their log probability increased by a lot, um, start dominating over the ones where it's increased by a little bit, but really you'd prefer to maybe for the bad ones, closer to zero reward to just bring the probabilities down.
              
                  17:17
                  Can we do this? Um, because in that way, maybe shift our probabilities in a better way. Something called baseline subtraction will get us what we want. So we had this gradient here grad log probability of trajectories times reward now consider introducing a baseline b. So we grad log so that you can trajectory probably have trajectory times reward minus baseline.
              
                  17:45
                  Say, well, you're just going to subtract something out. Yeah. How about we subtract out the, if a bunch of rollouts, we look at the average reward along all trajectories, and we subtract that out as our baseline. We call it B. Then things that are above average will have an increase in their  log probability of action given state and the ones below baseline will have a decrease in their probability of action given state and accordingly  trajectories.
              
                  18:11
                  Can we do this? Is it still an okay, gradient estimate? Well, you can work through the math here, this side of the slide, but if you work through the math, what you'll see is that on  expectation, this extra term will be zero. It's a very interesting why we even care about it, why are we adding this minus b if an expectation it's a zero contribution? Well, on expectation it is.
              
                  18:38
                  But when there's finite samples, the estimate we're accumulating here, it will actually have a reduction of variance effect, which I'm not showing on the slide here, but which can be shown with the right baseline, the variance will be reduced. You'll get a better gradient estimate. And the one I mentioned, the average of rewards experienced along all trajectories is a pretty good one to reduce your variance with.
              
                  19:04
                  So what else can we do? We now have this new thing. Our gradient estimate is now grad log probability of trajectory, times reward minus this baseline, which we can think of for now as the  average reward along all trajectories. Um, so what else can we do? The reward also has a temporal decomposition, we already did a temporal of decomposition for the trajectory probability, which resulted in the grad log probability of action given state, we can do the same thing for reward.
              
                  19:34
                  And then we can start thinking through, well, should all these terms participate? Is it meaningful? When I think about grad law probability of an action given state that would be multiplied with a reward from the past? No, because actions that take now only influence the future. And so what can split this into rewards from the past, rewards from the future.
              
                  19:59
                  The rewards from the past are not relevant. And if you do the careful math, you'll see that indeed the expected value coming from rewards from the past is zero. So you can just remove it without a loss in accuracy in any way. And so after you do that, um, we can also see that you can actually let this bias depend on the state.
              
                  20:21
                  So you don't need to look at the entire trajectory. We can take the average of rewards experienced from, let's say this state onwards, or from this time onwards as a thing you subtract out. So we removed terms that don't contribute, but only introduced variance would get rid of that variance by removing those terms.
              
                  20:39
                  All of a sudden we have, what is the practical policy gradient equation. So what does it look like? We're going to have a bunch of rollouts, M rollouts. Each rollout is a bunch of steps in it, and we accumulate the grad log probability of the action we took in state s_t at time t and we multiply it with the reward from then onwards.
              
                  21:03
                  So the rewards, the action actually influenced, and we subtract out a baseline, which is what do we expect from that state onwards that we were in a time t. And so if we had a above average performance after the action, then the action probability will be increased with this update. If we had a below average performance, uh, from that action onwards, then the probability of the action will be decreased.
              
                  21:28
                  So we're shifting probability mass, specifically effectively for each action based on whether what we experienced after the action was below or above average. Okay. So it's important to realize this is from this time onwards, because that's the only thing is the action is influencing. Same for that, um, baseline we subtract it's how much reward you you expect from that time onwards? Um, because that's the only thing the action can influence.
              
                  21:56
                  Okay. What are some good choices for B I've loosely alluded to this notion of average. Because subtracting out the average gives us a notion of, are we above or below average. Um, well, um, that's a pretty good choice, but let's step through that. The choices people tend to use constant baseline is just, you know, you say take the average for the entire trajectory.
              
                  22:17
                  Um, but you can also try to optimize a bit more. There's something called a minimum variance baseline. I actually have not seen people use in practice, but there's a formal derivation you can do that rather than just taking the average. You're going to take a weighted average. When you average the reward along the trajectory, you weigh it by the, um, square of the grad log probability.
              
                  22:42
                  So what does it mean that, um, when you're great in this very high as a high number, You will more heavily weigh this trajectory in your average, then trajectories that have a lower gradient and you work through the math overall that works out to a lower variance policy gradient estimate. But of course you have to do a bunch of extra work.
              
                  23:04
                  Um, and practically I have not seen it in, in today's or last many years of, uh, implementations that people use, but maybe it's time to revisit who knows. Maybe it's something that. Uh, you, you could show in some research you do that actually should be revisited and we should be using. Um, I'd be curious if you, if you find out that we should be using it.
              
                  23:28
                  Time dependent, baselines are very popular. Why is that meaningful? Let's say you have finite horizon roll-outs. Well, at a later time, there's less reward left than at an earlier time. And so it's a nice way to capture that. Um, even more precise could be state dependent baselines for a specific state and time possibly how much reward do you still expect? And that's essentially a value function under the current policy.
              
                  23:52
                  Remember in lecture one, we talked about policy evaluation? What is the value function for this specific policy? That's what you would use here. And we'll see more about that later, how we can do that in these larger scale problems, but you would compute a value function and then use that as your base.
              
                  24:08
                  So once you use this, you get an increase of the log probability of the action proportionally to how much it's returns are better than expected, um, under the current policy. And to me, this value function as a baseline is most intuitive because if Al function is how much an average does my policy gets from this date onwards, and then you're saying, well, if I did better than that, well, I chose a better than average action.
              
                  24:32
                  If it did worse than that, I chose a worse than average action and I'll update accordingly. Of course need to be able to then still learn that value function, but we'll see soon how to do that. All right. So let's take a look at that now value function estimation. How do we get that value  function that we can use as a baseline? Well, um, that's where we're going to use it.
              
                  24:55
                  There's many ways to estimate it many, many ways. Here's one way we initialize a neural network and it flies with some parameter vector phi zero, and we try to estimate the value function for a policy pi. We collect trajectories, just like we're already doing for our policy gradient estimates.
              
                  25:11
                  And then we just do a regression against the empirical return. So this becomes a supervised learning problem. We say, hey uh, I have rewards that I accumulated from a certain state onwards in one of these rollouts, rollout i.. So that is a Monte-Carlo estimate of the value of that state. S K I, and I want my value function as represented by my neural network to be close to that.
              
                  25:38
                  So now I have to find a loss function based on squared error, or maybe Huber losses to be more robust, to any outliers. And I just run an optimization here. Standard deep lear ning framework and, you feedin effectively the sum of rewards as targets as the outputs of the network, the state, as the input, and then you still want it to predict, from state to value.
              
                  26:03
                  And then depending on how you do this, you might have maybe small batches of trajectories and you do a small number of updates, and then get more trajectories, do a few more updates and repeat, or maybe collect a large number of trajectories do a full optimization. Um, that kind of depends a bit on how you're running your algorithm.
              
                  26:23
                  Okay. That's one way to do this. And this is often the thing you might do first when you're implementing your own policy gradient approach. And you're going to estimate the value function. It's a natural first to go with because it's relatively simple. It's just a supervised learning problem.
              
                  26:38
                  And so it's very well understood what you can expect here. But you can also do bootstrapping. Remember the Bellman equations for finding value  functions, value iteration why not use that? Um, in this case it would be for policy evaluation. We know our policy V pi of s is equal to some averaging over the actions where the probability of action probably of next state and Then we have reward for the transition plus future value.
              
                  27:06
                  We've done this in the exact case in lecture one, what can they also do as an approximate sense? We can collect data and we put the data in our replay buffer, and then we say, well, let's do what is called fitted value iteration. We have targets, reward plus value at the next state, just like in Q learning really.
              
                  27:24
                  We have the neural network that we're currently updates. And then, um, We, uh, do some grant upsets and maybe some regularization to keep our neural net parameters phi close to the neural net parameters of the previous iteration phi i do not jump too far, from where we were before, because we want to gradually optimize this because we're just getting a little bit of new data, do a little bit of an update and get new data to a bit more of an update and so forth.
              
                  27:50
                  So it's a bootstrap approach to estimating the value function. Then what does this give us for full algorithm. This is what some people call  vanilla policy gradient, uh, kind of a simple version of policy gradient. You initialize the policy parameter vector theta. This is typically the weights in your neural network.
              
                  28:08
                  And then the baseline B, which often would be a value function of some types of maybe another neural network, but it could also be something simpler. Then we iterate we collect a set of trajectories by executing the current policy. At each time step in each trajectory we compute the return from then onwards.
              
                  28:26
                  SoR_t here is the return from time t onwards, discounted some of rewards from that time onwards and the advantage estimate. What is the advantage estimate? It's the difference between the return from that time onwards with the baseline, because we really care about how much is this better than average.
              
                  28:43
                  Then, after we've collected all that, and we know how much we're better than average for each, time onwards in each trajectory. We're going to then do some learning. We're going to first refit the baseline by having our neural network that in this case approximates the value function. Use the supervised learning approach, a Monte Carlo approach fit it to the, uh, rewards that we got then onwards in our rollouts.
              
                  29:06
                  And. Optimize them and all that work to have a better fit here. Um, and then, um, we can also update the policy with some gradient estimates. Uh, we know grad log probability action given state times the advantage, associated with that state. Okay. And of course, fitting the baseline here the value function could be a neural net value function that depends on state that's being learned here.
              
                  29:30
                  Like we saw. That's this thing over here, Monte Carlo estimation of V pi. But,  of course you could swap in this bootstrap estimate over here. Um, you'll still have a policy gradient algorithm just as well. Um, often people think of the bootstrap estimates as likely being more sample efficient, but sometimes a bit less stable.
              
                  29:50
                  And so often start with a Monte Carlo version. And then once you have that up and running works well, you might see if you can also get it to work with a bootstrap version and then maybe slightly more sample efficient way. Okay. Now this advantage estimation, this thing that's multiplied  with the grad log probability is really key to further improving the efficiency of what we looked at.
              
                  30:12
                  So a lot of work gone into that, and there are two quite popular approaches. There's advantage estimation as proposed in A3C slash A2C and there is generalized advantage estimation, which are very related and we're going to cover now. So at this point, our policy gradient method has this grad log probability action given state multiplied with the advantage .
              
                  30:36
                   This is the future rewards experienced minus the average we would expect the value of that state. Okay. Using the value here is kind of interesting. Um, something we learned about a state on average, how well we did. But then this thing here is still a single sample estimate. And you might wonder, can we do something better there than just a single sample estimate? So how do we reduce the variance? Even for a single rollout, we can reduce the variance.
              
                  31:04
                  And the way that's done is by introducing discounting. Um, another way that can be done is by introducing function approximation. So discounting, instead of using the sum of future rewards, you can use a discount at some of future. You might say, well, that's kind of weird. Haven't we talked about that as a problem definition haven't we said an MDP is defined by a discount factor and that discount factor is capturing things like, well, we could earn interest rate on money if we had it earlier and so forth.
              
                  31:32
                  That's correct. The problem definition of the MDP has a discount factor. It turns out that algorithmically the discount factor can also be a hyper parameter you want to play with. So it's kind of interesting because you have a problem definition with discount factor, but also algorithmically, it might be something you play with.
              
                  31:53
                  Why is that? Well, think again about what we're doing with looking at the grad log probability of an action given state and seeing how good that action was. You could say, well, maybe that action has more influence to things that are nearby and that things that come much later. So maybe when I look about, is this action above average or below average.
              
                  32:13
                  I shouldn't look at the, everything in the future with equal weighting, I should discount things further in the future. Cause the action has less influence on it. Is that always true? That depends. Sometimes an action now can have a very long-term influence, but very often it's the case that actions have more influence nearby in time than they have later in time.
              
                  32:32
                  And so this discounting here is not about the economics of, let's say, earning interest on your reward, money or something. It's about discounting because the effect of actions tends to decay over time. And so by discounting, you're putting in that prior information and actually get a lower variance policy gradient which will help you learn faster.
              
                  32:53
                  Of course, you then also have to do it for the baseline. We compute the value of function baseline. You have to do discounting there too. You have to write a difference. What else can we do now? We need to have a discount at some of rewards, but can we do it? Well, some of your words is just like the one thing you experienced.
              
                  33:11
                  Uh, as I alluded to it, maybe we can have some kind of value function instead of here. Actually we could, we could say reward experienced plus value from the next time onwards. Or we could do reward now, plus reward at next time plus value from s2 onwards . And we can keep doing this. There's many options.
              
                  33:30
                  All of these could be reasonable. And so the asynchronous advantage actor critic, A3C,  uses  one of the above choices, which becomes a hyper parameter. For example, k=5, the last row here has k equal 3. k equal five was a popular choice in their paper. They also looked at k equal 10. So what's going on here? Why, why using this mix? Um, well, in the limit, when you use the top row, if you have enough samples, you know, that's exact, that's exactly what you want.
              
                  34:04
                  Once you start using value function estimates, you might be introducing some error, but the benefit of introducing them is that you're reducing variance because it's an estimate based on many past experiences, that's now bundled into this value function. And so sometimes you're trading off a low zero bias estimate, which is the top row, but high variance, with in the second row, you have a very low variance, but now possibly high bias estimate and somewhere in between might be the optimal way of estimating your advantage for maximally fast learning.
              
                  34:41
                  And that's what they did in there. And the generalized advantage estimation work, uh, which I actually did in my lab. We saw that it's actually possible to take an exponentially weighted average between all of these in an efficient way. And so you can have a  exponentially weighted average between all of these choices, which in our experience gives even a better trade off than making one specific choice . Estimates will depend a bit on, um, your choice of lambda.
              
                  35:08
                  Okay. That's actually very related to, in some ways, actually equivalent to TD lambda eligibility traces from much earlier work by Sutton and Barto. Okay. So now let's take a look at our more complete approach: policy gradient with A3C or generalized advantage estimation. What does it look like? Initialize a neural network for encoding the policy, initialize a neural network for encoding the value of your current policy.
              
                  35:36
                  So two neural networks, sometimes they're the same network with, um, a lot of it's shared and then two different heads. One head for the policy, one had for the value they can also be separate networks. Uh, that's kind of up to you and you collect rollouts and these rollouts result in state action, next state reward quadruples, and you store those uh, for, for your update, that's about to come and gives you estimates of the Q value experienced because you have a rollout.
              
                  36:07
                  You have a Monte Carlo estimate from a specific state and action onwards of what you experienced. Then there are two things we want to learn. We want to learn a value function. We want to learn a policy. First update here is on the value function. So the value of function we regress. So we want it to be close to our Monte-Carlo estimates here.
              
                  36:28
                  And then there's some regularization to make sure we don't update too much based on the latest data. And then we have our policy update, which is our standard policy gradient; sum over all trajectories, sum over all state-action pair s, and those trajectories of the grad log probability of action given state times, and then this is the advantage is the reward experienced from that time onwards, minus the value.
              
                  36:57
                  Now reward experienced from that time onwards could be literally reward, the Monte Carlo estimate, or we could put estimates there that we saw over here, we can use a,GAE estimate or an A3C estimate  for that Q i hat. Then there are more variations. You could use a one-step for V full rollout for pi that would look like this.
              
                  37:20
                  So here's a bootstrap version of estimating V that's okay. Um, and then for pi here is a full rollout. So a full Monte Carlo estimate rather than a advantage estimate based on using also value functions. So, many variants but this is the general structure of what these policy gradient algorithms will look like.
              
                  37:41
                  There's a little more, what we'll cover in the next lecture that can make this even more efficient to stabilize the optimization a bit more, but this is the main intuition behind policy gradient algorithms, and once they use sophisticated advantages, it often also gets called an actor critic algorithm.
              
                  37:59
                  We'll look at some results here. This is from the A3C paper. This is on a few Atari games in yellow is the A3C algorithm is doing very well. And it's here in comparison to DQN remember in lecture two, we looked at DQN and we see this A3C approach is actually more efficient. This is wall clock timing, or how long did you have to run.
              
                  38:18
                  And performance gets better and better, more quickly with A3C than with DQN. Intuitively. Why might that be the case? Well, DQN puts things in a replay buffer and visits them multiple times, try to extract sometimes more signal from each experience, but also spends a lot of compute time doing so. Whereas A3C says I have a current policy I'm going to use the data I collect on the current policy, improve my policy and then collect new data.
              
                  38:43
                  And so you collect data in an environment best on the latest policy, which is maybe the most interesting, possible data you could get, do updates based on that. And so that can give you often faster training times, even if we maybe were to replot this in terms of sample efficiency, the sample efficiency of DQN might actually be better, but the speed of learning of this policy gradient method, in this case A3C, is faster.
              
                  39:08
                  They also did this, not just in Atari, they also did this in this kind of DeepMind labyrinth environment where the agent has to navigate, find rewards for finding apples and so forth. Under the hood is now a recurrent neural network, as it needs a bit of memory as it's navigating and finding rewards.
              
                  39:25
                  Some older work actually by Russ Tedrake for his PhD work back then at MIT. He showed it's possible to have this two legged robot learn to walk,  which is really interesting that this is actually  possible,  to do with a policy gradient method, trained on a real robot, learning to walk. Then in the Generalized Advantage Estimation paper, we looked at some of these hyper-parameters, so the discount factor, gamma, and the exponential averaging of which estimate you use Monte Carlo versus capping it off
              
                  40:00
                  with a value functionbut at what time. We see that the optimum is somewhere here with a gamma of 0.98, and a Lambda of 0.96. So the optimum is not at the extremes. And, that's interesting because that means that you really want this variance reduction effect from using the right setting of Lambda and gamma.
              
                  40:22
                  All right. In a quick summary of this lecture, we looked at the policy gradient their derivation, it's a pretty mathematical derivation. One of the most mathematical derivations there is in modern deep reinforcement learning. We started from a simple derivation then introduced temporal structure and saw that you can take policy gradients by just looking at the grad log probability of actions given state, you don't need gradients of the dynamics.
              
                  40:46
                  You don't need gradients of the reward. Just gradients of the log probability of action given state, which is in your policy in your neural networks so directly accessible. We saw we want to reduce variance by subtracting a baseline. We saw multiple types of baselines, most intuitive, and most often used is a value function baseline.
              
                  41:05
                  So we then looked at value function estimation, the value of the current policy. We saw many ways to do that. Directly based on Monte Carlos estimates or bootstrapping. And then we also looked at the more sophisticated advantage estimations with A3C and GAE.
              


                英语 (自动生成)英语（美国）
                
                  00:00
                  Welcome to lecture four in the six lecture series on the foundations of deep reinforcement learning. It's a quick refresher where we at, we already covered what are MDPs and some exact solution methods. That was lecture one. We looked at deep Q-learning, which is one approach that can deal with larger scale problems.
              
                  00:22
                  Then we started looking at policy gradients and advantage estimation in lecture 3. And we actually ended up with a quite complete and good policy. Great. And Auburn at the end. Um, but now in lecture 4, we are gonna see some ways to make that even better. So quick recap of our setting. We have an agent interacting with an environment by taking actions in the environment, getting to observe the environment and based on that, take those actions.
              
                  00:54
                  And the goal for the agent is to optimize expected reward accumulated over time. In policy optimization  specifically, the agent will be represented by a policy often under the hood, a neural network. And so we're trying to change the parameters of the neural network, the parameters of the policy that parameter vector theta, such that we hopefully find the choice of those parameters, that results in collecting high reward in the environment.
              
                  01:23
                  So more formally we're solving an optimization problem. We're trying to max over all choices of theta, the expected sum of rewards when using a policy encoded by that parameter vector theta. And it's a stochastic policy class in that the policy outputs a distribution over actions for each state.
              
                  01:41
                  And we've talked about this before, but it's a way to smooth out the optimization problem. And it actually even allows us to solve problems where the reward itself is non differentiable. We can still get gradients on this objective. Here is the vanilla policy gradient algorithm, the baseline system that we'll start from in this lecture.
              
                  02:02
                  So we'll initialize our policy by choosing some parameter vector theta. Our baseline is very often also neural network, we also initialize that and it'll often represent the value of function and we iterate, in each iteration we run our current policy to collect a set of trajectories. Then at each time step in each trajectory we compute the return from that time onwards.
              
                  02:30
                  We compute an advantage estimate, which is the difference between the return from that time onwards and our baseline, which typically would be our value  function estimate. How much do we expect from this state? How much did we get? The advantage shows us what if we did better or worse than expected and how much so.
              
                  02:47
                  Then we're going to refit the neural network that represents the baseline by fitting it to the rewards to go. This could be done directly using this monte-Carlo estimate of the return, or you could also use some bootstrapping as we saw in the previous lecture. And then what can update the policy using the policy gradient by taking a step in the direction of the grad log  probability action given state times the advantage.
              
                  03:13
                  So an action that resulted in higher than the average reward, meaning a positive advantage, will increase its probability through this and action that was below average will decrease its probability through this. And then we go back, collect more trajectories, keep repeating, until we reach hopefully a good optimum.
              
                  03:35
                  So in this lecture, we're going to look at some improvements we can make to that. At the end of last second, we actually saw some improvements in variance of estimating the advantage, which remains very important. We're going to look at some additional complimentary improvements here. One is we'll look at something called a surrogate loss, and then we'll look at step sizing cause a gradient tells you which direction to go, but how far? And actually we'll look at some  higher order optimization methods using
              
                  04:03
                  trust regions, that can have better step directions even, and lead to more stable optimization through trust region policy optimization. but higher order also tends to be often difficult to run in neural networks, which have very large numbers of parameters. And so then we'll look at an improvement upon that in the sense that it is able to do a first order approximation to the TRPO ideas.
              
                  04:29
                  And that last method, PPO is these days, maybe the most popular RL method out there. Okay. Let's rederive the policy gradient equation, starting from something called importance sampling. So we have our objective. Utility here, function of theta is the expected rewards when using a policy pi theta. That expectation, what's happening is a multiply and divide by P tau.
              
                  05:02
                  Given theta old, the divide is happening here and the multiply is happening by taking the expectation with respect to theta old, to multiply divide, it will cancel, and we really have an expectation, respect to theta. Why did we do this? Multiply and divide by P tau under theta old. Well, now when we compute our gradient what we see here is that we have gradient with respect to theta.
              
                  05:34
                  We have an expectation with respect to theta old, we have this ratio here. And the beauty is that we can collect data from our old policy, then see which direction we should improve theta. No matter what theta we're at this equation holds true. So if our current theta is equal to theta old, then this reduces to a standard policy gradient where we're able to now take a step in the policy gradient direction, but this all applies for any theta, no matter how close or far it is from theta old.
              
                  06:07
                  Now, of course, when it's close to theta old, the policy we used to collect the data, this kind of estimate will be much more efficient from a small amount of data, we can get a precise estimate. If  our theta that we use here is very far away from the theta old, where we collected the data, that will be a very high variance kind of thing.
              
                  06:25
                  But anyway, through this derivation we see that we can get the original policy gradient estimate where we use  data collected at theta old to take a step from theta old in the gradient direction, but we can also see this as so much more general. Through this derivation, we see the same idea can be applied to any theta.
              
                  06:42
                  And in fact, we can keep looking at this loss up here. I want to say really what we're doing when we're doing policy gradients is we're taking steps. A step on this loss. It's like, we're doing a first order approximation down here that gradient provides as a first order approximation to this alternative loss function here that maybe we could be optimizing by doing more than just a first order approximation.
              
                  07:08
                  And that's what we're going to be doing here. The thing at the top here, we're going to call our surrogate loss and we're going to be able to do more with it than just taking a gradient. So we are gonna keep that loss around, keep that in mind. But we're going to do more also with our step sizing.
              
                  07:23
                  So step sizing is necessary as green is only first order approximation and gives you a direction that locally is good, but that doesn't mean you should step infinitely far in that direction. Locally it's good. So how far should you go? Let's start with supervised learning. In supervised learning, if you have a bad step size, oh, okay, just the next update will correct for it.
              
                  07:43
                  The data is waiting for you to give you a correction. But in reinforcement learning, if you have a bad step size, you have a terrible policy. This terrible policy will give you terrible data and the terrible data might not have any signal at all in it. And now you don't get a correction. And what now, I guess you can just reset or something, but essentially all your learning might be lost.
              
                  08:05
                  Erased because your new data, it's just not informative anymore. So that's a problem and it's not so clear how to recover from that, unless you just shrink step size. But then you spend all this time collecting data on a bad policy and you're sharing the step size and try again. So is there any way to right away, maybe have a good step size? So we continue to collect good data and be able to keep improving our policy.
              
                  08:31
                  Simple step sizing would be you do a line search in the direction of the gradient. It's simple, conceptually, at least, but it's also a bit expensive because you have to evaluate, you have to say, okay, if I take a step size, this large, let's do a few rollouts,  let's see how well it does.
              
                  08:45
                  A bigger step size, how well does it do, a smaller step size,  how well does it do? Then finally you pick one and you do another policy gradient update. It's also a bit naive because it doesn't really include any additional information about your approximation you're making with your first order approximation.
              
                  09:02
                  And as I mentioned, we now have this kind of idea of a surrogate laws, other ideas that we'll bring into play, that can help us. So here is what we can do instead of taking gradient steps. And this is trust region policy optimization. We're going to use our surrogate loss, remember, which had the expectation with respect to the old policy.
              
                  09:20
                  So we've collected data under our old policy from the previous iteration . Then instead of us just computing a gradient from it, we set up a objective, a loss function with the ratio of the new policy over the old policy. So by changing the new policy, this loss here will change. We can run on optimization on this without collecting any new data.
              
                  09:42
                  The data has been collected on the old policy, and we're just changing this new policy in this objective here. And then of course we have here the advantage, which we estimated based on data collected from the old policy. Now, of course, because this is estimated based on the old policy, we need to be careful as we optimize this, we don't want to run too far away from the old policy.
              
                  10:02
                  Because then this term here, the old policy advantage will probably not be very precise anymore, but we can definitely do more than just a first order update on this. We can do multiple gradient steps on this to do better than just a single policy gradient. Then in addition, what we can do is we can say, hey, as we do the steps, we know this objective, this surrogate loss is only very good whenever we are somewhat close to the old policy.
              
                  10:28
                  So we should measure our distance from the old policy, the one that collected the data and make sure we stay close. So now we have a constrained optimization problem. We say let's optimize the surrogate loss while staying within a reasonable distance from the data collection, policy pi old. And then once we've done that we have a new policy we collect new data and repeat.
              
                  10:46
                  So that's the full algorithm. Run our policy collect data. We estimate the advantage function based on this data. Well, in this case, we don't compute an exact policy gradient, we set up the surrogate loss and then our deep learning framework will take care of computing the gradients on this. In some implementation, you'll use conjugate gradient to deal with this thing here, cause this could give you higher order things.
              
                  11:06
                  But let's not worry about that for now. Let's think about the high level. We solve now, a constrained optimization problem. And within this KL region, we find a new policy, collect new data and repeat. And ,then of course, a specific instantiation of this would be to do a first order approximation of the surrogate loss, and do conjugate gradient based on the second order approximation of the constraint.
              
                  11:29
                  And that would be a specific instantiation of TRPO, that actually is quite popular. But there are others too, that are simpler by directly optimizing the surrogate loss and directly using the KL. So to evaluate the KL, remember that when we're looking at the distance between distribution over trajectories, we have this product of an action given state and next state given state and action.
              
                  11:53
                  So when look at a KL between two distributions over trajectories between two policies, I'm going to expand this and we actually get a cancellation again of the dynamics here, simplification. We still have the dynamics up front there, but that's just an expectation. That means we just sample from the current policy.
              
                  12:11
                  And so we average based on samples from the current policy, the log probability ratio between the old policy and the new policy. And so once you do all this, here is some results. This is something I actually showed earlier in the overview of Deep RL progress in the last several years. These results that I showed then were obtained with TRPO, which is the algorithm we just talked about.
              
                  12:36
                  And it can solve a wide range of simulated robotics tasks, um, Hopper, Walker, Swimmer. And here are some learning curves . Okay. Let's look at the there's two versions of TRPO here. There's a vine version and single path, there were two versions then blue and the green, and they do  pretty well.
              
                  12:54
                  Some others are competitive though, but then for a harder problem, like swimmer, the blue and green have only one competitor that's close to them. Go to even harder problems, hopper and walker, the blue and green are clearly better than the prior works. And that's the two versions of TRPO. And also worked on a Atari games which is kind of interesting, it's a very general approach.
              
                  13:12
                  You can use policy gradient methods, uh, even though they're often investigated in the context of simulated robotics settings, can use them on games just as well. And then combined with generalized advantage estimation, which we covered in the previous lecture, you can get this kind of result here, a robot in 3d learning to run through its own trial and error learning, under the hood it's running TRPO and Generalized Advantage Estimation.
              
                  13:37
                  And that was the first way to get this to work now about five years ago at this point but at the time was the first robust method to achieve these kinds of results. And then of course, um, you know, in this case after 2000 iterations converged it's running very, very fast, but then you could actually run it on other environments.
              
                  13:59
                  For example, you could say, hey how about running it on a, maybe, four legged robot and then the neural network needs to be retrained, but over time it figures out how to control this four legged robot. In this case, you'll see actually it learns to run really, really fast, um, faster than might be realistic, but that's, I mean, it's just taking advantage of the simulator and trying to maximize reward in the simulator.
              
                  14:22
                  So lesson learned from this run might then be, oh, maybe the simulator needs to be adjusted a little bit. Um, if you want it, then also apply this policy in the real world. And then, um, you can also do it to learn, to get up. So humanoid here starts on the ground and through its own trial and error using TrPO and GAE learns to get up.
              
                  14:52
                  All right. Now the thing with TRPO, it captures, I would say a lot of the intuitions that we want. It has a surrogate loss so you can do multiple updates on the loss. You don't just have a first order approximation based on your latest data. It has this KL  that allows it to stay close to the policy that collected the data, which makes sure that your objective stays sufficiently accurate.
              
                  15:20
                  But what it also has is the fact that to then deal with this KL, you kind of end up with this,  well, the way it was done in the TRPO paper, a second order optimization that you need to deal with . And, and so the question that was asked at the time by John Shullman first author on the TRPO paper, and also the first author on the next pair from I'm going to describe PPO and which is currently, I would say the most popular RL algorithm was, is it possible to invent a version of TRPO that doesn't have this second order aspect to it, where everything is
              
                  15:56
                  first order, which makes it easier to use existing, deep learning frameworks, scales likely better to larger neural networks. Okay. So, other things that I say were on John's mind at the time where things like networks that have stochasticity like dropout was kind of difficult to deal with TRPO setup and same for parameter sharing in TRPO.
              
                  16:19
                  How do you do parameter sharing between policy and value function? It's not so clear with that trust region. So then not to mention that the conjugate gradient implementation is complex and it doesn't harness the existing optimizers that we have in pytorch,  TensorFlow and so forth. So TRPO on the left.
              
                  16:40
                  We have a surrogate loss. We have a KL. On the right, we say, hey what if we move the constraint into the objective with a weighting factor? I mean, in constrained optimization, that's done very often. And it's actually known that for the right choice of beta, of course, the right choice is it's very hard to know what that would be, at least ahead of time.
              
                  17:02
                  But there is a choice of beta that will make both of these problems equivalent and that they will have the same solution. Okay. So that's interesting. So, and now we don't have a constraint problem. We have just an optimization problem. Once they have just an optimization problem, we can run gradient descent, or SGD, or RMSProp, or Adam on this optimization problem.
              
                  17:25
                  Do a few steps and then our policy will collect more data and repeat. And so that makes everything a lot simpler. So I run a policy. Estimate the advantages. Do SGD on the above objective, or maybe RMS prop or Adam or something. You can then measure the KL. And if your KL is kind of close to your Delta .
              
                  17:45
                   If your KL is close to Delta, well, then you're good. But if you're KL tends to be quite a bit larger than Delta, you want to crank up your beta to pay more attention to the KL. And if your kale is much smaller than Delta, then you can decrease your beta a little bit. So in the next optimization round, it doesn't pay as much attention to the KL as it did before.
              
                  18:04
                  So  might sound very heuristic,  but there is actually a formal dual descent procedure that says that this is the right thing to do. This is a very natural way of turning TRPO into a unconstrained optimization problem where we can use a standard off the shelf optimization methods now, with a little dual descent update in the mix too.
              
                  18:29
                  This captures a lot of the intuition that you might naturally want to capture. It turns out that this is just PPO v1. It's not the one that's most popular. This has been simplified a little bit. So how was that simplified? Look at that ratio and the objective, right. Um, so let's go back for a moment.
              
                  18:50
                  We have the ratio and the objective, and we'll give that a name because we'll work with that in a bit more detail. And there's this other term here that is really trying to ensure that this ratio is valid that you don't start using this ratio here in your optimization, when you know the advantage, that's multiplied with it, is just invalid and you don't want to be there.
              
                  19:09
                  So can we simplify this further by just looking at that ratio and thinking through it  carefully. So that ratio, if the policie is the old policy, the ratio is 1. And as you start deviating from the old policy it'll go above one or below one, depending on which direction you, you start deviating for that action and state.
              
                  19:28
                  And so the V2 of proximal policy optimization says, hey let's directly do the kind of trust region aspects in some sense in the objective. And so it says we're going to go do some clipping. Originally there is just ratio times advantage, but here it says the ratio should stay between one minus Epsilon, one plus Epsilon.
              
                  19:53
                  Then I have another thing which is also look at the original one. And what I'm comparing is there is a clipped version that keeps the ratio within certain bounds and there's the original, and I'm going to be pessimistic about it. I'm going to only trust the most pessimistic, one of the two.
              
                  20:11
                  And so it's very interesting because what's happening here, this is the original, and I'm going to say, well, I'm not always going to follow that cause I'm not going to trust it when this other thing is more pessimistic. And what this is doing, it's saying that if my ratio goes out of bounds out of the one minus Epson or one plus Epsilon, well, once we go out of bounds, changing theta will have no effect anymore.
              
                  20:35
                  Right. And so I can change my theta to reach a certain one plus, one minus Epsilon, but beyond that, it starts stops having effect, and I can't influence the optimization with that specific state action pair. And so that's really what this is doing. It's saying that as you're, let's go back to the original objective.
              
                  20:54
                  As we look at this objective here, each of these terms, What are they trying to do? For every term where the advantage is positive is going to try to push the probability up every term where the advantage is negative. It's going to try to push this down. And what it's saying is that for any single term, if you push this ratio beyond one plus Epsilon, your objective cannot be optimized anymore for that term.
              
                  21:19
                  So you can only have that much influence based on one term in your objective. Similarly, when the advantage was negative, once you pushed your policy such that this ratio has become one minus Epsilon. This term can have no more influence. And so you're bounding the influence of every individual term.
              
                  21:35
                  In the process, you're also saying the policy, moving it further than one minus epsilon, one plus Epsilon can have no effect. There's nothing to be gained from that. And so it's a different way of defining a trust region, which is directly doing it by looking at the objective and the advantages and bounding, how far you can go for every single term here.
              
                  21:55
                  And the math of it is also simpler. If you look at this, it's just. Oh, some clipping and a min. And so you end up with effectively a simpler implementation than still having that KL  to deal with . And once you do this, this has become one of the most popular approaches today for RL. So let's go back for a moment.
              
                  22:12
                  This is going to be this clip loss is what you optimize as sum of those terms. And here's some examples of things done with it. . Two humanoids learning to play soccer was trained with PPO. They both are trained with PPO, try to beat each other. So it's a, it's a game they're playing against each other.
              
                  22:30
                  Then OpenAI's Dota bots were trained with PPO. And it's a sign for how scalable this approach. Very easy to scale up because this was trained on a massive amount of game experience. The Rubik's cube manipulation that we saw in a highlight in the first lecture was also trained this way. So let's see if this wants to play, uh, well, I'll scroll through it.
              
                  22:55
                  So this robot hand is executing a policy that was trained a hundred percent in simulation, uh, on a ton and ton of simulation, uh, which interestingly transferred over to the real world. Thanks to setting the simulation up in an interesting domain randomization way. But for the purpose of what we're covering here, what's interesting is that this was done with PPO.
              
                  23:16
                  So proximal policy optimization was used to train the policy in simulation that then was deployed on the real robot. And so I guess this is not the Rubik's cube is in hand manipulation. In the first lecture we saw the Rubik's cube. Oh. And here is the Rubik's cube itself was also achievedwith  PPO training in simulation, and then it figures out how to solve the Rubik's cube.
              
                  23:42
                  So, um, let's see, what did we cover? We looked at this notion of surrogate loss, where we looked at effectively importance sampling as a way to reinterpret what policy gradients really mean. Based on rollouts under the current policy, what can you evaluate the other policies based on this important sampling surrogate loss.
              
                  24:04
                  This allows us to get more than just a local gradient. We now have an objective. this objective can then be used within a certain region where we can trust it, which means close to where the policy that was used for data collection is. We can define a trust region with a KL  as was done in TRPO. Or we can actually do it by having a clipping in the objective.
              
                  24:27
                  So the surrogate loss gets clipped. So any single term, whenever the policy would run too far away from where the data was collected, that term can not contribute anymore to further optimize the objective. And that's, what's done in PPO and gives an objective, that's convenient to optimize with first order methods, including existing implementations of SGD and RMS prop and Adam in existing deep learning frameworks.
              
                  24:50
                  And so this is actually today, probably the most popular RL algorithm, especially when data collection is very efficient because this really optimizes in many ways for wall clock time. Uh, in case of efficient data collection. Whereas if you want maximum sample efficiency, then often you want to do a bit more off policy processing of the data.
              
                  25:12
                  We saw some of that in deep Q-learning in lecture two, and  we'll see some more off policy methods in the next lecture.
              



                英语 (自动生成)英语（美国）
                
                  00:01
                  Welcome to lecture five of this, uh, six lecture series on the foundations of deep reinforcement learning. So what have we covered so far in previous lectures? We've looked at what are MDPs and exact solution methods, which applied to small, uh, MDPS, but not large ones that we typically want to solve.
              
                  00:21
                  We looked at deep Q learning which can solve larger MDPs, um,a off policy method,which makes it quite data efficient. It can also introduce the instabilities at time, which can be, uh, a downside in terms of the amount of tuning involved. But these days people can get it to work really well on a wide range of problems.
              
                  00:40
                  We looked at policy gradients as well as the latest incarnations thereof in the form of TRPO and PPO with PPO probably the most widely used RL algorithm today which directly optimize a policy. And by directly optimizing the policy, they're actually a lot more stable, easier to debug because you're always getting data from the latest, best policy.
              
                  01:00
                  And this keeps monotonically or close to monotonically if it's run well, improving over time. But the downside of these on policy methods is that they tend to be not as sample efficient. Would you care about sample efficiency? Depends on your problem. Some problems will, all you care about is compute efficiency and if your data can be collected really quickly because you're running in a very fast simulator, sample efficiency might not be the bottleneck, compute efficiency might be your bottleneck.
              
                  01:29
                  But if you do care about sample efficiency, then often you might have a preference for the methods we'll see in this lecture here DDPG and SAC over PPO,  because they will reuse the data that's collected from the past more. They'll reuse it more often. So do more gradient updates per data collected, which allows the neural network to extract more information from the data collected and hence learn more from less data very often.
              
                  01:57
                  And so if you look at learning curves, often you see on a horizontal axis sample complexity, for vertical axis performance, and you'd see that often SAC and DDPG will have a very good sample complexity. And so we'll take a look at those now. So we'll start with DDPG and then we'll go to SAC.
              
                  02:15
                  At a high level, you can think of SAC as the maximum entropy version of DDPG. In fact, at every level you can think of it as the maximum entropy version of the DDPG. So let's start with DDPG. How does it work? You get rollouts on the current policy, plus maybe some noise to make sure there's exploration, if your policy is, not naturally stochastic or something.
              
                  02:40
                  Then there's a Q function update. So based on the roll-outs you have estimates of your Q function, and then you do a update on this. We've seen this before what this could be. This target here could be a reward plus gamma times Q at the next state, or you could use reward at the current transition plus reward at the next plus Q after that.
              
                  03:03
                  We saw this in A3C and Generalized Advantage Estimation, there are many variants of, to which extent you take the Monte Carlo rollout signal versus the bootstrap signal. In the original DDPG paper, they use the one step bootstrap signal. Variants have been done since that are often still called DPG, where you would use multiple steps of rewards followed by the Q function as your target .
              
                  03:31
                   So we're doing Q learning, but we're doing it based on data collected from the current policy or from a recent policy. And then we update the policy. So in regular Q learning, you just keep track of the Q function, but here we also have a policy and the policy is optimized as follows. You can look at the Q function at each state that we encountered, and then there's an action we can take.
              
                  03:56
                  This action will be chosen by the policy. We want to optimize the policy, such that if we apply the policy at the states where we've collected samples, then the Q function will predict that we'll achieve a higher value. So what is this saying? It's saying optimize your policy to shift the weight or shift the actions towards actions that have high Q values.
              
                  04:24
                  And by the way, unlike the previous policy gradient methods. Uh, you know, standard policy gradient as well as PPO and TRPO, which rely on the likelihood ratio policy gradient. This policy gradient here goes through to Q function. And by going through to Q function here, actually, if you want to, your policy could be a deterministic policy.
              
                  04:47
                  Of course, for your data collection, you might still want some stochasticity. And that's why it says here, maybe plus some noise, but you can have a deterministic policy. And that's why it's called deep deterministic policy gradient. It doesn't have to be a deterministic policy, but it can be if you want it to be.
              
                  05:03
                  Okay. And then you repeat so more rollouts, use the data to further improve your Q function estimate, and then update your policy  such that it maximizes  Q value at the states in the replay buffer. There are a couple of extra things, if, especially if your policies deterministic,  you want to add noise to ensure there's exploration.
              
                  05:23
                  Replay a buffer and target network ideas from DQN increase stability. And often people use some lag or Polyak averaging version of  and PI theta for the target values of Q hat. So many of the target values you use this just like in DQN you use an older version of your Q function to stabilize things. So it doesn't like hop up around too quickly.
              
                  05:48
                  Once you do this, it can get actually really nice results. For example, in simulated robotics environments this is from the original DDPG paper. Let's see here, a Reacher, uh, legged robot and then the Reacher, uh, Knocks a ball up. And so there's a written deception, also done from pixels. So image inputs able to learn a control policy this way.
              
                  06:11
                  So, and even a racing game was trained this way. So very, very interesting that this is all possible with a policy gradient method. And it'll be more data efficient than a regular policy gradient method. So it's nice, very sample efficient, thanks to off policy updates. The downside of DDPG traditionally has been that it can be a bit unstable and that's where soft actor critic has come in and has in many places become the method of choice.
              
                  06:41
                  It stabilizes things by adding entropy in the objective. So it's going to be a max ent formulation, and this will ensure a better exploration and less overfitting of the policy. Of course, need to make sure that entropy doesn't decay too quickly. Otherwise you don't get that exploration, of course.
              
                  06:59
                  And then the entropy, when I say less overfitting on the policy to any quirks of the Q function, because Q function favors a specific action in the DDPG and policy might heavily favor that action, but maybe the Q function is still noisy and maybe by having a max ENT in the objective, you'll have a more spread out policy that doesn't seek out the peak so highly on that specific action the Q function currently thinks is best.
              
                  07:24
                  Okay. So what does it look like? It'll use a soft policy evaluation. So if you look at a Q function, the target will be reward plus expected future rewards summarized in the Q function. In this first part,  has a standard target, but then there will be an additional entropy term here that naturally pops us as we saw in the first lecture when you look at maximum entropy reinforcement learning.
              
                  07:49
                  We have a reward in the objective plus beta times entropy you're a beta is one. Here beta is one, that's why there is no factor in front. Your Q targets are now adjusted to be also account for the max ent, then updating the policy through information projection, meaning when you're, again, due to thing like we saw in DDPG, optimize your policy to maximize your Q value, you have to account for the fact that you actually now want a max ent policy, taht optimizes Q value, which we know means a policy that is effectively the exponentiated Q values.
              
                  08:23
                  And so it's done by saying we're going to minimize the KL divergence between the policy and the exponentiated Q values encoded policy on our samples. And repeat until convergence. Soft Actor Critice, of course, these things are not exactly optimized this is a iterative optimization. We'll take one gradient step or a couple of gradient steps here again here on this KL objective, just a few gradient steps.
              
                  08:46
                  And then of course we keep repeating. So what does this look like as a whole? Our objective is now a max-ent objective. So if you want to keep track, how well is my agent doing? You don't want to just plot reward, if, if you want to know what your optimization is succeeding, you might care about reward.
              
                  09:02
                  If you want to know what your optimization is doing, what it's expected to be doing, you want to plot rewards plus, uh, entropy, and then iterate. The value of function estimate that's this thing here. The value  target is based on Q and entropy is introducing the entropy here through the value function neural network.
              
                  09:24
                  Then the new Q target is reward plus expected value at the next time. And then the policy is based on the KL to the policy, defined by exponentiated Q values. Okay, once you do this, here are some learning curves for humanoid, one of the harder simulated control problems. And we see soft actor critic doing really well.
              
                  09:45
                  DDPG is here. What does that mean? It probably means either its exploration wasn't very good. Or when it had some good exploration, it wasn't able to extract the signal from it. With more tweaking, probably you can make this do better, but the beauty here, when you read the soft actor critic paper, which actually, I'm one of the authors of, is that, it's not so sensitive to the hyper parameter settings and it consistently tends to work quite well.
              
                  10:08
                  Even across multiple runs, you can often see that even the worst run does very well which is a really nice property. Then here a few other environments, again in yellow, soft actor critic, consistently having really strong learning curves, uh, compared to the other methods. And that's also why it has become, very often that method of choice.
              
                  10:31
                  Here's the video. So hopper learns to hop really well. Walker learns to run. Cheetah runs off the grid, um, ant runs off the grid. Humanoid learns to  walk in some interesting ways. Um, so what we see here is that all these canonical environments soft actor critic and do really well on. Um, it's even been used to train some real robots.
              
                  10:56
                  And robust performance there. Um, manipulation robots. And block, lego  block stacking. So, um, it's a very efficient, robust learning method and that's why it's been okay, amenable to running on real robots where of course data collection tends to be somewhat expensive and you want to, um, be fairly, uh, data efficient to make it practical.
              
                  11:38
                  Okay. So this was one of our shorter lectures. We covered two of the most common methods,  in RL today, um, DDPG and soft actor critic. Um, they compute policies ultimately, but at the same time are learning a Q function and the policy gets extracted from that Q function. Um, and that's in some sense, what characterize a actor critic method, they're both actor critic methods where the policy gets extracted fairly directly from, uh, the Q function that's being learned.
              


                英语 (自动生成)英语（美国）
                
                  00:01
                  Welcome to lecture six of this six lecture series on the foundations of deep RL. Lecture six so the final one, we already covered a lot. Um, we looked at the foundations rL, starting with exact solution methods. Then we looked at Q learning, policy gradients, more advanced versions of policy gradients with TRPO and PPO.
              
                  00:24
                  We looked at actor critic methods, DDPG and SAC. All of these methods that we've looked at so far, fall under this category called model free RL. Model free reinforcement learning means that when your agent collects the data, it's going to use that data to directly learn a Q function and/ or a policy.
              
                  00:45
                  Now there's something else you could in principle do with the data. An agent, use it to learn a model of the world. A dynamics modelthat allows me to simulate what the world is like. And when I have that world model, then maybe I can use it to then find a good policy or a good Q function by using that world model.
              
                  01:08
                  And by doing so I can do a lot of learning without needing to collect new data in the real environment. And so that way maybe you could be a lot more sample efficient is the thinking. And so that's what is model based RL. It's where you use the data to learn a model and then use that model to learn a better policy or Q function.
              
                  01:32
                  Typically you have to go back and then collect more data and keep improving your model and so forth. And we'll see soon how it exactly looks. But that's what we're going to talk about in this lecture here. So we'll start with the kind of basic framework of model based RL. Then I would say that model-based RL is maybe not as converged as model free RL.
              
                  01:52
                  Model free RL, I would say it's kind of pretty clear that the methods of choice these days tend to be either SAC, or DDPG, or, TD#, which is very related to those two, or PPO for a more pure policy gradient method, or something like rainbow DQN for DQN style method. In model based RL, it's still pretty, open-ended just a lot of variants at play.
              
                  02:17
                  So can't enumerate all of them. I'm going to give you two variants that are exemplary of the kind of things people are thinking about as they think about improving upon the standard model based RL approach.. But I would say it's still a lot in flux in how people do model based RL. So what is vanilla model-based RL? You iterate: your agent collects data under the current policy.
              
                  02:41
                  Then it learns a dynamics model from past data. Some people call it a world model. I usually call it a dynamics model. And then you improve the policy by using this dynamics model. Maybe by backtrack through time through the learned model, the policy and the rewards. Or by using the learned model as a simulator, and then you can run model free RL inside the learned simulator.
              
                  03:03
                  Both are fun. And so once you've done that, you want to say, are we done? Actually, we're not. We need to iterate because if we just optimize the policy in the simulator, it's going to be really good at doing well under that learned dynamics model, but that dynamics model is probably not perfect.
              
                  03:22
                  This is learned from a limited amount of data collected under the initial policy. And so the initial policy might not get that interesting data actually. And so now your new policy might be exploiting some things might be not as good as you hope for. You put it in the real environment, collect more data and repeat.
              
                  03:38
                  And so you'd go around this loop many, many times every time, updating your dynamics model, updating your policy. Okay, so why model-based RL? You might anticipate data efficiency, indeed. You get a model out of data, which might allow for more significant policy updates than just a policy gradient or what you get in a Q function.
              
                  03:59
                  Though, Q functions can extract a lot too, very close to model best methods in some ways. And by learning a model, what you learn could be reusable for other tasks. It's not specific to the reward you're learning with. Later somebody has a different reward. You can still use your model and optimize this new reward in your model.
              
                  04:21
                  So again, here's the algorithm. Iterate: collect data under the current policy, learn a dynamics model from past data, improve policy by using dynamics model, and repeat. So, again, an anticipated benefit sample efficiency. So why is it not used all the time? When you look at RL papers, actually model-based RL is, I would say, one of the least frequently used among RL approaches.
              
                  04:48
                  Why is that? Well, I think part of is that actually it's not as mature yet as it could be and there is more work to be done. Another reason could be that, if you have access to a simulator, why run things in a learned simulator, if you have a simulator anyway? Like why learn a simulator for Atari if you have the Atari simulator right there.
              
                  05:07
                  Or why learn a simulator for simulated robots, you already have the simulated robots. But there are some more subtle reasons beyond the fact that maybe when you have access to a simulator, that it's not the natural approach to use. Training and stability is one. And we'll look at one approach that can address some of this.
              
                  05:24
                  There are other approaches out there, but I just want to give you a flavor of what's possible. And often not achieving the same asymptotic performance as model free methods. And I'll give you a flavor of one method that is resolving that. So let's start with the robustifying, and then we'll look at the adaptive model based RL.
              
                  05:46
                  So robust model of best all through model ensemble TRPO. This is with TRPO because TRPO was state-of-the-art method when this was developed. If somebody were to redo this today, they would probably do a model ensemble PPO rather than model-ensemble TRPO, but ideas are the same. So we need to first understand over fitting the happens in model based RL.
              
                  06:11
                  Standard over fitting that you have in supervised learning, your neural network performs well on the training data, but poorly on the test data. And this could be happening in all of this they're out to, if you learn a dynamics. Then you could overfit and learn a dynamics model that's overfit. So you need to avoid that and let's avoid that.
              
                  06:31
                  And we can regularize, we can have holdout data and so forth, avoid overfitting, but there's something else. There's a new kind of overfitting that pops up in model based RL. And this new type of overfitting is as follows: your policy optimization step tends to exploit regions where insufficient data is available to train the model.
              
                  06:52
                  And this leads to catastrophic failures. It's called model bias. And the proposed fix here is going to be model ensemble trust region policy optimization. So to give this a bit more color, you have your learned simulator, and you're optimizing your policy. In some regions the learned simulator is accurate.
              
                  07:10
                  In other regions it's not so accurate. Among the regions where it's not accurate, for some of them, it'll think you can get very high reward, even though in the real world you can not. But the inaccurate simulator thinks you can get very high reward. And so if you optimize your policy in that simulator, you'll be over-fitting to your simulator, which is guiding you to parts of space where things are not good because it's inaccurate.
              
                  07:36
                  And so that's the kind of over-fitting we're worried about here. And that's referred to by model bias, the learned dynamics model has some kind of biases that you over fit to that you want to avoid. So let's compare a vanilla model based RL top here with model ensemble trust region policy optimization.
              
                  07:56
                  So, the red block is where things are different. Collect samples, sure. Train all models. So we have an ensemble of dynamics models here, or world models some people like to call it. Then we repeat, you collect fictitious samples from these different models from the ensemble, and then you update the policy using TRPO or PPO, or maybe SAC or DDPG.
              
                  08:23
                  And then you estimate the performances and you keep doing this until you stop improving. And so what's interesting about this is that as you estimate performances, you'll ask my performance across multiple members of the ensemble. And as you train an ensemble of dynamics models, what happens is wherever there's enough data to support an accurate model, these models will typically agree.
              
                  08:51
                  But when there's not enough data to support an accurate model, because these models are initialized differently, they will make different predictions because there's been no data to say what should be predicted there. They will disagree. And thanks to that disagreement, you will know that you're outside of the region where the simulator is precise.
              
                  09:06
                  And that's being leveraged here to then say, we need to collect new data, improve our ensemble models, and keep repeating. Once you do this, you can actually get a very robust model based RL algorithm. And so we evaluated this, actually some of our own work, across a range of simulated robotic environments.
              
                  09:25
                  We look at the learning curves. In green is the approach I described. It's consistently shooting up quickly and very robust and very little variance on these curves. Whereas compared this here with other methods, let's say model free methods, PPO TRPO DDPG, they are a lot slower. And this is on a logarithmic scale.
              
                  09:43
                  So there's actually a factor of 10 to a hundred difference in sample efficiency, even more on some of them. Here we have a very robust way of doing model based RL. Now we did some ablations in that work. We can compare back progressions for time with standard policy gradients with TRPO in green, and we see that TRPO in green is always among the best in all three environments.
              
                  10:07
                  And there's some variance between the other two, how well they do. Backpropagation through time struggles a bit more, has higher variants. Why might that be? When you backpropagate through time, computing your gradients through the learned dynamics model, it just might introduce more of errands because that's a learned dynamics model than we have with repeated rollouts against the dynamics model, which doesn't need accurate derivatives of the dynamics model only expects accurate, or reasonably accurate rollout.
              
                  10:36
                  We also ablated the number of members in the ensemble. One member, of course, it's not really an ensemble, just one model that's in green, that doesn't do that well, but once you have five members, it does quite well. This plot suggests that 10 is probably enough for these types of environments.
              
                  10:51
                  Okay. So we now have a way to robustify model-based RL by learning ensemble of models and making sure that we don't get this kind of over-optimization against the one specific model, but only optimize against the ensemble as a whole, by which we optimize the policy against things that the data supports in our learned simulator, as opposed to potentially overfit to what works in the learned simulator.
              
                  11:16
                  But can we do more. We learned something very robust here that works across a range of simulators, but the real world is not a range of simulators. It's one specific thing. So you could wonder, can we learn something that is adaptive that can quickly adapt to the real world rather than something that is robust and maybe conservative as a consequence.
              
                  11:37
                  So here's again, model-based RL, we said, resolve training instability. But what about the asymptotic performance? By being conservative we might give up some asymptotic performance. So can we add some adaptiveness to all of this? One fix could be just on a better dynamics model, but the problem with that is that it's very hard to learn really good dynamics model.
              
                  11:57
                  Another way to do it would be. Do model based RL via meta-policy optimization. We haven't really covered that in this six lecture series, but what it means is to learn a policy that is very good at adapting quickly to a new environment that's related to the environment it's trained in. So you learn an ensemble of models still, but instead of trying to learn a single policy that works well across all of them, you try to learn, a adaptive policy.
              
                  12:22
                  When you're deployed in any one of the models, you can adapt very quickly and then hopefully you can also quickly adapt to the real world. Here's a pseudo code for that. You can look at that a little more slowly on your own time. But key here is that when we essentially, we, again, train an ensemble of models, and then for all models, we're going to be sampling trajectories, each model generates their own trajectories, we have this single policy PI theta, and we see the policy PI theta with one policy gradient step can do well on
              
                  13:00
                  all these different learned models. And if it can, that's a good policy pi theta, can also probably adapt quickly in the real world. And then we optimize theta to achieve such a policy. We evaluated this actually across a range of simulated environments. And we actually get really, really good performance.
              
                  13:20
                  On the left is the model based meta policy optimization on the right is model free. Model free, we know, is going to be less sample efficient. And we see the evidence here, after 20 minutes of experience, model free is still kind of the very initial struggles of learning a policy whereas the model based meta-policy optimization approach is already having a cheetah run quite well.
              
                  13:46
                  And then let's see, similarly for Walker on the left and the ant on the right. We're comparing MP-MPO with PPO. The model based approach is way more sample efficient, after less than an hour of experience already doing well. Whereas PPO is still in the early stages of learning. All right. So here are some learning curves showing the performance going up more quickly is better.
              
                  14:17
                  The red curve, it says "ours", which is the MB-MPO model based metal policy optimization. And then the other curves are model-free approaches and we see it learns 10 to a hundred times faster. And achieves the same asymptotic performance. This paper was the first one to show that with model based methods, we can truly achieve the same level of performance model free, by using this adaptive learning approach.
              
                  14:41
                  And then comparing with state-of-the-art model based methods, we see it outperforms the model ensemble TRPO, which we saw before, which is in blue and, also outperforms a model predictive control method in purple. And actually we're able to test this on real robots. Because it is model based, it's very sample efficient.
              
                  15:07
                  It's able to get very good performance also, and we see on the left here, PPO learned to Blockstack, oh, sorry on the right PPO. And on the left, the model-based meta policy optimization approach, in about 10, 20 minutes it can learn to stack that block, which is very, very fast. And so we see that this is a very effective method, even including capable of learning with real robot.
              
                  15:31
                  So quick summary of this lecture, which is also our final lecture. We covered the basics of model based RL. As I mentioned, model-based RL is less converged, I would say than model free RL,  maybe more opportunities for researchers, what it means, but also more kind of uncertainly what is the necessarily the exact go-to algorithm when you're going to run some model based RL.
              
                  15:50
                  I gave you two examples of some pretty good choices. ME-TRPO and MB-MPO. ME-TRPO uses an ensemble of learned models to be more robust, both during learning in terms of final policy. MB-MPO uses an ensemble of models to learn, to be adaptive. And it's the first one to achieve, asymptotic performance with model-based RL that matches, the asymptotic performance of the model free methods in these simulated robotics environments.
              
                  16:20
                  So taking a step back, what did we cover? We looked at reinforcement learning, the very basics, what are MDPs, and what are some exact solution methods? We covered value iteration, policy iteration, we covered the max ent formulation of MDPs. We also realized that those tabular approaches to exact value iteration, exact policy iteration, can only work for small MDPs.
              
                  16:48
                  So if you want to solve more interesting things, we need approximate methods. We saw a range of approximate methods that can do quite well, actually. Deep Q learning is a off policy method that was behind the big breakthrough on Atari in 2013. Policy gradients, advantage estimation, TRPO, PPO were behind many of the breakthroughs in reinforcement learning for robotic control, including vision-based robotic control.
              
                  17:16
                  DDPG and soft active critic made that more efficient. They can do some off policy learning, being a bit more data efficient. You might still want the more pure policy gradient methods because they are simpler, and if you have fast simulation where the data collection is not the bottleneck, but the neural network updates are the bottleneck, then, maybe you don't care about reusing all data all that much.
              
                  17:38
                  And then we saw model-based RL, which generally is expected to be maximally sample efficient, but there's many more components to it and it's a little less converged as a whole, but the general principle is pretty clear. You collect data, you learn a model, train a policy against that model. That policy will be over fit to the model you learned.
              
                  17:55
                  You don't want to train it for too long. You go use that policy to collect more data, learn a better model and repeat. Okay, that's it for this series. Hope you were able to learn a lot about the foundations of deep RL from this and good luck with your own, uh, work and research on this. Thanks.
              
            