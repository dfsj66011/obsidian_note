
https://zhuanlan.zhihu.com/p/648924115


**Attention 层单步中间激活值显存表**

| 计算步骤                                  | 中间激活                   | 形状               | 占用显存             |
| ------------------------------------- | ---------------------- | ---------------- | ---------------- |
| $Q/K/V = xW_{Q/K/V}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $QK^T$                                | $Q, K$                 | $[b, a, s, h/a]$ | $2 \cdot 2bsh$   |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $\text{score} = \text{softmax}(QK^T)$ | $QK^T$                 | $[b, a, s, s]$   | $2bs^2a$         |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, a, s, s]$   | $bs^2a$          |
| $x = \text{score} \cdot V$            | $\text{score V}$       | $[b, a, s, s]$   | $2bs^2a + 2bsh$  |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $xW_o$                                | $x$                    | $[b, a, s, h/a]$ | $2bsh$           |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, s, h]$      | $bsh$            |
| $\text{layernorm()}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $\text{SUM}$                          |                        |                  | $13bsh + 5bs^2a$ |

**MLP 层单步中间激活值显存表**

| 计算步骤                          | 中间激活          | 形状            | 占用显存    |
|---------------------------------|-----------------|---------------|----------|
| $xW_{\text{up}}$              | $x$           | $[b, s, h]$ | $2bsh$  |
| $f_{\text{gelu}}(xW_{\text{up}})$ | $xW_{\text{up}}$ | $[b, s, 4h]$ | $8bsh$  |
| $xW_{\text{down}}$            | $x$           | $[b, s, 4h]$ | $8bsh$  |
| $\text{dropout}$              | $\text{dropout\_mask}$ | $[b, s, h] \text{ int}$ | $bsh$   |
| $\text{layernorm}$            | $x$           | $[b, s, h]$ | $2bsh$  |
| $\text{Sum}$                  |                 |               | $21bsh$ |



至少在模型训练上来讲 是可以接受的误差 所以基本上大家都会选择 把模型先 转成FP16的格式 那它的储存空间也会 砍一半 就是会训练得比较快 那第二个原因就是说 因为像大部分的GPU 尤其是NVIDIA的GPU 其实对于FP16的这个格式 是有做特别的加速 所以你算FP16 会比散热还要快 不少 这是第二个component 所以其实在训练的时候 会储存两份LM的位置 一份是它最原始 精确度最高的位置 另外一个是 专门在GPU上跑的时候 所训练的位置 那现在讲完了这两个 我们有LM的位置之后呢 就会有个input 那至于这个input的大小是多少 其实有时候你input有长有短 所以我们这边先
              
                  08:27
                  等一下再讨论inputs的长度 好,那你有inputs之后呢 你会跑backpropagation backpropagation会算出一个gradient 那这个gradient其实就是 和LM的位置是一样大的 因为一个位置就会有一个 它的gradient更新的方向 所以这个gradient跟位置会是一样大的 那这个gradient 会被送到 右上角的这个 atom的optimizer去做运算 那这个optimizer 运算的时候呢 它其实大小是 跟LM的位置是一样 因为atom这个optimizer的精神 就是说我们希望 每一个 LM模型里面的权重 都有一个它自己的learning rate 所以一个权重就会有 一个FP32的momentum 还有一个FP32的 variance去模拟一个 指数decay的一个learning rate 所以说可以看到这两个 都是32GB的原因 是因为每一个位置都会有一个
              
                  09:31
                  momentum跟variance去更新 所以这个atom optimizer的用途就是说 它会把以gradient为输入 然后再加上它目前 这个权重的momentum跟variance 去更新最原本32GB的这个 LM的位置 那这边基本上就是 扣除了input以外 我们会需要的所有 所有训练的时候会用到的component 那我这边 简单的帮大家加了一下 全部加起来大概会是128GB 所以这个数字其实 是超过了现在 几乎所有GPU的 储存的大小 那也是为什么我前面在 overview的时候讲说 如果你直接用PyTorch跑 是没办法训练的 也是因为这些component 在一般的类神经网路来说可能都很小 只是你放到LM里面 它就会变得非常的大 这样子 那其实 讲的这些还没讲完
              
                  10:34
                  因为这个input其实在 训练的时候 才是佔最多的 储存空间的部分 那所以第三个 我们就来讲input 这边的input我们讨论的是 activation 那activation其实在 之前机器学习的课程的时候 它其实就是叫做 每一个模型 一个模型的某一层 对于这个input 所产生出来的反应 就叫activations 所以每一层的output 都会有一个activations 那这个activations 为什么会有这个东西 就是因为我们在做 backpropagation的时候 我们会用那个偏微分 从后面一路算回前面 那这个时候就会需要每一个 每一层的output 还有loss去得到它的gradient 那 这边简单的举两个例子 就是说 假设我们有一个8B的模型 那这个8B的模型呢
              
                  11:38
                  它的输入可能有两种 第一种就是正常的输入很短 可能最多256个token 那如果大家问个简单的问题 譬如说今天天气怎么样 加上system prompt可能就只有 256个token 这是输入短的时候 可能会到 16k或甚至 Google的Gemini到200k 那就是它可能会有 reasoning data或是一些 多模态的输入的时候 它的输入就可能到很长很长 那这么长的输入 会有什么问题呢 它的问题就是说 因为我们知道模型的每一层都会存 activations 所以右边这个方格表示的就是 activations 8B的模型会有32层 那Activations就是每一层都有一个 就会有32层 那这个Activations 最佔空间的其实是Attention 因为我们知道Attention 它是两个Sequence 之间所算出来的一个 方形的矩阵 所以它在空间複杂度上面
              
                  12:42
                  会从原本的ON变成 N平方,那N就是你 输入的Sequence的长度 所以以一个 8B的模型来讲 每一层的 Activations里面的Attention 就佔了40G 那你简单的乘以32 其实就达到一个很不可思议大的一个 大小 那这个大小呢其实就是 你在训练很长的 输入的时候会碰到的问题 就是这个Activations 实在是太大了 那其实 一个很直观的想法就是 其实你并不需要在训练的时候 把这些activations都存下来 那下面就跟大家介绍一个名词 就是这个 Activation Recomputation 那其实它也叫做 Gradient Checkpointing 它的意思就是说 你在Forward Pass的时候 你不要把所有的Activation都存下来 你只存一些重要的就好 那你在ByWord需要用到的时候 再重新算一次就可以了 所以其实有用到
              
                  13:46
                  这个技巧的话呢 这个package 应该说 这个实作的演算法就是 它会只存必要的东西 因为你byword是从后面往前算 那你可以在需要的时候 再一层一层的去算出它的activation就好 这样你就不需要存到 这个1.35TV这个 这么夸张的数字 那当然这个recompute的过程 也会增加一些 computation的overhead 所以你在训练的时候会稍微慢 个一点点 所以如果大家在训练 譬如说transformer training的argument里面 可能会看到一个叫做gradient checkpoint 它其实就是在 节省activations的 计算训练的时候 所需要的储存空间 那还有 最后一个小细节 是batch size的部分 那我们都知道 大语言模型或甚至是 一般的neural network 它所依靠的就是 输入会算出一个gradient
              
                  14:50
                  那用这个gradient来更新 你的神经网路 那这个gradient呢 它不能是随便的gradient 它必须要是 非常稳定的告诉模型说 它的权重应该往哪边降低 所以如果这个batch size不够大的话 会造成说它的gradient 会非常的bias 所以基本上在训练的时候 通常一个batch的大小 都要有4到60个million 这么多个token 那举例来说 像DeepSeek的V3 就是它在训练的时候 的某一个阶段 它在训练32K的context的时候 它的batch size是 1920 那等于说它输入了1920个 32K的context之后 才会做一次optimizer的更新 那1920乘以32K 大概就是61M的token 所以还有一个challenge是说 这个batch size会需要很大很大 才能够符合你这个
              
                  15:53
                  才能对你的训练有帮助 所以 我们不可能再一次的训练 就把1920个32K的context都丢进去 所以这边有另外一个名词叫做 Gradient Accumulation 那这个Gradient Accumulation就是说 我们把原本的Global Size 就是上面这个1920 给切成若干个MiniBatch 那这个MiniBatch就是 Pu可以装得下一次 forward backward的数字 如果说这个mini batch size是16的话 那我们一次forward 就是把16个 输入给丢进去 算出一个gradient 那这个gradient我们先不拿去更新模型的参数 而是先保留 那保留多久呢? 保留120个forward backward之后 再一起拿去更新最后的global的模型 那这就是gradient accumulation 它可以让你在 你没有办法batch size开那么大的时候 也可以近乎于 大的batch size 所算出来的gradient 那这是一个 小细节
              
                  16:57
                  讲到这边其实 大语言模型训练的时候会遇到的困难 都讲完了 所以大家可以看一下这个图表 那这个图表 就是 他讲的是你训练的时候会需要多少 memory 这三张图分别是 8B、70B 405B的模型 那下面 横轴指的是你输入的长度 那 纵轴就是你所需要的 GPU的memory 那大家可以看到 说底下的这个 紫色、红色、绿色 这部分在所有context 长度来说 都是一样的 因为它就是你的parameter的大小 然后你的gradient的大小 你的optimizer的大小 然后会随着 你的context变长 而指数增长的呢 就是activations 可以看到它基本上就是指数型的成长 所以当你要训练8B的模型 的时候你需要克服的就是
              
                  18:02
                  你需要的memory这么多 那究竟你要怎么样 用那么小的一个 GPU去训练呢 这就是我们今天要讲的 问题 那这个talk我总共会分成 三个阶段 第一个阶段我会先讲 就是这个橘色的activation以外的部分 要怎么样优化 所以part1讲的是你要怎么优化parameter gradient还有optimizer的 state,这是part1 part2要讲的就是 这个activation,然后 part3才会讲quantization 那quantization基本上只是 for inference,所以training的部分 会需要用到的package会在 part1跟part2的时候就讲完 好,那我们就先来讲 第一个,就是 我们究竟要 我们有这么多的parameter gradient 还有optimizer,要怎么样装到 GPU里面呢 那这边讲的component就是 前面这边
              
                  19:04
                  前面这边讲的 就是1,2,3,4,5 因为input是算是 activation要处理的 就我们part1先 不考虑 那这五个component加起来的 这个128GB的话 一个GPU基本上是绝对装不下的 那既然一个GPU装不下 我们其实就可以用多个GPU来凑 所以意思就是说呢 如果说我们今天有四个GPU 那 刚刚所提到的这个东西 其实你就可以把它 把它切成小片小片的 装到GPU里面就可以了 就是说如果说你一个GPU只有 32G 那你其实可以把 这个128G切成四份 就装得下了 那这边的component 就会对应到刚刚左边 part1的图里面的component 这样 那究竟怎么样切 才会是一个好的切法 因为其实 它这个还会
              
                  20:07
                  有很多个需要考量的 譬如说你的GPU是 GPU有几张 那你的模型训练的过程 所以你在训练的时候 这些东西其实是需要仔细考虑 那Microsoft就 帮大家写好了一个package 叫做DeepSpeed 那这个DeepSpeed呢 它主要就是Zero的这个演算法 那Zero是Zero Redundancy Optimizer的意思 那这个Zero呢 它的演算法一共有三个Level Level越高省下来的记忆体就越多 那Zero 1指的是说 我们上面看到的这个图 这个GPU里面 它佔最多空间的 其实是这个Optimizer 所以第一个就先找它开刀 你就把它 你有几个GPU 你就把它切成几片 所以可以看到蓝色这一部分 我们把一个Optimizer 切成四片 那一个GPU里面装一点 所以这样就 有的GPU就会装得下 这样子
              
                  21:08
                  好 那 等于说 你的GPU 等于说你可能会需要用到 第一个Optimizer的时候呢 你就把它的资料 传到第二个 第三个 第四个GPU去 这样其实就是 在GPU上面的意思 只是它 可以看到 大家可以看到的话 就是 这边多了一个蓝色的箭头 那这个箭头指的就是 你把 一旦把东西给切小了之后呢 它就会多出一些 Transfer的 Overhead 不过大家也不用太担心 这个Transfer的Overhead 在GPU跟GPU之间 是非常快的 好 那Zero2的话 其实也很好理解 就是你最大块的切完了 就是找第二大块的去切 那第二大是什么呢 就是Gradient跟LM的位置 就是绿色的这个部分 你也把它切小 那可以看到一个GPU里面要装的东西
              
                  22:11
                  就基本上少了很多很多 那你就更有可能 把它全部都塞到一个小的GPU里面 那也可以看到说 这个箭头从 蓝色变橘色代表说它需要的 传输的东西又更多了 因为有些东西你GPU-1在算的时候 它在GPU-4里面 你就要把它传输过来 你才能计算 所以会多一些传输的overhead 然后0-3 就是全开的状态 你把Zero这个演算法 每一个level都开了的话 它就是把最后一个 LM的位置也切开 那基本上 这个就是 你把 LM的training可以切到最小 大概就是Zero3这个状态 那刚刚有提到这个 transfer的部分 在GPU跟GPU之间是非常快的 究竟有多快呢 如果说你用的是NVIDIA的GPU 它用的是一个NVLink的技术 那它基本上每秒可以传900GB 这个速度是非常快的
              
                  23:15
                  然后不只是单纯很快 在DeepSpeed这个package里面 Microsoft他们也有针对传输这个东西做一些schedule 所以基本上它会尽量 能先传好就先传好 不会delay到training 那其实 你这样子把 LM训练的时候 需要用到的memory都切一切之后呢 大家可以看 下面这个图表 来看说究竟我们节省了多少空间 那上面这个 这个图呢 一样是一个8B的模型 那这边DP等于8 指的是说它一共有8张卡 那其实就是上面的图 只是每一个component 把它切成4片 改成切成8片 那 最左边这个是切之前 如果你没有开zero的话 它其实 这个红色的虚线 是80GB 那它这边假设的是如果你有一个 H100那它是80GB的话
              
                  24:21
                  你8B的模型 就训练不了 但如果你开了zero1的话 可以看到就是 橘色的部分没有变 但是紫色啊红色绿色这边 应该说红色 那跟绿色 是没有变的 因为zero1只有切optimizer 所以是紫色的部分变成1% 那如果你连gradient都切的话 就是这个红色再变1% 所以就是1% 最后你可以知道说 80GB的H100里面 其实可以训练到16K长度的 8B的模型 如果你用的是zero3 这个演算法的话 那刚刚有提到说 因为你把模型切成小片 所以它要用到的这个东西 有可能在那个当下并不在 这个GPU里面 所以它会有一些transfer的overhead 那这个overhead的速度呢 基本上它会跟 很多东西有关係 譬如说你的GPU跟GPU之间 间隔的bandwidth 或是你的model size
              
                  25:24
                  或是一些硬体的设计有关 但基本上不会到很夸张的慢 就是说不会慢到10倍或以上 它就是一个 大家都可以接受的一个training的速度 而且基本上你有 用了Zero这个演算法的话呢 它就是让你从不能训练变可以训练 所以大家基本上都是 会用Zero这个方法去训练的 好 那讲完了 讲到目前 我们其实介绍了DeepSea的 一个部分就是Zero的这个演算法 那还有一个就是 如果说 你连这样子切 都还装不下的话 就你的GPU真的很小颗的情况下 其实你还可以用一个东西 叫做CPU的RAM来凑 那大家可以看到说这个图 其实CPU的RAM相较于 现在GPU的RAM都是 大概是10倍大的这个量级 所以说如果你今天 GPU真的很小颗 或是说你真的想要训练很大的模型 你还是可以把他一部分的 训练要用到的component 搬到CPU去存着
              
                  26:28
                  要用的时候再拿出来 那 所以说刚刚看到的这个图 它DeepSeed 提供了两种方法 一种叫做Optimizer的 offload 那简单来说就是把跟Optimizer有关的东西 搬到CPU上面 所以留在GPU上面 就只有Large Language Model的权重而已 剩下的所有东西都 放在CPU上面 要用到的时候再来搬 那第二个就是 甚至可以连Optimizer都放上去 那基本上你的GPU就可以很小 只要你CPU的RAM 够大的话呢 你就可以训练这个模型 但这个offload的方法 有一个缺点 所以导致它其实不太好用 的原因是因为 GPU跟CPU之间的传输速度 是很慢的 就比刚刚的900GB per second 还要再慢一个量级 所以你的训练会变得非常非常的慢 那究竟有多慢呢 我这边有稍微跑了一个小实验
              
                  27:32
                  那这个实验就是 如果大家在实验室或是一些课 可能会用到国网中心的资源 那国网中心上面 的主要的 GPU是 V100 最近有H100可以使用 但大部分之前做实验 都还是做在V100上面 所以如果大家有用国网中心的资源的话 可以参考这上面的数字 那 详细的script我有放在这个repo里面 所以大家如果想要自己试试看reproduce的话 可以参考这个 那首先我做了两组实验 一组是有offload的 就是刚刚说的 会搬到CPU的range上面 会跑得很慢的那一组 第二个是没有offload 那有offload的话呢 就是四张GPU的情况下 就是四张V100 那V100一张是32G 所以四张加起来其实就是128G 刚刚算的那个 8B的模型 其实是刚好 勉强可以装得下的那一种
              
                  28:36
                  那 不过要注意的是 我这边其实input开的很小 就是batch size只有1 然后长度最多隻有1、2 那这边这个表格呢 是DeepSpeed它有提供的一个工具 就是如果你今天有一个环境 然后你有一个模型想要训练 它可以帮你算出 你需要多少memory 那有offload的话 可以看到 我这边offload的就只有 optimizer 等于说我把 optimizer给放到 CPU上面 就是 上面的这个 第一个case 我把 模型的parameter留在上面 然后 其他的optimizer有关的东西 放去CPU 那这样的话 你的per GPU要的memory 其实只有5.7GB 然后 但是你的CPU要够大 等于说你一个CPU 里面要有179GB 这样 这是它预估的数字 所以跟实际的数字 其实是会差非常多的 那我实际上测出来呢 它一个GPU大概会佔
              
                  29:39
                  15GB的记忆体 然后一个step 大概会跑74秒 那74秒这个数字其实是很夸张的慢 因为这个输入是很短的 不过它一个forward backward还有update 就要跑74秒 那这个意思就是说 因为你把东西搬到CPU 然后国网中心 它的硬体其实比较 针对GPU做加速 它的GPU跟CPU之间传输是 比较慢一点所以才会是 74秒这个速度 那这个速度基本上就是 很夸张的慢 那同样的东西 如果我有8张GPU的话 故事就会不太一样 就是它per GPU 只要18.7G 就装得下了 所以32G的V100其实是装得下的 你就不用用到任何offload的技巧 它跑起来就会顺很多 那它一个step就会变7.3秒 那它的VRAM 虽然说它预估的是18.7 但实际上用到的
              
                  30:43
                  还会考量一些 譬如说PyTorch它自己会有一些 它需要保留的一些记忆体 以及你输入的长度也会有影响 所以最后它是在24到32之间跑 它会有点上到下下的 上到下下的话就是 因为我这边是有开Zero 所以它传来传去的时候 有时候GPU会 某一张会记忆体会比较多 但总之它不会OM 然后就是 训练的时候会比上面这个setting才要 快10倍 所以这边有大概有两个小结论 给大家 第一个小结论就是说 如果你想要训练的模型是8B的模型 你其实是可以fully fine-tune它的 在国网中心上面用8张V100 是可以做到的 那速度大概是7.3秒 这一个level 那第二个小结论就是 能不要用offload就不要用offload 因为它真的很慢 那为什么他们会开发这个东西 纯粹就是他们那时候想做一个 很无聊的实验
              
                  31:45
                  就是要怎么样用 16张V100去训练 175B的模型这样 所以他们为了做这个东西才开发 offload这个东西 但实际上在做实验的时候 是不好用的 好 那这个就是Part 1 Part 1的部分帮大家做个总结 就是Part 1我们介绍的是 DeepSpeed DeepSpeed这个东西 对于这一张图来说 它改进的是 橘色以外的地方 让它可以压到 一张GPU状态下的程度 这个是DeepSpeed做的事情 那刚刚所展示的case其实是8B的模型 所以对于大公司来讲 他们有的可能不是8张卡 是80张卡 那他们就可以训练10倍大的模型 只是把模型切成80顺 其实是可以装得下的 那Part 2要介绍的是 Activations 那我们可以知道这个 这个地方要解决的问题是说
              
                  32:50
                  因为Attention这个 这个算法 它会需要的记忆体空间是 n平方 那这实际上是一个 非常大的一个数字 尤其是在你的输入很长的时候 所以 要介绍的就是要怎么样解决 activation的问题 那在介绍以前呢 我们先讲一下这个问题通常会怎么解决 因为我们想要 改善的是 Attention这个Operation计算的效率 那Attention这个东西 在GPU上面跑 所以这个问题解决方法 就是我们重写 Attention实作的Code 让它跑快一点 那要怎么样重写呢 这边先跟大家介绍 Kernel这个概念 Kernel其实就是 跑在GPU上面的function的意思 所以如果你要写一个 可以跑在GPU上面的function 你可能会有一些选择 第一个是PyTorch PyTorch它其实就可以把Tensor 或是一些
              
                  33:52
                  常用的Operation放到GPU上面跑 是可以比CPU跑 还要快很多的 那第二个呢 我先介绍一下这边的排序 就是上面指的是 High Level的Language 它比较方便 但它跑的就相对没有那么快 而且 它没有办法让你很精细的 可以控制配置你的记忆体 以及一些 算的 算数的顺序 我下的是越low level的language 它跑起来会比较快 然后它也可以让你控制所有 所有运算过程中的detail 但它相对来讲 它的learning curve就比较难一点 所以 那讲回来这边的第二个 torch compile 它是PyTorch的一个function的decorator 那这个decorator就是 你把它加在你的 譬如说加在模型的forward的function上面 它就可以帮你 像是compile整个PyTorch
              
                  34:56
                  它计算的graph 这个graph因为你的function是确定的 所以它会针对这个graph 去做一些memory的 调度跟prefetch之类的 会让你的code跑得快很多 那基本上这是一个无痛 就可以让你的code跑得更快的方法 那再下来 这个叫做Triton Triton是OpenAI他们开发的一个 在Python里面写的 让你用Python 可以写 Kernel Function的一个介面 那它基本上就比PyTorch还要灵活很多 那再来最下面最后一个是CUDA CUDA就是NVIDIA他们开发的 C或C++的套件 那基本上最low level的GPU的计算 就是透过CUDA去实作的 那这边就是四种常用的 如果说你想要写kernel function 你可以用这边的这四种 你可以考虑用这边的这四种语言 然后 讲完了
              
                  35:59
                  kernel是什么东西之后 我们就可以在 kernel function这个东西 上面去实作更快的kernel function 来让attention的这个 运算算得更快 所以第一个就是flash attention 这个或许 大家都有听过flash attention这个东西 那它实际上要做到的就是 它要加速attention的计算 让它不仅可以算得更快 还用更少的记忆体 不过当然 要做到这件事情当然不是没有成本 它的成本就是它有些东西会放到 CPU的RAM上面 所以它是透过 把大部分的东西放在CPU上面 要用的时候再把它fetch到 GPU上面去算 才能让它train得更快 又用得更少的memory 那我会分两步来介绍 一个是它究竟 为什么可以train得更快 就是它为什么可以用比较少的记忆体 那第一个 speed的话在速度上面的部分 大家可以看到左边的这个图 这个图的来源就是
              
                  37:02
                  attention is all you need 那它讲的是scale到product attention这个运算 那这个运算的输入呢 就是在下面这边的 QKV的矩阵 然后输出就是一个attention的矩阵 那可以看到说 这个attention的运算 它会有矩阵成分 scale,然后softmax 然后再一次矩阵成法 四五个複杂的 function 所以flash attention这边做的事情就是 它重新实作一个kernel function 这个kernel function就叫做flash attention 那大家可以这样子理解 那这个kernel function 它的输入就是QKV,输出就是attention 它把这四五个东西 全部压成一个东西 那就可以让它算得比较快 因为它在这个kernel function里面 做了很多的优化 这样子 那记忆体的话相比 原本naive的 attention而言,它其实也可以降到 接近线性的 记忆体的空间 那这边再
              
                  38:06
                  介绍一个小的名词 就是FuseKernel 那这个FuseKernel,这张图的来源 其实就是FlashAttention,他们的GitHub 那这个的意思就是说 当你把这五个function 全部压成一个function的时候,它实际上 执行时间可以快很多了 还有一个小的 insight就是 其实我们可以想像说 attention算的就是矩阵的乘法 所以理论上矩阵乘法应该要花最多的时间 不过实际上 因为矩阵乘法多数的GPU 都有做加速,所以实际上它不会 真的算那么久,在attention里面算比较久的 居然是dropout softmax 或是mask之类的运算 所以flash attention就是做了一个新的kernel 那来去做算这个attention 这样的事情 就可以让它速度快很多 那第二个部分就是memory 那memory其实这张图要表示的 就是刚刚跟大家有提到的 把大部分的东西 放在CPU这样的技巧 所以白色底的指的是 存在CPU上面
              
                  39:10
                  那黄色底的是 放在GPU上面 所以基本上Q 然后这个QKV还有A 这些矩阵呢 在你的context到非常长的时候 会佔很多的记忆体 所以SizeAttention用的技巧就是 它在你要运算的时候 再把它搬到GPU上面去算 就可以了 那这个图表示的就是 这个意思而已 那讲完了原理 可以看一下它究竟有多快 那在Performance上面 这个是跑在H100 上面的实验 那这是 横轴指的是你输入有多长 然后纵轴指的是 你跑得多快 越高就是越快的意思 那可以看到蓝色的这个 Standard Attention 它会碰到的问题就是第一个它算得很慢 然后第二个就是它在16K 或以上的时候就会 甚至H100的都会 记忆体都会不够 所以如果你用了 Triton去实作的
              
                  40:14
                  Attention的话 它会快很多 就是绿色的这个部分 然后CUDNN就是 CUDA实作的 Kernel的话是红色的这个 那如果你用的是FlashAttention的 2或3,就它们有很多版本 它们一直在更新 现在最新的应该是FlashAttention 3这样,那3的话呢 它就可以在 不仅是Context很长的时候 可以顺利的计算 它也可以比,甚至比你用CUDA 去实作Naive的Attention 还要再快一点,这样 好,那以上就是 Part 2有关 FlashAttention的介绍 好,那 FlashAttention这个东西其实大家 可以知道就好 因为现在有名的一些 Package像是Transformer之类的 它基本上原生都会支持 FlashAttention,所以大家就知道说 这个东西,如果它跟你说 你没有安装的话,你就要去安装一下 因为会差很多 好,那 再来Part2要讲的就是
              
                  41:18
                  LagerKernel,那从这个名字 大家可以看到说 其实也是一种Kernel 那Kernel就是跑在 GPU上面的function,那所以LagerKernel 这个东西就是它写了 很多个kernel,那这个kernel 它要做的事情就是说呢 它把LM在 进行运算的时候,很常会用到 一些function,给包成一个 fuse的kernel,所以 然后它是用Triton去 实作的,等于说你把 你的PyTorch的code 的某些部分给抽掉 换成Triton的 kernel的话,它就可以让你的 LM算得更快 以外,也可以用更少的memory 那值得一提的是 这个套件其实是一个 台湾人写的这样 就是大家如果有听过 他的演讲就知道说他是 在美国的LinkedIn工作 只是他某一个 Open Source的project就说他想要做这个东西 然后后来做出来之后就 受大家的喜欢 那我觉得最神奇的一个地方就是 他用法非常的简单
              
                  42:23
                  他用法就只是 你更改 你载入LM的方法 其实就可以了 上面这个是大家通常用Transformer 会用这个AutoModel for Cosal LM的方法去载入你的模型 那如果你要用的是 Liger Kernel的话,你就把这个东西 换成Auto Liger Kernel for Cosal LM 那他基本上支援的模型 还算蛮多的,就是开源的 基本上他都已经帮你把 Triton的code给插入 到PyTorch里面 所以你就直接这样load的话 他就可以帮你把很多个运算都加速 那 他的performance的话,大家可以 参考这张图,那这也是他们的GitHub 接下来啦,就是 不管是速度啊,记忆体 都可以用得比较少怪,拍泡比较快 然后你也可以训练 你之前所训练不了的 比较长的context的模型 好 那以上其实就是 Part 2的介绍 总结来说,如果说你今天碰到的问题是 你想要训练很长的
              
                  43:26
                  context,譬如说你想要训练的是 reasoning的模型 或者是你想要训练的是 以影片的token 当作输入的话,你可能就会 碰到activation佔 太多记忆体的问题 那解决这个问题呢,通常就是 透过改写kernel去做到 然后常见的 package,一个是lagger kernel 一个是flash attention 这样,好,那 到目前为止呢 都讲完了 那可以知道说 这一张图里面所有的元件 我们都有相对应的package 可以去加速 所以下面 下面这张图 基本上每一个东西 我们都可以尽量把它往下压 就往下压,让它可以 想办法塞到我们的GPU里面 所以虽然说LM 它训练会需要很多的GPU的memory 不过也是有方法 可以让它 塞进去并且做训练 那这些工具也都是开源的 所以基本上大家用起来应该是
              
                  44:31
                  都会非常的方便 好 那最后 我还想花点时间讲的是 Quantization这个东西 那刚刚的 刚刚的图里面有提到说 有些东西的浮点数是用 32bit去存的 有些东西是用16bit去存的 所以实际上 我们也可以不要用32 甚至用更低一点的 精度去存 那Quantization这个东西 它其实做的就是一个Lossy Compression 也就是说这个蓝色的 Tensor你可以把它 压缩之后还原 它可能会变一个 它有点稍微不太一样 但还是很近似的一个Tensor 透过类似这样的技术 就可以让LM在 底下的这个 就是它储存大部分都可以用 中间这根比较短的 压缩过后的Tensor去 进行储存 需要计算的时候再把它还原回来就可以了 它会有一个叫做 Quantization的演算法 那基本上会有很多种不一样的
              
                  45:36
                  演算法 会让你可以把32bit的东西 压成8bit、4bit 甚至有的可以近似到 1bit这么小 那常见的演算法 我把它列在这边 但实际上真的是五花八门 很多种演算法都可以做到 quantization还有dequantization 所以我就大概放了一些 那中间这个最大的就是 大家在做 Colab的时候就是 就是用GGML family 这一个演算法 的quantization的模型 所以 大家用到的其实是这一颗模型 就是meta-llama-8b 这是default的一个模型 这个模型是什么意思呢 大家可以看这两个数字 讲说它是 LLaMA-3-8b 那Q8指的是说 它把里面的每一个tensor 都压到8个bit里面去储存 那这样的话 其实模型整个载到 记忆体里面的时候就会很 它的空间就会比较小
              
                  46:40
                  那如果是8个bit的话 你一个位词 用的是8个bit去存 总共其实就只需要8G的记忆体 就可以载入一个8B的模型 那所以 Colab上面的GPU 免费的话是T4 T4就是15GB 很可以装得下 那你就还有7G左右的 空间去存 你要用的输入 好 那最后做个总结 就是 今天这个talk主要就是 告诉大家说有哪些使用的package 可以去解决训练 大语言模型的时候可能会碰到一些问题 那同时如果 如果说你今天有 大于一张的GPU 可以使用的话 你要怎么最大化的利用好 这么多的GPU 那最后还有一些 就是推荐的Reading的话 主要是这两个,我可以大概介绍一下 上面这个呢,是 他们写的一个非常详细的 有关于你要怎么样
              
                  47:44
                  好好的用 GPU Cluster去计算的 一个网站,然后这个网页呢 非常的长 它写的 推荐的Reading时间是两到三天 所以大家可以 所以大家可以去看一下 然后如果你有手机看的话 可能就是它会载很久 你可能用电脑看 然后下面这个就是 其实这算是一个坑啦 就是我在做DeepSpeed实验的时候 它官方的文档我觉得很难看懂 因为它官方有一个它自己的API 我觉得很难用 所以如果大家想要用DeepSpeed 跑东西的话 比较推荐就是你直接用Transformer的 Trainer去载 然后你就看这个连结里面 它会告诉你怎么样把DeepSpeed打开 那实际上你要做的就是写一个JSON档 就可以了 你就可以把DeepSpeed的功能全部都打开 这样子 好,那我大概就讲到这边 谢谢大家 好,那
              
                  48:48
                  大家有问题想问的吗? 我知道 有留言 好,那我就 回答一下这个问题 这个问题 我觉得 为什么会那么慢 是因为他们储存的GPU的时候 他们储存的可能是用 高效的RAM去存 那CPU的话 它的Level就会 比较低一些 那等于说CPU它原生 用的RAM是希望RAM可以 越大越好,那比较大的RAM 相对来讲它的Overhead就是 传输的速度会比较慢 那GPU的RAM比较小 但是它传输的速度 还有存取的速度会比较快 它基本上是一个 跟你的RAM的大小 去做一个 就是你可能会需要Tradle 的一个
              
                  49:52
                  所以基本上是跟它硬体的设计 有关啦 这可能是硬体 要解决的问题 那软体这边可以用的就是 让它可以在需要的时候再去拿 这样子 希望有回答到你的问题 大家还有 问题要问的吗 如果你们要 打字的话你可以直接举手 对 就 Torch有一个 就是有点 套件 是在多到GPU上面 就有一个 但跟刚刚介绍的 差别 我刚刚介绍多个GPU的话 应该是Part 2的部分 那Part 1的部分 Part 1讲的是 DeepSpeed这个套件 所以刚刚跟刚刚同学 你讲的那个应该是 不衝突 它是两个可以一起用的东西 刚刚的那个Torch的Distribution
              
                  50:55
                  这个东西用的是说 你今天它把同一份Code放到四个GPU上面 所以四个GPU上面都是跑一样的东西 只是资料不一样 这是一种平行化的方法 然后DCP这边做的事情是 它不只跑一样的Code 它会把一个模型切成四份 所以它多处理的是 你模型在计算的时候 它会需要从别的GPU那边 拿一些参数过来算 所以DeepSpeed处理的是 多个GPU之间 你在切模型的时候需要 考虑的问题 但是Torch的distribution 它做的事情是把一份code放到 多个GPU上面跑一模一样的东西 所以实际上你在跑DeepSpeed的 script的时候,你下的指令 其实不是Python 然后Torch Distribution,然后TorchDistribution 让它可以,因为TorchDistribution 就是处理GPU之间的通讯 然后District只是额外处理切模型的部分 所以它其实是两个 可以一起搭在一起使用的东西 对,这样有回答到你的问题吗?
              
                  52:00
                  那我可以理解为就是 有点相较之下 比较不会传递到GPU的问题 然后这个比较会是GPU 对,它处理的就只是 GPU之间的通讯而已 就是你资料要怎么样传过去 然后什么时候要传回来 所以说 其实还有一个细节就是Distributed这个东西 它会有一个 算是Master的Node 那这个MasterNode就是负责 把所有GPU所计算的Gradient 全部整合在一起 更新模型那个东西 叫做MasterNode 所以具体来说就是 它在管理MasterNode 还有WorkerNode之类 之间的 资料传输 谢谢 好,大家有问题要问吗? 如果没有的话,其实大家也可以hold在slido,然后我之后会一一回覆之后呢,就可能放在NTU Core,一个大家会看得到的地方 好,那我们之后,我其实想问,帮大家问一个问题
              
                  53:12
                  那个D-Speed,它有三个版本 它一开始选择的是 分割Optimizer 所以我觉得它这些选择应该,你们comment一下它这些选择的choice 因为它应该是先选择说 如果分割Optimizer 我需要的传输是最少的 然后接下来的分割 需要的传输是越来越多的 可以comment一下 为什么会这样 好,就是 它zero one 其实它选择分割 因为大家可以看到 左边的这个图 那左边的这个图呢 我中间这边其实用蓝色的线 切开,它上面是32bit 然后下面是16bit 所以实际上 32bit的东西不仅比较大 它用到的时间也比较少 像是这个Atom的Optimizer 它会用到的时候其实是 你要更新一个Global Step的时候 才会用到Atom的Optimizer 所以上面这三个东西其实 是比较少用到的
              
                  54:16
                  那他们就选择说 先优先放在CPU上面 那剩下的东西 还是放在GPU上面 那基本上就是有这么样的分割 去让他们有一个initial的想法 说可以第一个选择先放 Optimizer,然后速度也不会 降的那么夸张 好,大家还有 问题想问吗? 好,没有的话我们就 再次谢谢助教,谢谢 谢谢教授,谢谢教授 好,谢谢 好,谢谢 好,那我们还是休息五分钟 然后让助教换个场 等一下这样 作业五,那我们四点五 五十五分回来吧
              
            