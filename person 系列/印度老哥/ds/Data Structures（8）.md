
## 37、二叉搜索树中的中序后继节点

在本节课中，我们将解决二叉搜索树上的另一个有趣问题：给定二叉搜索树中的一个节点，我们需要找到它的中序后继节点，即在二叉搜索树的中序遍历中紧接在该节点之后的节点。众所周知，在二叉树的中序遍历中，我们首先访问左子树，然后访问根节点，最后访问右子树。左右子树均以相同方式递归遍历。

因此，对于每个节点，我们首先访问其左子树，然后访问节点本身，最后访问其右子树。在本系列的前一节课中，我们已经详细讨论了中序遍历。你可以查看本视频的描述以获取相关链接。中序遍历的实现基本上是一个递归函数，类似于我在这里展示的内容。这个函数中有两个递归调用，一个用于访问左子树，另一个用于访问右子树。中序遍历的时间复杂度为O(n)，其中n是树中的节点数量。

我们访问每个节点恰好一次，因此所花费的时间与树中的节点数量成正比。我在这里画了一个整数的二叉搜索树。众所周知，二叉搜索树是一种二叉树，其中对于每个节点，左侧节点的值较小，右侧节点的值较大。让我们快速看看这个二叉搜索树的中序遍历会是什么样子。我们将从树的根节点开始。对于任何节点，我们首先需要访问其左侧的所有节点，然后才能访问该节点本身。

所以我们必须向左走。基本上，我们会递归调用以访问这个节点的左子节点。对于这个节点，我们再次发现左边有东西。因此，我们会进行另一个递归调用并访问它的左子节点。现在我们到达值为8的这个节点，我们需要再向左走一次。现在对于值为6的这个节点，它是一个叶子节点，左边没有任何东西。所以我们可以简单地认为它的左子树已经完成，因此我们可以访问这个节点。对我来说，访问就是读取该节点中的数据。我会在这里写下数据。

ancestor
successor

现在对于这个节点来说，右边也没有任何东西了。所以我们可以简单地认为它的右边也处理完了。现在这个节点已经完全处理完毕。因此，这个节点对应的递归调用将结束，我们将返回到其父节点对应的调用。如果我们从一个节点的左子节点返回，那么这个节点将是未被访问的状态。因为只有在其左子树处理完成后，我们才能访问该节点。所以当我们返回到8时，8是未被访问的状态。

因此，我们可以直接访问这个节点，也就是读取该节点中的数据。当我访问一个节点时，我会将其标记为黄色。现在这个节点的右侧没有任何内容，所以我们可以直接认为右侧已完成。现在我们已完成对该节点的处理，因此与该节点对应的调用将结束，我们将返回到其父节点。再次，我们从左侧返回到父节点。此时，父节点（即值为10的这个节点）尚未被访问。

如果我们从右侧返回一个节点，那么它已经被访问过了。现在我正在访问10，现在我们可以去10的右侧。到目前为止，我们已经访问了三个节点。我们首先访问了值为6的节点，然后访问了值为8的节点。所以8是6的后继节点，然后10是8的后继节点。现在让我们看看10的后继节点会是什么。对于值为6和8的节点，右侧没有任何东西。所以我们正在回溯并转到父节点。但是对于一个节点，如果右侧有东西，也就是说如果有一个右子树，那么它的后继节点肯定会在它的右子树中。

因为在访问那个节点后，我们将向右移动。现在在这个阶段，我们处于值为12的节点。对于这个节点，我们首先会向左移动，现在处于值为11的节点，这是一个叶子节点。左边没有内容。所以我们可以简单地说左边已经完成，可以打印数据了。也就是说访问这个节点。因此，10的有序后继是11。现在对于值为11的节点，右边没有内容。所以我们将回到它的父节点，现在可以访问这个节点了。所以在11之后，我们有12。12的右边没有内容。所以对这个节点的调用将结束，我们将回到它的父节点。现在我们又回到了10。

但这次是从右边来的。所以这个节点已经被访问过了。所以我们不需要做任何操作，可以直接去它的父节点。现在我们来到了值为15的这个节点。我们是从左边过来的。这个节点还没被访问过。所以我们可以访问它。然后我们可以去它的右边。我们会继续这样下去。15的后继节点会是16。在16之后我们会打印17。然后在17之后我们会打印20。接着是25。最后一个元素会是27。这就是这个二叉搜索树的中序遍历。注意到我们打印出来的数字是按顺序排列的。

当我们对二叉搜索树进行中序遍历时，元素会按排序顺序被访问。现在我们要解决的问题是，给定树中的一个值，我们想找到它的中序后继节点。在二叉搜索树中，这将是树中下一个更高的值。但这有什么大不了的？难道我们不能直接进行中序遍历并在遍历过程中找出后继节点吗？当然可以这样做，但这样做的代价很高。中序遍历的时间复杂度是O(n)。

我们可能希望做得更好。在某些数据中查找下一个和上一个元素可能是一项频繁执行的操作。而二叉搜索树的优点在于，像插入、删除和搜索这样的频繁操作的时间复杂度为O(h)，其中h是树的高度。因此，如果我们能在O(h)的时间复杂度内找到后继和前驱节点，那将是非常好的。我们总是尽量保持树的平衡以最小化其高度。平衡二叉树的高度为以2为底的log n。而任何操作的O(log n)运行时间几乎是我们能获得的最佳运行时间。

那么我们能否在O(h)的时间复杂度内找到中序后继节点呢？我在这里重新绘制了示例树。让我们看看在不同情况下我们能做些什么。值为10的这个节点的后继节点会是哪个？我们能否通过逻辑推理得出答案？回想一下我们之前模拟的中序遍历过程，如果已经访问过当前节点，说明其左子树已经遍历完毕。此时我们已经读取了该节点的数据，接下来需要转向右子树。在右子树中，只要存在左子节点，我们就需要不断向左遍历。就像这个值为11的节点，当无法继续向左时（此处该节点没有左子节点），就到达了后继节点。

那么这就是我接下来要访问的节点。所以对于我们的节点来说，如果存在右子树，那么中序后继节点就是其右子树中最左边的节点。在二叉搜索树（BST）中，它将是右子树中值最小的节点。我称这种情况为第一种情况。在这种情况下，我们只需要在右子树中尽可能地向左走。在BST中，这也意味着寻找右子树中的最小值。最左边的节点同时也是子树中的最小值。这是一种情况。我们这里的节点有一个右子树。如果没有右子树，后继节点会是什么呢？

访问完值为8的节点后，我们该访问哪个节点？这个节点没有右子树。如果我们已经访问过这个节点，说明我们已访问过它的左子树和它本身。而右侧没有内容，因此可以认为右侧也已访问完毕。但我们还没找到后继节点。那么接下来该往哪里走？回想我们之前模拟的过程，这时需要回溯到这个节点的父节点。如果是从左侧回溯到父节点（当前就是这种情况），那么这个父节点应该尚未被访问过。

对于这个值为10的节点，我们刚刚完成了它的左子树遍历并返回。因此现在我们可以访问这个节点了。这就是我的后继节点。现在让我们选择另一个没有右子树的节点。对于这个值为12的节点，什么节点不会成为它的后继节点？接下来我们会访问哪个节点？这里再次说明，该节点没有右子树。所以我们必须回溯到它的父节点，看看它是否未被访问过。

但如果我们从右侧回溯到父节点，即当前刚访问的节点是右子节点（如本例所示），那么其父节点必然已被访问过——因为我们是在遍历完右子树后返回的。该父节点一定是在转向右子树前就被访问过了。此时我们该怎么做？

递归会继续回溯到10的父节点。现在我们从左侧抵达15。由于这个节点尚未被访问，因此可以访问它——这就是我们要找的后继节点。若目标节点没有右子树，则需要向上查找最近的祖先节点，且要求目标节点位于该祖先节点的左子树中。以节点12为例：我们先回溯到10。

但12位于10的右子树中。所以我们继续查找下一个祖先节点15。而12位于15的左侧。因此，15就是12在其左侧的最近祖先节点。于是，这就是我的中序后继节点。这个算法运行良好，但存在一个问题：我们如何从一个节点访问其父节点？其实，我们可以设计树的结构，使节点能够引用其父节点。到目前为止，在大多数课程中，我们将节点定义为包含三个字段的结构体，类似于这样。这就是我们在C或C++中定义节点的方式。

我们有一个字段用于存储数据，还有两个指向节点的指针用于存储左右子节点的引用或地址。通常，再增加一个字段来存储父节点的地址会非常有用。我们可以这样设计树结构，这样在利用父链接向上遍历树时就不会有问题。我们可以轻松地访问祖先节点。但如果没有指向父节点的链接呢？在这种情况下，我们可以从根节点开始，沿着树从根节点走到给定的节点。在二叉搜索树（BSD）中，这非常简单。对于12，我们将从根节点开始。12小于根节点的值。

所以我们需要向左走。现在我们到了10。因为12比10大，所以我们需要向右走。现在我们到了12。如果我们从根节点走到给定节点，就会经过该节点的所有祖先节点。中序后继节点就是这条路径上最深的那个节点或最深的祖先节点，且给定节点位于该节点的左子树中。12只有两个祖先节点。我们有10，但12位于10的右侧。然后我们有15，12位于15的左侧。所以15就是我的后继节点。

现在，我们使用这个方法来寻找6的后继节点。首先，我们将从根节点向下走到这个节点。6位于所有祖先节点的左侧。但对于6来说，最深的、6位于其左侧的祖先是值为8的这个节点。所以这就是我的后继节点。记住，我们只需要在没有右子树的情况下查看祖先节点。对于6来说，没有右子树。好的，这个算法看起来不错。现在让我们来编写代码。在我的C++代码中，我将编写一个名为get_successor的函数，它将接收根节点的地址和需要查找后继节点的另一个节点的地址。

这个函数将返回后继节点的地址。我们可以用不同的方式设计这个函数。不同于传入需要查找后继节点的指针作为参数，我们可以直接传入数据作为参数。对于这个数据元素，我们可以找到后继节点并返回其地址。因此这里的返回类型是结构体节点指针（struct node*），因为我们将在指针中返回地址。或者我们也可以直接返回元素本身，即后继元素本身。

我们可以使用其中任何一种签名来实现。让我们来实现这个。我们将传入当前节点的数据，并返回后继节点的地址。现在，我们需要做的第一件事是搜索包含此数据的节点。我将调用一个名为find的函数，该函数将接收根节点的地址和数据，并返回指向包含此数据的节点的指针。如果此函数返回null，即如果在树中未找到该数据，我们可以直接返回null。否则，我们在这个名为current的节点指针中拥有当前节点的地址。

在二叉搜索树（BSD）中，查找操作的时间复杂度为O(h)，其中h是树的高度。在BSD中进行查找并不算特别耗时。如果我们传递当前节点的地址而非数据作为第二个参数，本可以避免这次查找。不过暂且先这样处理。现在让我们来找出该节点的后继节点。如果该节点存在右子树（即右子树不为空），我们需要找到右子树中最左侧的节点。这里我声明了一个临时节点指针，初始时将其指向current.right。

使用这个while循环，我将到达最左侧的节点。只要左侧还有节点，就继续前进。最终，当我退出这个循环时，我将获得右子树中最左侧节点的地址，并可以返回这个地址。这个特定的节点也将是右子树中值最小的节点。我将把这段代码移到另一个函数中。我编写了一个名为findMin的函数，它将返回树或子树中值最小的节点。在getSuccessor函数中，我只需说return findMin，并传入当前节点右子节点的地址。

简单来说，我在这里传递的是右子树。好的，现在我们来讨论第二种情况。如果没有右子树，我们需要做的是从根节点遍历到当前节点，并找到最深的祖先节点，使得当前节点位于该祖先节点的左子树中。接下来，我要做的是声明一个名为successor的节点指针，初始时将其设为null。同时，我还会声明另一个名为ancestor的节点指针，初始时将其设为根节点。通过这个while循环，我们将遍历树直到到达当前节点。

要遍历这棵树，我们将利用二叉搜索树的性质：对于每个节点，左侧节点的值较小，右侧节点的值较大。如果当前节点的数据小于祖先节点的数据，那么这个祖先节点可能是当前节点的中序后继，因为当前节点位于其左侧。因此，我们可以将这个祖先节点设为后继节点，并在遍历过程中继续向左搜索。如果找到一个更深的节点，且当前节点位于其左子树中，那么后继节点将被更新。

否则，如果当前节点位于右侧，我们只需向右移动。当我们退出这个while循环时，后继节点要么为null，要么是某个节点的地址。并非树中的所有节点都有后继节点。具有最大值的节点不会有后继节点。退出这个while循环后，我们可以返回后继节点。所以，这就是我的getSuccessor函数，我认为它应该可以工作。你可以在本视频的描述中找到完整源代码的链接。总的来说，这个函数的时间复杂度将是O(H)。

而这正是我们想要的。我们希望在O(H)时间内找到后继节点。这里，我们已经在O(H)时间内进行搜索。寻找最小值同样需要O(H)时间，在二叉搜索树中从根节点走到某个节点也需要O(H)时间。因此，整体复杂度是O(H)。如果你理解了这个代码和逻辑，那么编写查找前驱节点的函数对你来说应该非常容易。我鼓励你动手写一写。今天就讲到这里。在接下来的课程中，我们将解决更多关于二叉树和二叉搜索树的有趣问题。感谢观看。

## 38、图

大家好。在本系列关于数据结构的讲解中，我们已经讨论了一些线性数据结构，如数组、链表、栈和队列。在这些结构中，数据是以线性或顺序的方式排列的，因此我们可以称它们为线性数据结构。我们还讨论了树，这是一种非线性数据结构。树是一种层次结构。现在我们知道，数据结构是存储和组织数据的方式。对于不同类型的数据，我们会使用不同类型的数据结构。

在本节课中，我们将向大家介绍另一种非线性数据结构，这种结构在计算机科学的众多场景中都有广泛应用。它被用来建模和表示各种系统。这种数据结构就是图。当我们学习数据结构时，通常首先将它们作为数学或逻辑模型来研究。同样地，我们也将首先把图作为一种数学或逻辑模型来研究，稍后再深入探讨其实现细节。

好的，那我们开始吧。图（graph）和树（tree）一样，都是由称为节点（node）或顶点（vertex）的对象或实体组成的集合，这些节点通过一组边（edge）相互连接。但在树中，连接方式必须遵循特定规则——在具有n个节点的树中，必须有恰好n减1条边，每条边对应一个父子关系。正如我们所知，树中的每条边都代表一个父子关系。

树中除根节点外的所有节点都有一个父节点，且只有一个父节点。这就是为什么如果有n个节点，就一定有n减1条边。在树中，所有节点都必须可以从根节点到达。并且从根节点到任意节点必须只有一条可能的路径。而在图中，没有规定节点之间如何连接的规则。图由一组节点和一组边组成。边可以以任何可能的方式连接节点。树只是一种特殊的图。

如今，图作为一种概念已在数学领域得到广泛研究。如果你修过离散数学课程，那么你一定已经了解图的概念。在计算机科学中，我们主要学习和实现数学中的这一相同概念。对图的研究通常被称为图论。用纯数学术语来说，我们可以这样定义图：图G是由顶点集V和边集E组成的有序对。

现在，我在这里使用一些数学术语。有序对就是一对数学对象，其中对象的顺序很重要。这就是我们书写和表示有序对的方式。对象之间用逗号分隔，放在括号内。由于顺序在这里很重要，我们可以说V是有序对中的第一个对象，E是第二个对象。有序对AB不等于BA，除非A和B相等。在我们这里的图定义中，有序对的第一个对象必须始终是一组顶点，第二个对象必须是一组边。

这就是为什么我们称这对为有序对。我们也有无序对的概念。无序对就是两个元素的集合。在这里顺序并不重要。我们用花括号来表示无序对，因为顺序在这里无关紧要。无序对AB等于BA。哪个对象在前，哪个对象在后并不重要。好的，回到正题。所以图是一组顶点和一组边的有序对，G等于VE是我们用来定义图的正式数学符号。现在，我在右边画了一个图。

这张图有8个顶点和10条边。现在我要给这些顶点命名，因为图中的每个节点都需要有标识。可以是名称也可以是索引编号。我将这些顶点命名为V1、V2、V3、V4、V5等。这种命名方式并不表示任何顺序关系，不存在第一、第二、第三节点之分。我可以随意给任何节点命名。因此我的顶点集合如下，共包含8个元素：V1、V2、V3、V4、V5、V6、V7和V8。

这就是这个图的顶点集。那么边集是什么呢？要回答这个问题，我们首先需要知道如何表示一条边。一条边由其两个端点唯一确定。因此，我们可以简单地将一条边的两个端点的名称写成一个对，作为这条边的表示。但边可以分为两种类型：可以是有向边，其中连接是单向的；也可以是无向边，其中连接是双向的。

在我展示的这个示例图中，边是无向的。但如果你还记得我之前展示的那棵树，那棵树中的边是有向的。通过我在这里展示的有向边，我们表示存在从顶点u到v的链接或路径。但我们不能假设存在从v到u的路径。这种连接是单向的。对于有向边来说，其中一个端点是起点，另一个端点是终点。我们用箭头指向终点来绘制这条边。

对于我们这里的边，起点是u，终点是v。有向边可以表示为一个有序对。有序对中的第一个元素可以是起点，第二个元素可以是终点。因此，这个有向边表示为有序对u、v，我们有一条从u到v的路径。如果我们想要一条从v到u的路径，我们需要在这里画另一条有向边，以v为起点，u为终点。这条边可以表示为有序对v、u。上面的是u、v，下面的是v、u。它们并不相同。

如果边是无向的，那么连接是双向的。无向边可以表示为一个无序对。由于边是双向的，起点和终点并不固定，我们只需要知道边连接的是哪两个端点。既然我们已经知道如何表示边，那么就可以为这个示例图写出边的集合。这里有一条连接v1和v2的无向边，还有一条连接v1和v3的无向边。

接下来我们有v1和v4。这其实很简单，我直接把它们都写出来。这就是我的边集合。通常在图中，所有边要么是有向的，要么是无向的。虽然一个图可以同时包含有向边和无向边，但我们不会研究这类图。我们只研究所有边要么全是有向的，要么全是无向的图。所有边都是有向的图称为有向图（digraph），而所有边都是无向的图就称为无向图。无向图没有特别的名称。

通常，如果图是有向的，我们会明确称其为有向图（digraph）。因此，图可以分为两种类型：有向图（digraph），其中边是单向的或有序对；以及无向图，其中边是双向的或无序对。现在，许多现实世界的系统和问题都可以用图来建模。图可以用来表示任何具有某种成对关系的对象集合。让我们来看一些有趣的例子。像Facebook这样的社交网络可以表示为一个无向图。

用户就是图中的节点，如果两个用户是朋友关系，就会有一条边连接他们。真实的社交网络会有数百万乃至数十亿个节点。由于空间有限，我在这里的图表中只能展示少数几个。社交网络是一种无向图，因为友谊是一种双向关系。如果我是你的朋友，那么你也是我的朋友。所以连接必须是双向的。

一旦系统被建模为图，许多问题就可以通过应用图论中的标准算法轻松解决。比如在这个社交网络中，假设我们想为用户推荐朋友。以Rama为例，一种可能的推荐方法是推荐那些尚未与他建立联系的朋友的朋友。Rama有三个朋友：Ella、Bob和Katie，我们可以推荐这三个朋友中尚未与Rama建立联系的朋友的朋友。Ella的朋友中已经没有尚未与Rama建立联系的人了。

然而，鲍勃有三个朋友——汤姆、山姆和李——他们与拉玛并非好友关系，因此可以推荐给拉玛。凯蒂也有两位朋友（李和斯瓦蒂）与拉玛没有关联，其中李已被计入推荐名单。综上，我们总共能向拉玛推荐这四位用户。虽然我们以社交网络为背景描述这个问题，但它本质上是一个标准的图论问题——用纯粹图论术语来说，就是找出所有与给定节点最短路径长度为2的节点。

标准算法可以用来解决这个问题。我们稍后会讨论图中的路径等概念。现在只需知道，我们在社交网络背景下描述的这个问题是一个标准的图论问题。好的，像Facebook这样的社交网络是一个无向图。现在让我们看另一个例子。互联网或万维网上相互链接的网页可以表示为一个有向图。具有唯一地址或URL的网页将成为图中的一个节点。

如果某个页面包含指向另一个页面的链接，我们就可以画一条有向边。现在再次强调，网络上有数十亿个页面，但我这里只能展示少数几个。这个图中的边是有向的，因为这次的关系不是相互的。如果页面A有指向页面B的链接，并不意味着页面B也必须有指向页面A的链接。假设mycodeschool.com上的某个页面有一个关于图的教程。

在本页面中，我放了一个维基百科关于图的文章链接。假设在我展示的这个示例图中，页面B是我的MyCodeSchool关于图的教程，地址或URL为mycodeschool.com/videos/graph。而页面Q则是维基百科关于图的文章，URL为wikipedia.org/wiki/graph。现在在我的页面，也就是页面B上，我放了一个指向维基百科图页面的链接。如果你在页面B上，可以点击这个链接跳转到页面Q。但维基百科并没有这样做。

如果你在页面 Q 上，就无法通过点击链接跳转到页面 P。这里的连接是单向的，因此我们用有向边来表示。好，现在既然我们能够将网络表示为有向图，就可以应用标准的图论算法来解决问题和执行任务。像谷歌这样的搜索引擎经常执行的一项任务就是网络爬取。

搜索引擎使用一种名为网络爬虫的程序，该系统性地浏览全球网络以收集和存储网页数据。随后，搜索引擎可利用这些数据针对搜索查询提供快速准确的结果。尽管在此情境下我们使用了"网络爬取"这样专业且重要的术语，但实际上网络爬取本质上就是图遍历——用更简单的话说，就是访问图中所有节点的行为。不用说，图遍历自然有标准算法可供使用。我们将在后续课程中学习图遍历算法。

好的，接下来我要讲的是加权图的概念。有时候在图中，所有的连接并不能被视为同等重要。某些连接可能比其他连接更可取。例如，我们可以将城市间的公路网络（即城市间的高速公路和快速路网络）表示为一个无向图。我假设所有高速公路都是双向的。

城市内部的道路网络，即城市内的道路系统，必然包含单向道路，因此城市内部的道路网络必须表示为有向图。而在我看来，城际道路网络则可以表示为无向图。显然，我们不能将所有连接视为等同。道路的长度各不相同，为了执行许多任务、解决许多问题，我们需要考虑道路的长度。在这种情况下，我们为每条边赋予一定的权重或成本。我们用权重来标记这些边。在这里，权重可以是道路的长度。

所以，我在这里要做的是给这些边标上一些长度值。假设这些值的单位是公里，现在这个图中的边就有了权重，这个图就可以称为加权图。假设在这个图中，我们想从城市A到城市D选择最佳路线。看看这四条可能的路线。

我用不同颜色标出了这些路径。如果我把所有边都视为同等重要，那么经过B和C的绿色路线与经过E和F的红色路线就是同等优秀的——这两条路径都包含三条边。而经过E的黄色路线则是最佳选择，因为这条路径只有两条边。但当给连接赋予不同权重时，我需要累加路径中各边的权重来计算总成本。考虑权重因素后，最短路线其实是经过B和C的那条。

连接具有不同的权重，这在这张图中非常重要。实际上，我们可以将所有图视为加权图。未加权图基本上可以看作是所有边权重相同的加权图，通常我们假设权重为1。好的，我们已经将城际公路网络表示为一个加权无向图。社交网络是一个未加权无向图，万维网是一个未加权有向图，而这个是一个加权无向图。这就是城际公路网络。

我认为城市内部的道路网络可以建模为加权有向图，因为城市中会存在一些单行道。城市道路网络中的交叉路口可以作为节点，而路段则是我们的边。顺便说一句，我们也可以将无向图表示为有向图。只不过对于每条无向边，我们会有两条有向边。我们可能无法将有向图重新绘制为无向图，但我们总是可以将无向图重新绘制为有向图。好的，我就先讲到这里。这些内容作为入门课程已经足够了。下节课我们会讨论图的更多性质。这节课就到这里。感谢观看。

## 39、图的性质

在上节课中，我们向大家介绍了图的概念。我们将图定义为一种数学或逻辑模型，并讨论了图的一些性质和应用。在本节课中，我们将进一步探讨图的更多性质，但首先我想快速回顾一下上节课的内容。图可以被定义为一个有序对，由一组顶点和一组边组成。我们用这个正式的数学符号G=(V,E)来定义一个图。这里V是顶点的集合，E是边的集合。有序对就是一对数学对象，其中对象的顺序很重要。

配对中哪个元素在前、哪个元素在后很重要。我们知道表示集合中元素数量的方式（也称为集合的基数），使用的是与模数或绝对值相同的符号。因此，这就是我们表示图中顶点数量和边数量的方法。顶点数将是集合V中的元素数量，边数将是集合E中的元素数量。在接下来的所有解释中，我将用这种方式表示顶点数和边数。正如我们之前讨论过的，图中的边可以是有向的（即单向连接），也可以是无向的（即双向连接）。

仅包含有向边的图称为有向图或关系图，而仅包含无向边的图称为无向图。有时图中所有连接不能被视为同等重要，因此我们会给边赋予权重或成本值（如图所示）。这种在连接上关联了成本值或权重的图称为加权图。若图中各边不存在成本差异，则该图属于无权图。好的，现在我们还可以在图中看到一些特殊的边。这些边会使算法复杂化，增加处理图的难度，但我还是要谈谈它们。如果一条边只涉及一个顶点，那么它就被称为自环或自边。

如果一条边的两个端点相同，则称为自环。我们可以在有向图和无向图中都存在自环，但问题是为什么我们会在图中使用自环呢？其实，有时候如果边表示某种关系或连接，而这种关系或连接可能发生在同一个节点既作为起点又作为终点的情况下，那么我们就可以有自环。例如，正如我们在之前的课程中讨论的那样，互联网或万维网上相互链接的网页可以表示为有向图。

具有唯一URL的页面可以成为图中的节点，如果某个页面包含指向另一个页面的链接，我们就能建立一条有向边。在这个图中可能会出现自循环情况，因为网页完全有可能包含指向自身的链接。以mycodeschool.com/videos这个网页为例——在页面顶部导航栏中设有"训练题库"、"习题集"和"视频库"的链接。虽然当前我已处于视频库页面，但仍可点击视频库链接，此时只会触发页面刷新，因为我的起始位置和目标位置是相同的。

所以，如果按照我们刚才讨论的方式，将万维网表示为一个有向图，那么这里就有一个自循环。接下来我想讨论的另一种特殊边类型是多边。如果一条边在图中出现多次，就称为多边。同样地，多边既可以出现在有向图中，也可以出现在无向图中。我这里展示的第一个多边是无向的，第二个是有向的。那么问题又来了，为什么我们有时需要多边呢？

假设我们用图来表示城市之间的航线网络。城市可以看作节点，如果两个城市之间有直达航班，就可以画一条边。但同一对城市之间可能有多个航班。这些航班有不同的名称和价格。如果我想在图中保留所有航班的信息，可以绘制多重边。为每个航班画一条有向边，然后给每条边标注价格或其他属性。

我刚才只是随便用一些航班号给这些边标了号。正如我们之前所说，自环边和多边常常会让图的处理变得复杂。它们的存在意味着我们在解决问题时需要格外小心。如果一个图不包含自环边或多边，它就被称为简单图。在我们的课程中，我们主要处理的是简单图。现在，我想让你回答一个非常简单的问题。给定一个简单图（即没有自环边或多边的图）的顶点数量，那么可能的最大边数是多少呢？让我们来看看。

假设我们要画一个有四个顶点的有向图。我已经在这里画了四个顶点。我会将这些顶点命名为V1、V2、V3和V4。这就是我的顶点集合。集合V中的元素数量是4。现在，如果我选择不在这里画任何边，这仍然是一个图。边的集合可以是空的。节点可以完全不相连。因此，图中可能的最小边数是0。那么，如果这是一个有向图，你认为这里可能的最大边数是多少？每个节点都可以有指向所有其他节点的有向边。

在此图中，每个节点可以指向其他三个节点。我们总共有四个节点。因此，这里可能的最大边数是4乘以3，即12。我用同一种颜色显示了从一个顶点出发的边。如果没有自环或多重边，这就是我们能画出的最大边数。一般来说，如果有n个顶点，那么有向图中的最大边数就是n乘以n-1。

因此，在一个简单的有向图中，边的数量范围是从0到n乘以n-1。那么你认为无向图的最大边数会是多少呢？在无向图中，一对节点之间只能有一条双向边。我们无法在不同方向上拥有两条边。所以这里的最大值是有向图最大值的一半。因此，如果图是简单且无向的，边的数量范围是从0到n乘以n-1再除以2。请记住，这只有在没有自环或多重边的情况下才成立。

现在你可能会发现，图中的边数相对于顶点数来说可能非常非常大。例如，一个有向图中如果有10个顶点，那么最大的边数可以达到90条。如果顶点数是100，那么最大边数可以达到9900条。最大边数接近于顶点数的平方。如果一个图中的边数接近于可能的最大边数，那么这个图就被称为稠密图。

也就是说，当边的数量与顶点数量的平方处于同一数量级时，该图被称为稠密图；而如果边的数量明显较少（通常接近顶点数量且不超过），则称为稀疏图。稠密与稀疏之间并没有明确的界限划分，完全取决于具体情境。但这一分类至关重要——在处理图结构时，许多决策都是基于该图属于稠密图还是稀疏图来制定的。

例如，我们通常会在计算机内存中为稠密图选择一种不同的存储结构。对于稠密图，我们通常使用邻接矩阵来存储；而对于稀疏图，我们则通常采用邻接表。我将在下一课中详细讲解邻接矩阵和邻接表。好的，接下来我要讨论的概念是图中的路径。图中的路径是指一系列顶点，其中序列中每一对相邻顶点都由一条边连接。在这个示例图中，我正在高亮显示一条路径。顶点序列A、B、F、H就是该图中的一条路径。现在我们这里有一个无向图，边是双向的。

在有向图中，所有边也必须沿一个方向对齐，即路径的方向。如果路径中没有重复的顶点，则称为简单路径；如果顶点不重复，那么边也不会重复。因此，在简单路径中，顶点和边都不会重复。我在这里高亮显示的路径A、B、F、H是一条简单路径，但我们也可以有这样的路径。这里的起始顶点是A，结束顶点是D。在这条路径中，有一条边和两个顶点被重复了。在图论中，关于"路径"这一术语的使用存在一些不一致之处。大多数情况下，当我们说路径时，指的是简单路径；若允许重复，则使用术语"行走"。因此，路径本质上是一种顶点和边都不重复的行走。若允许顶点重复但边不重复，则这种行走被称为"轨迹"。

在这个示例图中，我正在高亮显示一条路径。好的，现在我想再次说明一下，虽然walk和path经常被当作同义词使用，但大多数情况下，当我们说path时，指的是简单路径。也就是顶点和边都不重复的路径。在两个不同的顶点之间，如果存在一条像我在这个示例图中展示的这样顶点或边重复的路径，那么也必然存在一条不重复顶点或边的简单路径。在这个我展示的路径中，我们从A出发，最终到达C。从A到C其实有一条只需经过一条边的简单路径。我们只需要避免经过B、E、H、D然后再绕回A即可。

这就是为什么我们主要讨论两个顶点之间的简单路径，因为如果存在任何其他路径，简单路径也必然存在，并且寻找简单路径是最合理的做法。因此，在接下来的课程中，我将采用这种做法。当我说"路径"时，我指的是简单路径；如果不是简单路径，我会明确说明。如果一个图中任意顶点到任意其他顶点都存在路径，那么这个图就被称为强连通的。

如果是一个无向图，我们简单地称它为连通的；如果是一个有向图，我们称它为强连通的。在我展示的最左边和最右边的图中，我们可以从任意顶点到任意其他顶点找到一条路径，但在中间的图中，我们无法从任意顶点到任意其他顶点找到路径。我们无法从顶点C到顶点A。我们可以从A到C，但不能从C到A。所以这不是一个强连通图。

记住，如果是无向图，我们直接说它是连通的；如果是有向图，我们则说它是强连通的。如果一个有向图不是强连通的，但可以通过将所有边视为无向边而变成连通图，那么这样的有向图被称为弱连通的。如果我们忽略边的方向，这个图是连通的，但我建议你只需记住连通和强连通这两个概念。

最左侧的无向图是连通的。我移除了其中一条边后，它就不再连通了。现在我们有两个互不相连的连通分量，但整个图不再连通。图的连通性是一个非常重要的属性。如果你还记得市内道路网络，城市内的道路网络有很多单行道，可以用有向图来表示。一个市内道路网络应该始终是强连通的。我们应该能够从任何街道到达任何街道，从任何交叉口到达任何交叉口。

好的，现在我们理解了路径的概念，接下来我想谈谈图中的循环。如果一条行走路径的起点和终点是同一个顶点，那么它就被称为闭合行走。就像我这里展示的这样，还有一个条件是行走的长度必须大于零。行走或路径的长度是指路径中边的数量。比如我展示的这个闭合行走，长度是五，因为这条路径中有五条边。所以，闭合行走就是起点和终点相同且长度大于零的行走路径。

现在有些人可能会把闭合路径称为环，但我们通常用“环”来指代简单环。简单环是一种闭合路径，除了起点和终点外，其他顶点或边都不会重复出现。目前我在这个示例图中展示的就是一个简单环，或者我们可以直接称之为环。没有环的图被称为无环图。如果一棵树用无向边绘制，那就是无向无环图的一个例子。在这棵树中，我们可以有闭合路径，但不能有简单环。在我现在展示的这个闭合路径中，我们的边是重复的。

树中不会有简单的循环，除了树之外，我们还可以有其他类型的无向无环图。树还必须是连通的。现在我们也可以有有向无环图。正如你在这里看到的，我们也没有任何循环。你不能有一条长度大于零的路径，起点和终点是同一个顶点。有向无环图通常被称为DAG。

图中的循环在设计算法时会导致很多问题，比如寻找从一个顶点到另一个顶点的最短路径。在接下来的课程中，当我们学习一些高级算法时，我们会经常讨论循环。这节课我就讲到这里。下节课我们将讨论如何在计算机内存中创建和存储图。这节课就到这里。感谢观看。

## 40、图的表示：边列表

大家好。在前面的课程中，我们向大家介绍了图，并且还看了一些图的特性并进行了讨论。但到目前为止，我们还没有讨论如何实现图，如何在计算机内存中创建像图这样的逻辑结构。所以让我们试着讨论一下这个问题。

众所周知，图由一组顶点和一组边构成，这正是我们从纯数学角度对图的定义。图G被定义为顶点集V与边集E的有序对。要在计算机内存中创建并存储图，最简单的做法或许是建立两个列表：一个用于存储所有顶点，另一个用于存储所有边。

我们可以使用适当大小的数组或动态列表的实现来表示列表。实际上，我们可以利用语言库中提供的动态列表，比如C++中的vector或Java中的array list。现在，一个顶点是通过其名称来标识的。因此，第一个列表，即顶点列表，将只是一个名称或字符串的列表。在这个示例图中，我已经填写了所有顶点的名称。

那么我们应该在这个边列表中填写什么内容呢？一条边是通过它的两个端点来确定的。因此，我们可以将边创建为一个包含两个字段的对象。我们可以将边定义为一个结构体或类，其中包含两个字段：一个用于存储起始顶点，另一个用于存储结束顶点。边列表本质上就是这种边结构体类型的数组或列表。

在我这里写的关于edge的这两个定义中，第一个我使用了字符指针，因为在C语言中我们通常用字符指针来存储或引用字符串。我们也可以用字符数组。在C++或Java中我们可以创建类，字符串对我们来说是一种可用的数据类型。

因此我们也可以使用那个。因此我们可以使用这些中的任何一个作为字段。我们可以使用字符指针或字符数组，或者如果可用的话，使用字符串数据类型。这取决于你如何设计你的实现。现在让我们为这个示例图填充这个边列表。这里的每一行现在有两个框。

假设第一个用于存储起始顶点，第二个用于存储终止顶点。我们这里的图是一个无向图。因此，任何顶点都可以称为起始顶点，任何顶点也可以称为终止顶点。顶点的顺序在这里并不重要。我们这里有九条边。一条在A和B之间，另一条在A和C之间，还有一条在A和D之间。然后我们有BE和BF。我们也可以用FB代替BF作为条目，但只需要其中一个。接着我们有CG、DH、EH和FH。

其实还有一个。我们还有GH。这里总共有10条边，而不是9条。现在，再次因为这是一个无向图，如果我们说有一条从F到H的边，我们也等于说有一条从H到F的边。不需要再添加HF这条边。

我们将不必要地使用额外的内存。如果这是一个有向图，FH和HF将表示两种不同的连接，其中起始顶点和结束顶点会有所不同。也许在无向图的情况下，我们应该将这些字段命名为第一顶点和第二顶点；而在有向图的情况下，我们应该将它们命名为起始顶点和结束顶点。现在，我们的图也可以是一个加权图。我们可以为边关联一些成本或权重。如你所知，在无权重图中，所有连接的成本是相等的。

但在加权图中，不同的连接会有不同的权重或不同的成本。现在，在这个示例图中，我已经为这些边关联了一些权重。那么，你认为我们应该如何存储这些数据呢？边的权重。那么，如果图是带权重的，我们可以在边对象中再增加一个字段来存储权重。现在，我的边列表中的每个条目有三个字段。一个用于存储起始顶点，一个用于存储结束顶点，还有一个用于存储权重。所以，这是存储图的一种可能方式。我们可以简单地创建两个列表，一个用于存储顶点，另一个用于存储边。但这并不是很高效。

对于任何可能的数据存储和组织方式，我们都必须考虑其成本。当我们提到成本时，我们指的是两个方面：各种操作的时间成本和内存使用量。通常，我们会衡量时间消耗随输入或数据规模的增长速度，也就是我们所说的时间复杂度。同时，我们也会衡量内存消耗随输入或数据规模的增长速度，即所谓的空间复杂度。时间复杂度和空间复杂度通常用我们所说的大O符号表示。这节课我假设你已经了解了时间与空间复杂度分析和大O表示法。如果你想复习这些概念，可以查看本视频描述中的相关课程链接。我们总是希望尽量减少最频繁执行操作的时间成本，同时确保不会消耗不合理的高内存。好，现在我们来分析这个用于存储图的特定结构。

我们先来讨论内存使用情况。对于第一个列表——顶点列表，所需或消耗的最小行数等于顶点数量。现在，顶点列表中的每一行都是一个名称或字符串，而字符串可以是任意长度。目前所有字符串都只有一个字符，因为我只是简单地将节点命名为A、B、C等。但我们也可以使用多字符的名称，由于字符串长度可能不同，因此并非所有行消耗的内存都相同。就像这里这样。

在这里，我将城市内部的道路网络展示为一个加权图。城市是我的节点，道路距离是我的权重。现在，对于这个图，正如你所看到的，名称的长度各不相同。因此，顶点列表中的所有行或边列表中的所有行的成本并不相同。更多的字符将消耗更多的字节。但我们可以放心地假设名称不会太长。

我们可以安全地假设，在几乎所有实际场景中，字符串的平均长度都会是一个非常小的值。如果我们假设它总是小于某个常数，那么这个顶点列表消耗的总空间将与消耗的行数（即顶点数量）成正比。换句话说，这里的空间复杂度可以表示为顶点数量的O。我们就是用这两个竖线来表示顶点数量的。这里我们基本上指的是集合V中的元素数量。现在，对于边列表，我们再次在前两个字段中存储字符串。因此，这里的每一行消耗的内存量也不相同。

但如果我们只是存储字符串的引用或指针，就像第一行这里所示，那么这两个字段中就不需要填充具体值，而是可以存储指向顶点列表中名称的引用或指针。如果我们这样设计，每一行将消耗相同的内存。实际上这种方式更好，因为在大多数情况下，引用的开销远低于存储名称副本的开销。

作为参考，我们可以获取字符串的实际地址，这就是为什么我们说起始顶点和结束顶点可以是字符指针。或者，也许更好的设计是直接在顶点列表中存储名称或字符串的索引。假设顶点列表中A的索引是0，B的索引是1，C的索引是2，以此类推。

现在，对于起始顶点和结束顶点，我们可以使用两个整数字段。正如你在我的边定义中看到的，起始顶点和结束顶点现在都是整型。在边列表的每一行中，第一个和第二个字段都填充了整数值。我已经填写了适当的索引值。这无疑是一个更好的设计。而且，你可以看到，现在边列表中的每一行将占用相同的内存空间。

因此，总的来说，边列表所占用的空间将与边的数量成正比。换句话说，这里的空间复杂度是O(边数)。好了，这就是我们对内存使用的分析。这个设计的总体空间复杂度将是O(顶点数 + 边数)。这样的内存使用是否过高呢？实际上，如果我们想在计算机内存中存储一个图，很难做得比这更好了。

所以在内存使用方面我们没问题。现在让我们讨论一下操作的时间成本。你认为在处理图时最频繁执行的操作会是什么？处理图时最频繁执行的操作之一就是查找与给定节点相邻的所有节点。也就是查找与给定节点直接相连的所有节点。你认为查找与给定节点直接相连的所有节点的时间成本会是多少？嗯，我们将不得不扫描整个边列表。我们将不得不执行线性搜索。

我们需要遍历列表中的所有条目，检查每个条目的起始节点或结束节点是否是我们给定的节点。对于有向图，我们只需查看条目的起始节点是否与给定节点相同；而对于无向图，则需要同时检查起始节点和结束节点。运行时间将与边的数量成正比，换句话说，这个操作的时间复杂度将是边数的大O表示法。

好的，现在另一个常见的操作是判断两个给定的节点是否相连。在这种情况下，我们也需要对边列表进行线性搜索。在最坏的情况下，我们需要查看边列表中的所有条目。因此，最坏情况下的运行时间将与边的数量成正比。所以，对于这个操作来说，时间复杂度也是边数的大O表示法。现在，让我们来看看这个运行时间（边数的大O）是好还是坏。如果你还记得我们之前课程中的讨论，在一个简单图中，即没有自环或多重边的图中，如果顶点的数量（即集合V中的元素数量）等于n，那么在有向图中，边的最大数量将是n乘以n减1。每个节点都将与其他所有节点相连，当然，边的最小数量可以是零。我们可以有一个没有任何边的图。

如果图是无向的，最大边数将是n乘以n减1除以2。但总的来说，你可以看到，边的数量几乎可以达到顶点数量的平方。边的数量可以是顶点数量的平方级。让我们在这里将顶点数量表示为小v。因此，边的数量可以是v的平方级。在图中，通常任何以边数级运行的操作都会被认为是非常昂贵的。我们尽量保持操作在顶点数量的级数内。

当我们比较这两种运行时间时，这一点非常明显。V的大O表示法比V平方的大O表示法要好得多。总的来说，这种顶点列表和边列表的表示方式在操作时间成本方面并不高效。我们应该考虑其他更高效的设计。我们需要想出更好的方案。在下一课中，我们将讨论另一种存储和表示图的可能方式。本节课到此结束，感谢观看。


## 41、图的表示：邻接矩阵

在上一节课中，我们讨论了一种存储和表示图的可能方式，即使用两个列表：一个用于存储顶点，另一个用于存储边。顶点列表中的每条记录是一个节点的名称，而边列表中的每条记录是一个对象，包含对边的两个端点的引用以及该边的权重。因为我在这里展示的这个示例图是一个加权图。

我们将这种表示方法称为边列表表示法。但我们意识到，就最频繁执行操作的时间成本而言，这种存储方式效率不高。例如查找与给定节点相邻的节点，或判断两个节点是否相连。要执行这些操作中的任何一个，我们都需要扫描整个边列表。我们需要在边列表上进行线性搜索。因此，时间复杂度是边数的大O，而我们知道图中的边数可能非常非常大。

在最坏的情况下，它可能接近顶点数的平方。在图中，任何以边数为阶的运行都被认为是非常昂贵的。我们通常希望将成本控制在顶点数的阶数内。因此，我们应该考虑其他更高效的设计。我们应该想出比这更好的方案。另一种可能的设计是，我们可以将边存储在二维数组或矩阵中。我们可以有一个大小为V x V的二维矩阵或数组，其中V是顶点的数量。

如你所见，我在这里画了一个8x8的矩阵，因为我示例图中的顶点数是8。我们把这个矩阵命名为A。现在，如果我们想存储一个无权重图，只需移除这个示例图中的权重，这样我们的图就变成无权重图了。如果我们为每个顶点分配一个0到V-1之间的值或索引（就像这里的情况），当我们将顶点存储在顶点列表中时，每个顶点都有一个0到V-1之间的索引。我们可以说A是第0个节点，B是第1个节点，C是第2个节点，以此类推。

我们正在从顶点列表中提取索引。好的，如果图是无权重的，并且每个顶点的索引在0到V-1之间，那么在这个矩阵或二维数组中，我们可以将第i行第j列的元素aij设置为1或布尔值true，如果存在从i到j的边；否则设置为0或false。如果我要为这个示例图填充这个矩阵，那么我将逐个顶点进行处理。

顶点0与顶点1、2和3相连。顶点1与0、4和5相连。这是一个无向图，因此如果我们有一条从0到1的边，同时也有一条从1到0的边。所以，第1行第0列也应设为1。现在，我们来看节点2。它与0和6相连。3与0和7相连。4与1和7相连。5再次与1和7相连。6与2和7相连。而7则与3、4、5和6相连。数组中所有剩余位置都应设为0。

注意，这个矩阵是对称的。对于无向图来说，这个矩阵会是对称的，因为aij等于aji。每条边会填充两个位置。实际上，要查看图中的所有边，我们只需要遍历这两个半部分中的一个。而对于有向图来说，情况就不同了。每条边只会填充一个位置，我们需要遍历整个矩阵才能看到所有的边。

好的。现在，这种将图中的边或连接存储在矩阵或二维数组中的表示方法称为邻接矩阵表示法。我在这里画的这个特定矩阵就是一个邻接矩阵。现在，有了这种存储或表示方式，你认为查找与给定节点相邻的所有节点的时间成本是多少？假设给定这个顶点列表和邻接矩阵，我们想要找到与名为f的节点相邻的所有节点。如果我们给定一个节点的名称，那么我们首先需要知道它的索引。而要知道索引，我们就必须扫描顶点列表。没有其他办法。

一旦我们确定了索引，比如f的索引是5，我们就可以在邻接矩阵中找到对应索引的那一行，然后扫描整行来找出所有相邻的节点。在最坏的情况下，扫描顶点列表以确定索引所需的时间将与顶点的数量成正比，因为我们可能需要扫描整个列表。而扫描邻接矩阵的一行将再次花费我们与顶点数量成正比的时间，因为在每一行中，我们恰好有V列，其中V是顶点的数量。因此，这个操作的总时间成本是大O(V)。现在，在执行大多数操作时，我们必须传递索引以避免一直扫描顶点列表。如果我们知道一个索引，我们可以在常数时间内找到对应的名称，因为在数组中，我们可以在常数时间内访问任何索引处的元素。但如果我们知道一个名称并想找出对应的索引，那么这将花费我们大O(V)的时间。我们将不得不扫描顶点列表，并在其上执行线性搜索。好的，继续。

那么，判断两个节点是否相连的时间成本是多少呢？再次说明，这两个节点可能以索引或名称的形式提供。如果节点是以索引形式传递的，那么我们只需要查看特定行和特定列的值。也就是说，我们只需查看某个i和j对应的a[i][j]值。这个操作的时间复杂度是常数级的——在二维数组中，我们可以在恒定时间内访问任意单元格的值。因此，若给定的是索引，该操作的时间复杂度就是大O(1)，这意味着我们只需要耗费恒定的时间。

但如果给定了名称，那么我们也需要进行扫描以确定索引，这将耗费我们O(V)的时间复杂度。总体时间复杂度将是O(V)。常数时间的访问将变得毫无意义。可以避免一直扫描顶点列表来确定索引。我们可以使用一些额外的内存来创建一个哈希表，将名称和索引作为键值对存储。这样，从名称查找索引的时间成本也将是O(1)，即常数时间。

哈希表是一种数据结构，到目前为止，我在任何课程中都没有提到过它。如果你不了解哈希表，只需在网上搜索一下它的基本概念即可。好的，正如你所看到的，使用邻接矩阵表示法时，我们一些最频繁执行的操作的时间成本是按顶点数量级计算的，而不是按边数量级计算的，而边的数量可能高达顶点数量的平方。

好的，现在如果我们想用邻接矩阵表示法来存储一个加权图，那么矩阵中的aij可以设为边的权重。对于不存在的边，我们可以设置一个默认值，比如一个非常大的或最大可能的整数值，这个值永远不会被用作边的权重。我这里只是填了无穷大来表示我们可以选择无穷大或者任何其他永远不会是有效边权重的值作为默认值。

好的，现在为了进一步讨论，我将回到无权图。邻接矩阵看起来确实不错。那么，我们是不是应该总是使用它呢？嗯，采用这种设计，我们在时间上有所改进，但在内存使用上却大大增加了。我们不再使用与边数完全相等的存储单元，而是采用了一种无边的存储方式，这里我们精确地使用了V平方个存储单元。我们使用了O(V²)的空间复杂度。我们不仅存储了这两个节点相连的信息，还存储了它们不相连的信息，这部分信息很可能是冗余的。

如果一张图是密集的，即边的数量非常接近V的平方，那么这是好的；但如果图是稀疏的，即边的数量远小于V的平方，那么我们就是在浪费大量内存来存储这些零值。以我在这里绘制的示例图为例，在边列表中，我们消耗了10个单位的内存，边列表中有10行被占用，但在这里我们消耗了64个单位。大多数具有大量顶点的图不会非常密集，边的数量也不会接近V的平方。

举个例子，假设我们用图来模拟像Facebook这样的社交网络，网络中的每个用户是一个节点，如果两个用户是朋友，则有一条无向边。Facebook有十亿用户，但由于空间有限，我在这里的示例图中只展示了少数几个用户。我们就假设我们的网络中有十亿用户吧。所以，我们图中的顶点数是10的9次方，也就是十亿。那么，你认为我们社交网络中的连接数有可能接近用户数的平方吗？那就意味着网络中的每个人都是其他人的朋友。我们社交网络的用户不可能和所有其他十亿用户都成为朋友。

我们可以安全地假设，普通用户拥有的好友数量通常不会超过一千个。基于这一假设，我们的图中将会有10的12次方条边。实际上，这是一个无向图，因此我们需要将总数除以2，以避免重复计算同一条边。所以，如果平均朋友数是1000，那么在我的图中，连接的总数就是5乘以10的11次方。现在，这比顶点数的平方要小得多。所以，基本上，如果我们对这种图使用邻接矩阵，我们会浪费大量的空间。此外，即使不从相对角度来看，10的18次方存储单位，绝对意义上也是一个巨大的数字。10的18次方字节大约相当于一千拍字节。这确实是一个庞大的存储空间。

这么多数据永远不可能装在一个物理磁盘上。而5乘以10的11次方字节，也就是0.5太字节。如今一台普通的个人电脑就有这么大的存储空间。所以，你可以看到，对于像大型社交图这样的东西，邻接矩阵表示法并不是很高效。邻接矩阵适用于密集图，也就是边数接近顶点数平方的情况。有时，当可能的连接总数（即V的平方）非常少时，浪费的空间甚至无关紧要。但大多数现实世界的图都是稀疏的，邻接矩阵并不适用。让我们考虑另一个例子。让我们将万维网视为一个有向图。如果你把网页看作图中的节点，超链接看作有向边，那么一个网页不会链接到所有其他网页。而且，网页的数量将达到数百万级别。一个网页只会链接到少数其他网页。

因此，这个图会是稀疏的。大多数现实世界的图都是稀疏的，而邻接矩阵虽然为我们最常执行的操作提供了良好的运行时间，但并不适合，因为它在空间利用上效率不高。那么，我们该怎么办呢？其实，还有另一种表示方法，它能提供与邻接矩阵相似甚至更好的运行时间，同时不会占用那么多空间。这被称为邻接表表示法，我们将在下一课中详细讨论。本节课的内容就到这里。感谢观看。


## 42、图的表示：邻接表


在上一节课中，我们讨论了邻接矩阵作为存储和表示图的一种方法。正如我们分析和讨论这种数据结构时所看到的，它在操作的时间成本方面非常高效。使用这种数据结构，判断两个节点是否相连的时间复杂度是O(1)，即常数时间。

它的时间复杂度是O(V)，其中V是顶点数，用于查找与给定节点相邻的所有节点。但我们也发现，邻接矩阵在空间消耗方面效率不高。我们消耗的空间与顶点数的平方成正比。在邻接矩阵表示法中，如你所知，我们将边存储在一个大小为V乘V的二维数组或矩阵中，其中V是顶点的数量。在我这里的示例图中，我们有8个顶点。这就是为什么我这里有一个8乘8的矩阵。

我们正在消耗8个平方单位，也就是64个空间单位。现在，基本上发生的情况是，对于每个顶点、每个节点，我们在这个矩阵中有一行，用来存储它所有连接的信息。这是第0个节点（即A）的行。这是第1个节点（即B）的行。这是C的行。我们可以继续这样下去。因此，每个节点都有一行。一行本质上是一个一维数组，其大小等于顶点数，即V。那么，我们在一行中具体存储什么呢？让我们先看看第一行，这里存储的是节点A的连接情况。这个二维矩阵或数组实际上是由多个一维数组组成的数组。也就是说，每一行都必须是一个一维数组。

那么，我们如何在这个大小为8的一维数组的8个单元格中存储节点A的连接关系呢？第0位上的0表示没有从A出发到第0个节点（即A本身）的边。一条起点和终点相同的边被称为自环，而A没有自环。第1位上的1表示存在一条从A到第1个节点（即B）的边。我们在这里存储信息的方式是：利用这个一维数组的索引或位置来表示边的终点。对于这整个行、整个一维数组来说，起点始终相同，永远是第0个节点，也就是A。一般来说，在邻接矩阵中，行索引代表起点，列索引代表终点。

在这里，当我们仅查看第一行时，起点始终是A，索引0、1、2等代表终点，特定索引或位置的值告诉我们是否存在一条以该节点为终点的边。这里的1表示边存在，0则表示边不存在。现在，当我们以这种方式存储信息时，你会发现，我们不仅仅存储了B、C和D与A相连的信息，还存储了与之相反的情况。

我们还在存储A、E、F、G和H不与A相连的信息。如果我们存储了所有相连的节点，通过这些我们也可以推断出哪些节点不相连。在我看来，这些零是冗余信息，会导致额外的内存消耗。大多数现实世界中的图是稀疏的，也就是说，连接的数量与可能的连接总数相比非常少。因此，大多数情况下会有太多的零和很少的一。想想看。假设我们试图在邻接矩阵中存储像Facebook这样的社交网络中的连接，在我看来，这将是最不切实际的做法。

但无论如何，为了便于讨论，假设我们正尝试这么做。仅存储一个用户的社交关系，就需要一个大小为10的9次方的行向量或一维矩阵。在社交网络中，平均而言，一个人拥有的好友数量不会超过1000个。如果我拥有1000个好友，那么在存储我社交关系的行向量中，将仅有1000个位置标记为1，其余位置（即10的9次方减去1000）都将标记为0。

我并不是要强迫你同意我的观点，但如果你也和我一样认为这些零在存储冗余信息并额外消耗内存，那么即使我们仅用一个字节以布尔值的形式存储这些1和0，这里的这么多零几乎会占用1GB内存，而1仅占1KB。因此，面对这个问题，让我们尝试做些不同的改变。我们只需保留这些节点相连的信息，而舍弃不相连的信息，因为这些信息是可以被推断和推导出来的。有几种方法可以实现这一点。为了存储a的连接关系，与其使用数组（其中索引代表边的端点，特定索引处的值表示是否有边终止于该点），我们只需简单地保留一个所有相连节点的列表即可。

这是与节点a相连的节点列表或集合。我们可以使用索引或节点的实际名称来表示这个列表。为了方便起见，我们直接使用索引，因为名称可能很长且占用更多内存。你可以随时查看顶点列表并在常数时间内找到对应的名称。在计算机中，我们可以将这组节点（本质上是一组整数）存储在像数组这样简单的数据结构中。正如你所见，这个数组的排列方式与我们之前的数组有所不同。

在我们之前的安排中，索引代表图中某个节点的位置，而值则表示与该节点是否存在连接。在这里，索引不再具有任何意义，值则直接代表我们所连接节点的实际索引。现在，除了使用数组来存储这组整数外，我们也可以使用链表来实现。

为什么只能是数组或链表呢？我认为这里也可以使用树结构。事实上，二叉搜索树是存储一组值的绝佳方式。我们有方法可以保持二叉搜索树的平衡，如果你始终维持二叉搜索树的平衡状态，就能以节点数量的对数时间复杂度完成搜索、插入和删除这三种操作。我们稍后会讨论这些可能方式的操作成本。现在，我只想说有很多方法可以存储节点的连接。对于我们一开始的例子图，我们可以尝试这样做，而不是使用邻接矩阵。

我们仍然存储着相同的信息。我们仍然表示第0个节点连接到第1、第2和第3个节点。第1个节点连接到第0、第4和第5个节点。第2个节点连接到第0和第6个节点，以此类推。但在这里我们消耗的内存要少得多。从编程角度来看，这个邻接矩阵只是一个8x8的二维数组。因此，我们总共消耗了64个单位的空间。

但这种右侧结构中的行并非大小相同。你认为我们该如何通过编程创建这样的结构呢？这取决于具体情况。在C或C++中，如果你理解指针的概念，我们可以创建一个大小为8的指针数组，每个指针可以指向一个大小不同的一维数组。第0个指针可以指向一个大小为3的数组，因为第0个节点有3个连接，我们需要一个大小为3的数组。第1个指针可以指向一个大小为3的数组，因为第1个节点也有3个连接。然而，第2个节点只有2个连接。因此，第2个指针应指向一个大小为2的数组。我们可以依此类推。



The 7th node has 4 connections. So, 7th pointer should point to an array of size 4. If you do not understand any of this pointer thing that I am doing right now, you can refer to myCodeSchool's lesson titled Pointers and Arrays, the link to which you can find in the description of this video. But think about it.

The basic idea is that each row can be a one-dimensional array of different size. And you can implement this with whatever tools you have in your favorite programming language. Now, let's quickly see what are the pros and cons of this structure in the right in comparison to the matrix in the left.We are definitely consuming less memory with this structure in right. With adjacency matrix, our space consumption is proportional to square of number of vertices. While with this 2nd structure, space consumption is proportional to number of edges.

And we know that most real world graphs are sparse. That is, the number of edges is really small in comparison to square of number of vertices. Square of number of vertices is basically total number of possible edges.And for us to reach this Every node should be connected to every other node. In most graphs, a node is connected to few other nodes and not all other nodes. In this second structure, we are avoiding this typical problem of too much space consumption in an adjacency matrix by only keeping the ones and getting rid of the redundant zeros.

Here, for an undirected graph like this one, we would consume exactly 2 into number of edges units of memory and for undirected graph, we would consume exactly E, that is number of edges units of memory. But all in all, space consumption will be proportional to number of edges or in other words, space complexity would be OE. So, the second structure is definitely better in terms of space consumption.

But let's now also try to compare these two structures for time cost of operations. What do you think would be the time cost of finding if two nodes are connected or not? We know that it's constant time or big O of 1 for an adjacency matrix because if we know the start and end point, we know the cell in which to look for 0 or 1. But in the second structure, we cannot do this. We will have to scan through a row.

So, if I ask you something like, can you tell me if there is a connection from node 0 to 7, then you will have to scan this 0th row. You will have to perform a linear search on this 0th row to find 7. Right now, all the rows in this structure are sorted. You can argue that I can keep all the rows sorted and then I can perform a binary search which would be a lot less costlier.That's fine. But if you just perform a linear search, then in worst case, we can have exactly V, that is number of vertices cells in a row. So, if we perform a linear search, in worst case, we will take time proportional to number of vertices.

And of course, the time cost would be big O of log V if we would perform a binary search. Logarithmic run times are really good. But to get this here, we always need to keep our rows sorted.

Keeping an array always sorted is costly in other ways and I'll come back to it later. For now, let's just say that this would cost us big O of V. Now, what do you think would be the time cost of finding all nodes adjacent to a given node? That is finding all neighbors of a node. Well, even in case of adjacency matrix, we now have to scan a complete row.

So, it would be big O of V for the matrix as well as this second structure here. Because here also, in worst case, we can have V cells in a row equivalent to having all 1's in a row in an adjacency matrix. When we try to see the time cost of an operation, we mostly analyze the worst case.

So, for this operation, we have big O of V for both. So, this is the picture that we are getting. Looks like we are saving some space with this second structure but we are not saving much on time.

Well, I would still argue that it's not true. When we analyze time complexity, we mostly analyze it for the worst case. But what if we already know that we are not going to hit the worst case? If we can go back to our previous assumption that we are dealing with a sparse graph, that we are dealing with a graph in which a node would be connected to few other nodes and not all other nodes, then this second structure will definitely save us time.

Things would look better once again if we would analyze them in context of a social network. I'll set some assumptions. Let's say we have a billion users in our social network and the maximum number of friends that anybody has is 10,000.

And let's also assume computational power of our machine. Let's say our machine or system can scan or read 10 to the power 6 cells in a second. This is a reasonable assumption because machines often execute a couple of millions instructions per second.

Now, what would be the actual cost of finding all nodes adjacent to a given node in a adjacency matrix? Well, we will have to scan a complete row in the matrix that would be 10 to the power 9 cells because in a matrix we would always have cells equal to number of vertices. And if we would divide this by a million, we would get the time in seconds. To scan a row of 10 to the power 9 cells, we would take 1000 seconds, which is also 16.66 minutes.

This is unreasonably high. But with the second structure, maximum number of cells in a row would be 10,000 because the number of cells would exactly be equal to number of connections. And this is the maximum number of friends or connections a person in the network has.

So here, we would take 10 to the power 4 upon 10 to the power 6, that is 10 to the power minus 2 seconds, which is equal to 10 milliseconds. 10 milliseconds is not unreasonable. Now, let's try to deduce the cost for the second operation, finding if two nodes are connected or not.

In case of adjacency matrix, we would know exactly what cell to read. We would know the memory location of that specific cell and reading that one cell would cost us 1 upon 10 to the power 6 seconds, which is 1 microsecond. In the second structure, we would not know the exact cell.

We will have to scan a row. So once again, maximum time taken would be 10 milliseconds, just like finding adjacent nodes. So now, given this analysis, if you would have to design a social network, what structure would you choose? No brainer, isn't it? Machine cannot make a user wait for 16 minutes.

Would you ever use such a system? Milliseconds is fine, but minutes, it's just too much. So now, we know that for most real world graphs, this second structure is better because it saves us space as well as time. Remember, I'm saying most and not all because for this logic to be true, for my reasoning to be valid, graph has to be sparse.

Number of edges has to be significantly lesser than square of number of vertices. So now, having analyzed space consumption and time cost of at least two most frequently performed operations, looks like this second structure would be better for most graphs. Well, there can be a bunch of operations in a graph and we should account for all kinds of operations.

So before making up my mind, I would analyze cost of few more operations. What if after storing this example graph in computer's memory in any of these structures, we decide to add a new edge? Let's say we got a new connection in the graph from A to G, then how do you think we can store this new information, this new edge in both these structures? The idea here is to assess that once the structures are created in computer's memory, how would we do if the graph changes? How would we do if a node or edge is inserted or deleted? If a new edge is inserted in case of an adjacency matrix, we just need to go to a specific cell and flip the 0 at that cell to 1. In this case, we would go to 0th row and 6th column and overwrite it with value 1. And if it was a deletion, then we would go to a specific cell and make the 1 0. Now how about this second structure? How would you do it here? We need to add a 6 in the first row. And if you have followed this series on data structures, then you know that it's not possible to dynamically increase size of an existing array.

This would not be so straightforward. We will have to create a new array of size 4 for the 0th row. Then we will have to copy content of the old array, write the new value, and then wipe off the old one from the memory.

It's tricky implementing a dynamic or changing list using arrays. This creation of new array and copying of old data is costly. And this is the precise reason why we often use another data structure to store dynamic or changing lists.And this another data structure is linked list. So why not use a linked list? Why can't each row be a linked list? Something like this. Logically, we still have a list here.

But concrete implementation wise, we are no more using an array that we need to change dynamically. We are using a linked list. It's a lot easier to do insertions and deletions in a linked list.

Now programmatically to create this kind of structure in computer's memory, we need to create a linked list for each node to store its neighbors. So what we can do is we can create an array of pointers, just like what we had done when we were using arrays. The only difference would be that this time each of these pointers would point to head of a linked list.

That would be a node. I have defined node of a linked list here. Node of a linked list would have two fields, one to store data and another to store address of the next node.

A0 would be a pointer to head or first node of linked list for A. A1 would be a pointer to head of linked list for B. And we will go on like A2 for C, A3 for D and so on. Actually I have drawn the linked lists here in the left but I have not drawn the array of pointers. Let's say this is my array of pointers.Now A0 here, this one is a pointer to node and it points to the head of linked list containing the neighbors of A. Let's assume that head of linked list for A has address 400. So in A0 we would have 400. It's really important to understand what is what here in this structure.

This one A0 is a pointer to node and all a pointer does is store an address or reference. This one is a node and it has two fields, one to store data and another a pointer to node to store the address of next node. Let's assume that the address of next node in this first linked list is 450.

Then we should have 450 here and if the next one is at let's say 500, then we should have 500 in address part of the second node. The address in last one would be 0 or null. Now this kind of structure in which we store information about neighbors of a node in a linked list is what we typically call an adjacency list.

What I have here is an adjacency list for an undirected unweighted graph. To store a weighted graph in an adjacency list, I would have one more field in node to store weight. I have written some random weights next to the edges in this graph and to store this extra information, I have added one extra field in node, both in logical structure and the code.

Alright, now finally with this particular structure that we are calling adjacency list, we should be fine with space consumption. Space consumed will be proportional to number of edges and not to square of number of vertices. Most graphs are sparse and number of edges in most cases is significantly lesser than square of number of vertices.

Ideally for space complexity, I should say O of number of edges plus number of vertices because storing vertices will also consume some memory. But if we can assume that number of vertices will be significantly lesser in comparison to number of edges, then we can simply say O of number of edges. But it's always good if we do the counting right.

Now for time cost of operations, the argument that we were earlier making using a sparse graph like social network is still true. Adjacency list would overall be better than adjacency matrix. Finally, let's come back to the question, how flexible are we with this structure if we need to add a new connection or delete an existing connection and is there any way we can improve upon it? Well, I leave this for you to think but I'll give you a hint.

What if instead of using a linked list to store information about all the neighbors, we use a binary search tree? Do you think we would do better for some of these operations? I think we would do better because the time cost for searching, inserting and deleting a neighbor would reduce. With this thought, I'll sign off. This is it for this lesson.

Thanks for watching.