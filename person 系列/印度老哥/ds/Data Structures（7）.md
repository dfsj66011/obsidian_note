
## 32、二叉树的遍历-广度和深度优先

在本课中，我们将讨论二叉树的遍历。当我们处理树结构时，经常需要访问树中的所有节点。现在，树不像数组或链表那样是一种线性数据结构。在线性数据结构中，会有一个逻辑起点和逻辑终点。因此，我们可以从一个端点开始移动指针，并持续向另一个端点移动。

对于像链表这样的线性数据结构，每个节点或元素只有一个下一个元素。但树不是线性数据结构。我这里画了一棵二叉树，这次数据类型是字符，我在节点中填入了这些字符。

对于任意时刻的树来说，如果我们指向某个特定节点，就可能存在多个可能的方向，即存在多个可能的下一节点。以这棵二叉树为例，如果我们从根节点开始指向，就有两个可能的方向。从F出发，既可以向左走到D，也可以向右走到J。当然，如果我们选择了一个方向前进，之后必然需要以某种方式返回并探索另一个方向。

因此，树的遍历并不那么简单直接。本节课我们将要讨论的是树的遍历算法。树遍历可以正式定义为按照某种顺序访问树中每个节点且仅访问一次的过程。所谓访问一个节点，指的是读取或处理该节点中的数据。在本节课中，访问意味着打印节点中的数据。根据访问节点的顺序，树遍历算法大致可以分为两类。

我们可以选择广度优先，也可以选择深度优先。广度优先遍历和深度优先遍历是遍历或搜索图的通用技术。图是一种数据结构，在本系列中我们尚未讨论过图。我们将在后续课程中讨论图。现在你只需要知道树只是图的一种特殊形式。本节课我们将讨论树的广度优先遍历和深度优先遍历。

在树的广度优先遍历中，我们会先访问同一深度或层级的所有节点，然后再访问下一层级的节点。以这棵二叉树为例，值为f的根节点位于第0层（此处标记为L0）。节点的深度定义为从根节点到该节点的路径上的边数，因此根节点的深度为0。图中D和J这两个节点的深度为1，所以我们称它们位于第1层。

现在这四个节点位于第二层，这三个节点位于第三层。最后，这个值为h的节点位于第四层。因此，在广度优先方法中，我们可以从第零层开始，第零层只有一个节点，即根节点。这样我们就可以访问根节点，我会在访问时写下节点中的值。现在第零层已经完成。接下来我可以进入第一层，从左到右访问节点。因此在F之后，我们会访问D，然后是J。现在第一层就完成了。接着我们可以进入第二层，依次访问B、E、G和K。然后我们可以进入第三层，访问A、C和I。最后我可以进入第四层。这种树的广度优先遍历在树的情况下被称为层序遍历。

我们将讨论如何通过编程实现这一点。但这是访问节点的顺序，我们会从左到右逐层进行。在广度优先方法中，对于任何节点，我们会先访问其所有子节点，然后再访问其孙节点。在这棵树中，我们首先访问F，然后访问D，接着不会沿着深度方向访问D的任何子节点，比如B或E。接下来我们会访问J。但在深度优先的方法中，如果我们访问一个子节点，我们会先完整遍历该子节点的整个子树，然后再访问下一个子节点。在这个例子中，从根节点F出发，如果我们向左访问D，那么我们应该先完整遍历这个左子树的所有节点，也就是要彻底完成这个左子树的深度遍历。

换句话说，我们应该沿着这条路径完成F的所有孙子节点，然后再去F的右子节点J。而当我们再次到达J时，我们将沿着这条路径访问所有的孙子节点。因此，基本上我们将访问完整的右子树。在深度优先的方法中，访问左子树、右子树和根节点的相对顺序可能会有所不同。例如，我们可以先访问右子树，然后访问根节点，再访问左子树。或者我们可以采取另一种方式，比如先访问根节点，然后访问左子树，最后访问右子树。因此，相对顺序可以有所不同。

但深度优先策略的核心思想是，访问一个子节点就意味着访问该路径下的整个子树。记住，访问一个节点就是读取、处理或打印该节点中的数据。根据左子树、右子树和根节点的相对顺序，有三种常见的深度优先策略。

一种方法是我们可以先访问根节点，然后访问左子树，再访问右子树，左右子树将以同样的方式递归访问。这种遍历称为前序遍历。另一种方法是我们可以先访问左子树，然后访问根节点，最后访问右子树。这种遍历方式被称为中序遍历。如果根节点在左子树和右子树之后被访问，那么这种遍历方式被称为后序遍历。总共有六种可能的左、右和根节点的排列组合。

但按照惯例，左子树总是先于右子树被访问。所以我们采用了这三种遍历策略。唯一变化的是根节点的位置：如果根节点位于左右子树之前，就是前序遍历；如果根节点位于左右子树之间，就是中序遍历；如果根节点位于左右子树之后，就是后序遍历。记住这三种深度优先算法有一个简单的方法。我们可以用字母D表示访问节点或读取该节点中的数据，用L表示进入左子树，用R表示进入右子树。因此，如果我们用D表示数据，L表示左，R表示右，那么在先序遍历中，对于每个节点，我们将按照DLR的顺序进行。首先，我们将读取该节点中的数据，然后进入左子树。

一旦左子树完成，我们将向右移动。在中序遍历中，首先我们会完成左子树，然后读取当前节点的数据，接着向右移动。在后序遍历中，对于每个节点，我们首先会向左移动。一旦左子树完成，我们将向右移动。然后我们将读取当前节点的数据。前序遍历的顺序是数据、左、右；中序遍历的顺序是左、数据、右；后序遍历的顺序是左、右、数据。使用递归来实现前序、中序和后序遍历非常简单直观。但我们稍后再讨论具体实现。现在让我们看看我在这里画的这棵树的前序、中序和后序遍历结果会是什么。

我们先来看看这棵二叉树的前序遍历是怎样的，需要从根节点开始。对于每个节点，我们首先需要读取数据，或者说访问该节点。实际上，这里我们不说DLR，也可以说VLR，这里的V代表访问（visit），我们可以使用这些假设中的任何一种——V代表访问或D代表数据，在这里我将使用DLR。

那么让我们从根节点开始。对于每个节点，我们首先需要读取数据。我在这里写下了F，也就是我刚读取的数据。然后我需要向左遍历，完成整个左子树。只有当左子树的所有节点都被访问过后，我才能转向右子树。这个问题实际上是以自相似或递归的方式在逐步简化。

现在我们需要关注这个左子树。现在我们位于F的左子树的根节点D。再次强调，对于这个节点，我们首先读取数据。现在我们可以向左走，只有当这三个节点A、B和C处理完毕后，我们才会前往E。现在我们专注于由这三个节点组成的子树。现在我们位于B，可以读取数据。然后我们可以向左走到A，A的左侧没有内容。因此可以说，对于A的左子树已经处理完毕。

右边也没有任何东西。所以我们可以说右边也完成了。现在对于B来说，左子树已经完成。所以我们可以向右移动到C，而C的左右都是空的。现在对于D来说，左子树已经完成。所以我们可以向右移动。对于E来说，左右两边再次为空。现在到了F的阶段，左子树已经遍历完毕，所以我们可以转向右边。接下来我们需要访问J的左侧，而G的左侧已经没有节点，因此可以转向右侧。现在我们可以转向I的左侧。对于H来说，左右两边都没有节点。此时，I的左子树已经完成遍历。

而右子树为空。现在我们可以回到 J，J 的左子树已经处理完毕。因此我们可以处理它的右子树。最后，我们在这里有 K，所有节点都已处理完毕。这就是我们手动执行前序遍历的方式。实际实现将是一个简单的递归过程。我们稍后再讨论这个问题。现在让我们来看看这棵二叉树的中序遍历会是什么样子。在中序遍历中，我们会先访问完左子树，然后访问当前节点，最后再访问右子树。

我们再次从根部开始，首先向左走。现在我们将先完成这个子树。对于D节点，我们再次首先向左走到B，然后从B走到A。对于A节点，左边已经没有内容了。所以我们可以说这个节点的左子树已经遍历完毕。现在我们可以读取数据了。接着我们可以转向它的右侧。

我们再次从根节点开始，首先向左走。现在我们将先完成这个子树。对于D节点，我们再次先向左走到B节点，然后从B节点走到A节点。现在对于A节点，左边已经没有节点了。所以我们可以说这个节点的左子树已经遍历完成。然后我们可以读取数据。现在我们可以向右走了。

但右边也没有东西。所以这个节点就完成了。现在B的左子树完成了，我们可以读取数据。现在对于B我们可以向右走。对于C再次检查，左边没有东西了，所以我们可以读取数据。而且右边也没有东西。现在D的左边已经完全处理完毕，我们可以访问它并在这里读取数据。现在我们可以向右走到E。对于E来说，左右子树再次为空。在这个阶段，F的左子树已经处理完毕。

这样我们就能读取数据了。现在我们可以移动到F的右侧。如果我们继续这样操作，最终就会得到中序遍历的结果。我这里展示的这棵树实际上是一棵二叉搜索树。对于每个节点来说，左侧节点的值较小，右侧节点的值较大。因此，如果我们按照左子树、当前节点、右子树的顺序打印，就会得到一个有序列表。对二叉搜索树进行中序遍历会给你一个有序列表。

好的，现在你应该能理解后序遍历了。这就是后序遍历的结果。我留给你自己验证这个结果是否正确。今天就讲到这里。下节课我们将讨论这些树遍历算法的实现。感谢观看。

## 32、层序遍历

在本课程中，我们将编写代码实现二叉树的层序遍历。正如我们在上一课讨论过的，层序遍历会先访问树中某一特定深度的所有节点，然后再访问下一更深层次的节点。以我展示的这个二叉树为例，如果要按层序遍历并打印节点数据，我们将按照这样的顺序进行。

我们将从第零层开始，打印F，现在第零层已经完成，我们可以进入第一层，从左到右访问第一层的节点。从F我们将到D，从D我们将到J。现在第一层完成，我们可以进入第二层。所以我们将依次访问B、E、G，然后是K，现在我们可以进入下一层。

首先访问节点A、C、I，最后在H节点完成。这是我们应当访问节点的顺序，但问题在于如何在程序中按此顺序访问节点。与链表不同，我们无法仅靠一个指针不断移动来实现。假设我从根节点开始，用一个名为current的指针指向当前访问的节点，那么通过这个指针可以从F节点移动到D节点，因为它们之间存在链接。

因此，我可以从左边走到D，但从D无法到达J，因为D和J之间没有连接。唯一能到达J的路径是从F出发，而一旦我们将指针移动到D，就无法再回到F，因为D到F没有反向链接。那么，我们该如何按层级顺序遍历这些节点呢？显然，仅靠一个指针是无法实现的。我们可以采取的方法是：在访问一个节点时，将其所有子节点的引用或地址保存在一个队列中，以便稍后访问。队列中的节点可以称为已发现节点——我们知道其地址，但尚未访问。

最初我们可以将队列中根节点的地址作为起点，这意味着最初这是唯一被发现的节点。假设在这个示例树中，根节点的地址是400，我也会为其他节点随机假设一些地址。我会用黄色标记已发现的节点。好的，一开始我将根节点入队，而将节点存入队列时，我指的是将节点的地址存入队列。所以最初我们从一个已发现的节点开始。现在，只要队列中至少有一个已发现的节点，也就是说只要队列不为空，我们就可以从队列前端取出一个节点，访问它，然后将它的子节点入队。

对我们来说，访问一个节点就是打印该节点的值，所以我在这里写下F，现在我要把这个根节点的子节点入队。首先我将左子节点入队，然后是右子节点。我会用另一种颜色标记已访问的节点。好的，现在我们有一个已访问的节点和两个已发现的节点。接下来，我们可以再次从队列前端取出节点，访问它并将其子节点入队。通过使用队列，我们在这里实现了两个目的。首先，当我们从一个节点移开时，不会丢失对其子节点的引用，因为我们存储了这些引用。其次，由于队列是先进先出的结构，最先发现的节点（即最先插入的节点）将最先被访问，这样我们就能得到期望的顺序。

仔细想想，这并不难理解。好了，现在我们可以从队列中取出地址为200的这个节点并进行访问，同样地，在离开这个节点之前，我需要将它的子节点加入队列。那么现在，我们有两个已访问的节点，三个已发现的节点，还有六个未被发现的节点。接下来，我们可以从队列前端取出下一个节点，访问它，并将它的子节点加入队列。如果我们继续这样下去，所有节点都将按照我们期望的顺序被访问。此时我们可以将120号节点出队，访问它并将其子节点入队。因此，我们将继续这样操作，直到所有节点都被访问且队列为空。在B之后，我们将在这里得到E，这次不会有任何节点进入队列。接下来我们将在这里得到G，而I的地址将进入队列。现在K将被访问。

现在，在这个阶段，我们引用了队列中的三个节点。接下来，我们将访问值为A的320节点，然后是C节点，接着我们会打印I以及值为H的节点，该数据为H的节点将被加入队列。最后，我们将访问这个节点，至此所有节点都已处理完毕，队列为空。一旦队列为空，我们就完成了遍历。这就是二叉树的层序遍历算法。正如你所看到的，在这种方法中，我们随时在内存中保留一堆地址，放在队列里，而不是仅用一个指针来移动。

所以我们当然会使用很多额外的内存，稍后我会讨论这个算法的空间复杂度。希望你已经理解了代码的逻辑。现在让我们为这个算法编写代码。我打算在这里写C++代码。这是我为二叉树定义节点的方式。我这里有一个包含三个字段的结构体：一个用于存储数据，这次的数据类型是字符，因为在我们之前展示的示例树中数据类型是字符；另外还有两个字段是指向节点的指针，一个用于存储左子节点的地址，另一个用于存储右子节点的地址。

现在我要做的是编写一个名为level order的函数，它应该以根节点的地址作为参数，并按层级顺序打印节点中的数据。为了测试这个函数，我需要编写大量代码来创建和插入二叉树的节点。我还需要编写更多的函数。我会跳过编写所有代码。你可以从前面的课程中选取创建和插入的代码。我要写的只是这个函数level order。现在在这个函数中，我会先处理一个边界情况。如果树是空的，也就是说如果根节点是null，我们可以直接返回；否则如果树不为空，我们需要创建一个队列。我这里不打算自己实现队列。

在C++中，我们可以使用标准模板库中的队列。首先，我们需要编写类似"#include \<queue\>"的语句，然后就可以创建任意类型的队列。在这个函数中，我将通过这样的语句创建一个指向节点的指针队列。正如我们之前讨论过的，最初我们会从队列中的一个已发现节点开始。

最初我们唯一知道的节点是根节点。通过语句 `queue.push(root)`，我将根节点的地址插入队列中。现在我将运行这个 while 循环，条件是队列不应为空。这里我的意思是，只要至少有一个已发现的节点，我们就应该进入循环。在循环内部，我们应该从队列前端取出一个节点。

该函数front返回队列前端的元素，由于数据类型是指向节点的指针，因此我将此函数的返回值收集在这个名为current的节点指针中。现在，我可以访问由current指向的这个节点，所谓访问即读取该节点中的数据。我将简单地打印这些数据，现在我们希望将该节点的子节点地址推入队列中。

所以我要说的是，如果左子节点不为空，就将其插入队列；同样地，如果右子节点不为空，也将其推入队列，或者更准确地说，将其地址推入队列。我还要写一条语句来移除队列前端的元素。调用front并不会将元素从队列中移除，而调用pop则会移除元素。好了，这就是C++中层次遍历的实现。

你可以在视频描述中找到源代码的链接，那里还可以找到用于测试此功能的所有额外代码。现在我们来讨论层序遍历的时间和空间复杂度。如果树中有n个节点，在该算法中访问一个节点意味着读取该节点的数据并将其子节点插入队列，那么访问一个节点将花费恒定时间，并且每个节点将被精确访问一次。

因此，所需时间将与节点数量成正比，换句话说，我们可以说时间复杂度是O(n)。无论树的形状如何，在所有情况下，层序遍历的时间复杂度都是O(n)。现在我们来谈谈空间复杂度。空间复杂度，正如我们所知，是衡量随着输入规模增长而额外使用的内存增长速度的指标。在这个算法中，我们并没有使用固定数量的额外内存。

在执行该算法时，我们会有一个动态增减的队列。假设队列是动态的，那么所使用的最大额外内存量将取决于队列在任何时刻的最大元素数量。我们可能会遇到以下几种情况。在某些情况下，使用的额外内存会较少，而在其他情况下则会较多。对于像这样每个节点只有一个子节点的树结构，队列中任何时刻最多只会有一个元素。每次访问时，会从队列中取出一个节点，并插入一个新节点。

因此，额外占用的内存量不会取决于节点的数量。空间复杂度将是O(1)。但对于这样的树来说，额外使用的内存量将取决于树中的节点数量。这是一棵完美二叉树。所有层级都已填满。如果按照算法执行，每个层级的节点都会在某一时刻进入队列。在完美二叉树中，最深层级的节点数为n除以2。

因此，队列中的最大节点数至少为n除以2。基本上，额外使用的内存与节点数n成正比。因此，在这种情况下，空间复杂度将是O(n)。我不会证明这一点，但对于平均情况，空间复杂度也将是O(n)。因此，在最坏和平均情况下，空间复杂度都是O(n)。当我们在这里讨论最佳、平均和最差情况时，仅针对空间复杂度而言。在所有情况下，时间复杂度都是O(n)。这就是层序遍历的时间和空间复杂度分析。

今天就到这里。下节课我们将讨论深度优先遍历算法，包括前序、中序和后序遍历。本节课就到这里，感谢观看。


## 33、前序、中序和后序遍历

在上一节课中，我们讨论了二叉树的层序遍历，也就是广度优先遍历。而本节课，我们将讨论这三种深度优先算法：前序、中序和后序遍历。

我在这里画了一棵二叉树，节点中填充的数据类型是字符。正如我们在之前的课程中讨论过的，在二叉树的深度优先遍历中，如果我们沿着一个方向前进，就会访问该方向上的所有节点。换句话说，我们会访问该方向上的完整子树。然后我们才会转向其他方向。在我画的这棵示例树中，如果我位于根节点并选择向左走，那么我将访问这个左子树中的所有节点。之后我才能向右走。

再一次，当我向右走时，我会访问这个右子树中的所有节点。如果你能看出这种方法，我们正在以一种自我相似或递归的方式减少问题，我们可以说，总的来说，访问树中的所有节点就是访问根节点、访问左子树和访问右子树。记住，访问一个节点意味着读取或处理该节点中的数据。

而访问子树，指的是访问子树中的所有节点。在深度优先策略中，访问左子树、右子树和根节点的相对顺序可以不同。例如，我们可以先访问右子树，然后访问根节点，最后访问左子树。或者我们可以先访问根节点，然后访问左子树，最后访问右子树。按照惯例，左子树总是先于右子树被访问。在这种约束下，我们会有三种排列方式：可以先访问根节点，然后访问左子树，最后访问右子树。

这种遍历方式被称为前序遍历。或者，我们可以先访问左子树，然后访问根节点，最后访问右子树，这种遍历方式被称为中序遍历。我们还可以先访问左子树，再访问右子树，最后访问根节点，这种遍历方式被称为后序遍历。左右子树将以与原始树相同的方式递归访问。因此，在前序遍历中，对于子树，我们再次按照根、左、然后右的顺序进行。在中序遍历中，我们会继续按照左、根、然后右的顺序进行。

这些算法的实际实现非常简单直观。让我们先看看前序遍历的代码。我首先在这里用文字描述了算法。在前序遍历中，我们需要先访问根节点和左子树，然后是右子树。现在我想编写一个函数，该函数应接收指向根节点的指针或引用作为参数，并按前序打印所有节点的数据。对我们来说，访问一个节点就是打印该节点中的数据。

在C或C++中，我的方法签名会像这样。这个函数将把根节点的地址作为参数。参数类型是指向节点的指针。我会将节点定义为一个包含三个字段的结构体，像这样。这个定义中的数据类型是字符型，还有两个字段用来存储左右子节点的地址。现在在预序遍历函数中，我会首先访问或打印根节点中的数据。

现在我将进行一次递归调用来访问左子树。我在这里进行了递归调用，并向这个调用传递了当前根节点的左子节点的地址，因为左子节点将成为左子树的根。同样地，我也会进行类似的调用来访问右子树。

我们还需要在这个函数中添加一个条件，然后就可以完成了。我们不能无限递归下去，必须有一个基本条件来终止递归。如果一棵树或子树为空，换句话说，在任何调用中如果根节点为空，我们就可以返回或退出。现在，有了这些代码，我的前序遍历函数就完成了。这段代码在C或C++中可以正常运行。

实际上在C语言中，请确保你写的是"struct space node"而不仅仅是"node"。其余部分都没问题。最好能将这种递归过程可视化。那么现在让我们快速看看，如果把我右边展示的这棵示例树传递给前序遍历函数，它是如何工作的。我会重新绘制这棵树并这样展示。在这里，我把节点描绘成一个包含三个字段的结构体。

假设最左边的单元格用于存储左子节点的地址，中间的单元格用于存储数据，最右边的单元格用于存储右子节点的地址。现在，我们为这些节点分配一些地址。假设根节点的地址是200，其他节点的地址也随机分配。现在，我可以为每个节点填写左和右字段。

众所周知，我们始终随身携带的树的标识就是根节点的引用或地址。这就是我们传递给所有函数的内容。在我们的实现中，我们经常使用一个名为root的节点指针类型的变量来存储根节点的地址。我们可以给这个变量取任何名字，可以叫它root，也可以叫它root_ptr。但这只是一个指针。我这里展示的这个特定块是指向节点的指针。

所有这些包含三个单元格的矩形都是节点。这就是内存中事物的组织方式。现在对于这棵树，假设我们正在调用这个前序遍历函数。我将调用前序遍历函数，传入地址200。对于这次调用，根节点不为空。因此我们不会在函数的第一行返回，而是会继续执行并打印地址200处这个节点的数据。

我将在这里输出所有打印语句的结果。现在这个函数会进行一次递归调用，这个特定函数调用的执行将暂停，只有在递归调用 preorder 150 完成后才会恢复。第二次调用是为了访问这个左子树。本次调用预购150是为了访问节点200的左子节点的左子树地址150。再次强调，对于此调用，根节点不为空。因此，我们将继续打印节点150处的数据D。现在，又将进行一次递归调用。

通过这次前序遍历调用400，我们表示将访问这个子树。再次，我们会打印数据并进行另一次递归调用。现在我们已发起调用访问这个仅含一个节点的特定子树。对于这次调用，我们将打印数据。而对于地址为250的节点，其左子节点地址为零或空，我们将进行前序遍历调用零。

但在这次调用中，我们会直接返回，因为变量 root 中的地址将是空值，我们已经达到了递归的基本条件。对 preorder 0 的调用将结束，preorder 250 将继续执行。现在在这个特定的函数调用中，我们将为节点 250 的右子树再发起一次调用。即使右子节点为空，我们仍会进行另一次递归调用，传入地址零。但这次调用同样会直接返回。此时，对前序遍历250的调用将结束，而对前序遍历400的调用将恢复执行。现在，在前序遍历400的调用中，我们将再次递归调用前序遍历180。通过这次对前序遍历180的调用，我们正在访问这个仅包含一个节点的特定子树。对于这次调用，首先我们会打印数据，然后递归调用前序遍历零。

现在预排序零将直接返回，然后我们将再次调用预排序零来处理180的右子节点。递归将像这样继续进行。在这个过程中，我想讨论一个正在发生的事情。尽管我们在函数中没有显式使用额外的内存，但由于递归的存在，函数调用栈会不断增长。我们在之前的课程中多次讨论过内存管理问题，你可以查看本视频描述中的链接，获取相关课程的详细信息。众所周知，每次函数调用时，我们都会在应用程序内存的栈区分配一定量的内存空间。

当函数调用结束时，这部分分配的内存会被回收。在这个递归执行的阶段，以本例来说，我的调用栈看起来会是这样。我用P作为前序遍历的简写，因为这里空间有限。假设我们从主函数发起了一个调用，传递地址200进行预购。在任何时候，主函数都会位于调用栈的底部，只有位于栈顶的调用才会执行，其他所有调用都会暂停。在程序执行过程中，调用栈会不断增长和收缩，因为要为新的函数调用分配内存。而当函数调用结束时，这些内存就会被回收。

因此，尽管我们在这里没有显式地使用任何额外的内存，但我们隐式地在调用栈中使用了内存。所以空间复杂度——即衡量随着输入增长而额外使用内存的速率——将取决于调用栈中使用的最大额外内存量。稍后我会再次讨论空间复杂度。现在，让我们回到我正在执行的递归调用。preorder zero 将会完成，而 preorder 180 将会恢复。为执行 preorder zero 分配的内存将被回收。现在对于 preorder 180，两个递归调用都已经完成。所以这个也会结束，即使是 preorder 400。

两次调用都已完成。因此，前序遍历150将继续执行。现在，这个节点将对前序遍历函数进行递归调用，传入其右子节点的地址450。栈内存将被分配用于执行前序遍历450。在这次调用中，我们将首先打印数据，然后分别传入地址零进行两次递归调用，因为位于450的这个节点的两个子节点均为空。两次调用将直接返回，随后前序遍历450将结束。现在预购150也将完成。如果你能看到调用堆栈，它只会增长到我们到达一个叶子节点，即没有子节点的节点，然后它又会开始缩小。由于这种递归，调用堆栈的最大增长将取决于树的最大深度或高度，我们可以说使用的额外空间将与树的高度成正比。

换句话说，该算法的空间复杂度为O(h)，其中h是树的高度。好了，回到递归部分，我们已经完成了前序遍历150。因此，前序遍历200将继续进行。现在我们将调用访问这个特定的子树。在这个调用中，我们将打印J，然后我们将传递地址60进行调用。所以现在我们在访问这个特定的子树。在这里，我们将首先打印G，然后这个家伙会调用preorder zero，它会简单地返回，然后会有另一个调用preorder 500。在这里，我们将打印I，然后我们将进行两次递归调用，每次都传递地址零，因为节点500是一个没有子节点的叶子节点。在这个家伙完成preorder 60后，将继续执行。

现在这个人也将完成，350的预购将恢复。现在我们将调用预购700，这又是一个叶节点。所以K是数据，这个节点将被打印。然后我们将进行两次调用，传递地址零，这将直接返回。现在在这个阶段，所有这些调用都可以完成，我们已经访问了所有节点。最后，我们将返回到预购200的调用者，可能是主函数。

这就是前序遍历的全部内容。希望你已经理解了递归的工作原理。中序遍历和后序遍历的代码会非常相似。在中序遍历中，我的基准条件也是一样的。我会说如果根节点为空，就返回或退出。如果根节点不为空，我首先需要访问左子树。我正在通过这个递归调用访问左子树，然后我需要访问根节点。所以现在我写下这个printf语句来打印数据。现在我可以访问右子树了。这是第二个递归调用，这就是我的中序遍历函数。我在这里画的这个示例树的中序遍历结果将是这样的。这棵特定的二叉树实际上也是一棵二叉搜索树。

对二叉搜索树进行中序遍历会以排序顺序给出树中的元素。好的，现在让我们来编写后序遍历的代码。对于这个函数，同样地，基本情况也是一样的。所以我会说如果根节点为空，就返回或退出。如果根节点不为空，我首先需要访问左子树。所以我进行了这个递归调用，然后是右子树。所以我将再次进行递归调用。现在我可以访问根节点了。对于这个示例树的后序遍历将是这样的。这就是前序、中序和后序遍历的演示。你可以在视频描述中找到所有源代码的链接。现在让我们快速讨论一下这些算法的时间和空间复杂度。

这三种算法的时间复杂度都是大O(n)。可以看到，每个节点对应一个函数调用，我们实际上是在访问该节点并打印其中的数据。因此，运行时间应该与节点数量成正比。有一种更正式、更数学化的方法来证明这些算法的时间复杂度是大O(n)。你可以在视频描述中查看相关链接。

正如我们之前讨论过的，空间复杂度将是O(h)，其中h是树的高度。在最坏情况下，树的高度为n减一。因此，在最坏情况下，这些算法的空间复杂度可能达到O(n)。而在最佳或平均情况下，树的高度为O(log₂n)。因此可以说，在最佳或平均情况下，空间复杂度将是O(log n)。今天就讲到这里。在接下来的课程中，我们将解决一些关于二叉树的问题。感谢观看。

## 34、

在本课程中，我们将解决一个关于二叉树的简单问题，这也是一个著名的编程面试题目。问题是：给定一棵二叉树，我们需要判断这棵二叉树是否是二叉搜索树。我们知道，二叉树是一种每个节点最多可以有两个子节点的树。我这里画的所有树都是二叉树，但并非所有都是二叉搜索树。二叉搜索树是一种二叉树，其中对于每个节点，左子树中所有节点的值都较小（如果允许重复值，也可以说小于或等于），而右子树中所有节点的值都较大。

我们可以将二叉搜索树定义为这样一种递归结构：左子树中的元素必须小于或等于根节点，而右子树中的元素必须大于根节点。这一规则不仅适用于根节点，还必须适用于所有节点。因此，左子树和右子树本身也必须是二叉搜索树。在我展示的这些二叉树中，A和C是二叉搜索树，但B和D不是。

在B中，值为10的根节点，其左子树中的11大于10。而在二叉搜索树中，任何节点的左子树中的所有值都必须小于该节点。在D中，根节点符合条件，根节点的值为5。左子树中有一个较小的值，右子树中有8、9和12三个较大的值。因此，根节点是符合要求的。但对于值为8的这个节点来说，它的左子树中有一个9。所以，这棵树不是二叉搜索树。那么，我们应该如何解决这个问题呢？基本上，我想写一个函数，该函数应该以指向二叉树根节点的指针或引用作为参数。如果这个二叉树是二叉搜索树，函数应该返回true，否则返回false。

我的方法签名在C++中会是这样的。在C语言中，我们没有布尔类型。因此，这里的返回类型可以是int，我们可以返回1表示真，0表示假。我还会在这里写下节点的定义。对于二叉树来说，节点将是一个包含三个字段的结构体：一个用于存储数据，另外两个用于存储左孩子和右孩子的地址。在我这里的节点定义中，数据类型是整型。我们有两个指向节点的指针，用于存储左右子节点的地址。好了，回到这个问题上来，有几种不同的方法，我们将逐一讨论。我要讲的第一个方法比较容易想到，但效率不是很高。

不过我们还是来讨论一下。我们所说的二叉搜索树应该具有这样的递归结构：对于根节点来说，左子树中的所有元素都必须小于或等于它，而右子树中的所有元素都必须大于它。同时，左子树和右子树本身也必须是二叉搜索树。所以，我们只需逐一检查这些条件即可。

我将编写一个名为`is_subtree_lesser`的函数，该函数接收二叉树（或子树）根节点的地址和一个整数值作为参数。如果左子树中的所有元素都小于该值，此函数将返回`true`。类似地，我还将编写另一个名为`is_subtree_greater`的函数，如果子树中的所有元素都大于给定值，该函数将返回`true`。

我刚刚声明了这些函数，稍后再编写它们的函数体。让我们回到这个函数，它用于判断是否为二叉搜索树。在这个函数中，我会先检查左子树的所有元素是否都小于当前根节点的值——通过调用子树检查函数来实现，传入当前根节点的左子节点地址（左子节点就是左子树的根节点）和根节点的数据值。如果左子树的所有元素都小于根节点数据，这个函数调用就会返回true。接下来我要检查的是右子树的所有元素是否都大于根节点的数据值。

这两个条件还不够。我们还需要检查左子树和右子树是否为二叉搜索树。因此，我将在这里再添加两个条件。我已经递归调用了二叉搜索树函数，传入了左子树的地址。我还进行了另一个调用，传入了右子树的地址。如果这四个函数调用——子树是否更小、子树是否更大以及左子树和右子树是否为二叉搜索树——都返回真。

如果这四个检查都通过了，那么我们的树就是一棵二叉搜索树。我们可以返回 true，否则需要返回 false。现在这个函数里我们只漏掉了一件事，那就是没有处理基本情况。如果根节点为空，即树或子树为空，我们可以返回true。这是我们递归的基本情况，此时应该停止。至此，isBinarySearchTree函数已经完成。不过我们还需要编写其子树较小和子树较大的函数，因为它们也是我们逻辑的一部分。这个函数必须是一个通用函数，用于检查给定树中的所有元素是否都小于给定值。我们需要遍历整个树或子树，查看所有节点的值，并将这些值与给定的整数进行比较。

我将首先处理这个函数的基本情况。如果树为空，我们可以返回true，否则需要检查根节点的数据是否小于或等于给定值。同时还需要递归检查当前根节点的左右子树是否具有更小的值。所以我在这里又加了两个条件。我做了两个递归调用，一个针对左子树，另一个针对右子树。如果这三个条件都成立，那就没问题；否则，我们可以返回false。

它的子树大于函数会非常相似。与其编写这两个函数——子树小于和子树大于，我们也可以这样做。我们可以在左子树中找到最大值，并将其与根节点中的数据进行比较。如果子树的最大值较小，那么所有元素都较小；同理，如果子树的最小值较大，则所有元素都较大。对于右子树，我们可以找到最小值。因此，与其编写这两个函数（子树较小和子树较大），我们可以编写类似查找最大值和查找最小值的函数，这样也能满足需求。

这就是我们采用其中一种方法的解决方案。让我们快速在一个示例二叉树上运行这段代码，看看它是如何执行的。我这里画了一棵非常简单的二叉树，实际上它是一棵二叉搜索树。让我们为树中的这些节点假设一些地址。假设根节点的地址是200，我也会为其他节点假设一些随机地址。为了检查这棵二叉树是否是二叉搜索树，我们将调用isBinarySearchTree函数。

我在这里用IBST作为isBinarySearchTree的缩写，因为空间有限。所以我会调用这个函数，可能是从主函数传入地址200，也就是根节点的地址。对于这个函数调用，局部变量中的地址，也就是收集在这个局部变量root中的地址，将会是200。

根节点不为空。Null只是地址0的宏定义。对于这次调用，根节点不为空。因此我们不会在这一行返回true。我们将进入下一步。现在这里我们将调用isSubtreeLesser函数。传入的参数将是左子节点的地址150和节点200中的数据7。

调用函数的执行将会暂停，只有在被调用函数返回后才会恢复。现在在这个对isSubtreeLesser的调用中，root不为空。所以我们不会在第一行返回true。我们会进入下一个if条件。现在在这里，第一个条件是如果root中的数据，这次的root是150，因为这次调用是针对这个左子树，对于这个左子树，root的地址是150。root中的数据是4，比7小。所以第一个条件为真，我们可以进入第二个条件，这是一个递归调用。

本次调用将暂停，我们将进入下一个调用。再次说明，节点在180，1处的数据小于7。因此第一个条件为真，我们将进行递归调用。节点在180处的左子树为空，没有左子节点。所以我们将在第一行返回。这次根节点为空。

这个特定的调用将直接返回 true。现在，在上一个调用中，当根节点为 180 时，if 的第二个条件也为真。因此，我们将对右子树进行另一个调用。同样地，传入的地址将是 0，我们将直接返回 true。现在对于这个调用，如果 subtreeLesser 为 187，所有三个条件都为真。因此，这个调用也可以返回 true，而现在这个 ISL 157 的调用将继续执行。

现在这个家伙会对右子树进行递归调用，而这个家伙在所有操作完成后也会返回true。对于这个调用来说，由于if语句中的所有三个条件都为真，这个家伙同样会返回true。现在isBinarySearchTree函数将继续执行。

在这次调用中，我们已经评估了第一个条件，结果为真。现在，这个函数将再次调用其子树greater，传入右子节点的地址和数值7。最终，这个调用也会返回真。接下来，我们将进行两次递归调用来检查左右子树是否为二叉搜索树。首先会对左子树进行调用。执行过程会继续这样进行，但我想让你注意一点。

在每次调用二叉搜索树函数时，我们会将根节点的数据与左子树中的所有元素进行比较，然后再与右子树中的所有元素进行比较。这个示例树可能非常大。在这种情况下，在第一次调用isBinarySearchTree时，对于这棵完整的树，我们会递归遍历整个左子树，以查看该子树中的所有值是否都小于7，然后我们会遍历右子树中的所有节点，以查看这些值是否大于7。然后在下次调用isBinarySearchTree时，当我们验证这个特定子树是否为二叉搜索树时，我们会递归遍历这个子树，看看值是否小于4，以及遍历另一个子树，看看值是否大于4。


So, all in all, during this whole process, there will be a lot of traversal. Data in nodes will be read and compared multiple times. If you can see, all nodes in this particular subtree will be traversed once in call to isBinarySearchTree for 200 when we will compare value in these nodes with 7 and then these nodes will once again be traversed in call to isBinarySearchTree for 150 when they will be compared with 4. They will be traversed in call to its subtree lesser.

All in all, these two functions, its subtree lesser and its subtree greater are very expensive. For each node, we are looking at all nodes in its subtrees. There is an efficient solution in which we do not need to compare data in a node with data in all nodes in its subtrees and let's see what the solution is.What we can do is, we can define a permissible range for each node and data in that node must be in that range. We can start at the root node with range minus infinity to infinity because for the root node, there is no upper and lower limit and now as we are traversing, we can set a range for other nodes. When we are going left, we need to reset the upper bound.So, for this node at 150, data has to be between minus infinity and 7. Data in left child cannot be greater than data in root. If we are going right, we need to set the lower bound for this node at 300. Range would be 7 to infinity.

7 is not included in the range. Data has to be strictly greater than 7. For this node at 180, the range will be minus infinity to 4. For this node with value 6, lower bound will be 4 and upper bound would be 7. Now, my code will go like this. My function is binary search tree will take two more arguments.

An integer to mark the lower bound or min value and another integer to mark the upper bound or max value and now instead of checking whether all the elements in left subtree are lesser than the data in root and all the elements in right subtree are greater than the data in root or not, we will simply check whether data in root is in this range or not. So, I'll get rid of these two function calls. Its subtree lesser and its subtree greater which are really expensive and I'll add these two conditions.

Data in root must be greater than min value and data in root must be less than max value. These two checks will take constant time. Its subtree lesser and its subtree greater functions were not taking constant time.Running time for them was proportional to number of nodes in the subtree. Okay, now these two recursive calls should also have two more arguments. For the left child, lower bound will not change.Upper bound will be the data in current node and for the right child, upper bound will not change and lower bound will be the data in current node. This recursion looks good to me. We already have the base case written.

The only thing is that the caller of this binary search tree function may only want to pass the address of root node. So, what we can do is instead of naming this function as binary search tree, we can name this function as a utility function like isbstutil and we can have another function named as binary search tree in which we can take only the address of root node and this function can call bst isbstutil function passing address of root minimum possible value in integer variable for minus infinity and maximum possible value in integer variable for plus infinity int min and int max here are macros for minimum and maximum possible values in int. So, this is our solution using second approach which is quite efficient.In this recursion, we will go to each node once and at each node, we will take constant time to see whether data in that node is in a defined range or not. Time complexity would be big O of n where n is number of nodes in the binary tree. For the previous algorithm, time complexity was big O of n square.

One more thing, in this code, I have not handled the case that binary search tree can have duplicates. I am saying that elements in left subtree must be strictly lesser and elements in right subtree must be strictly greater. I leave it for you to see how you will allow duplicates.

There is another solution to this problem. You can perform in-order traversal of binary tree and if the tree is binary search tree, you would read the data in sorted order. In-order traversal of a binary search tree gives a sorted list.

You can do some hack while performing in-order traversal and check if you are getting the elements in sorted order or not. During the whole traversal, you only need to keep track of previously read node and at any time data in a node that you are reading must be greater than data in previously read node. Try implementing this solution, it will be interesting.

Okay, I'll stop here now. In coming lessons, we will discuss some more problems on binary tree. Thanks for watching.

In this lesson, we are going to write code to delete a node from binary search tree. In most data structures, deletion is tricky. In case of binary search trees too, it's not so straight forward.

So, let's first see what all complications we may have while trying to delete a node from binary search tree. I have drawn a binary search tree of integers here. As we know, in a binary search tree, for each node, value of all nodes in its left subtree is lesser and value of all nodes in its right subtree is greater.

For example, in this tree, if I'll pick this node with value 5, then we have 3 and 1 in its left subtree which are lesser and we have 7 and 9 in its right subtree which are greater and you can pick any other node in the tree and this property will be true, else the tree is not a BST. Now, when we need to delete a node, this property must be conserved. Let's try to delete some nodes from this example tree and see if we can rearrange things and conserve this property of binary search tree or not.

What if I want to delete this node with value 19. To delete a node from tree, we need to do two things. We need to remove the reference of the node from its parent, so the node is detached from the tree.

Here, we will cut this link, we will set right child of this node with value 17 as null and the second thing that we need to do is reclaim the memory allocated to the node being deleted, that is wipe off the node object from memory. This particular node with value 19 that we are trying to delete here is a leaf node. It has no children and even if we take this guy out by simply cutting this link, that is removing its reference from its parent and then wiping it off from memory, there is no problem.

Property of binary search tree that for each node, value of nodes in left should be lesser and value of nodes in right should be greater is conserved. So, deleting a leaf node, a node with no children is really easy. In this tree, these four nodes with values 1, 9, 13 and 19 are leaf nodes.

To delete any of these, we just need to cut the link and wipe off the node, that is clear it from memory. But what if we want to delete a null leaf node? What if in this example, we want to delete this node with value 15? I can't just cut this link because if I'll cut this link, we will detach not just the node with value 15 but this complete subtree. We have two more nodes in this subtree, we could have had a lot more.

We need to make sure that all other nodes except the node with value 15 that's being deleted remain in the tree. So, what do we do now? This particular node that we are trying to delete here has two children or two subtrees. I'll come back to case of node with two children later because this is not so easy to crack.

What I want to discuss first is the case when the node being deleted would have only one child. If the node being deleted would have only one child, like in this example, this node with value 7, this guy has only one child. This guy has a right child but does not have a left child.

For such a node, what we can do is, we can link its parent to this only child. So, the child and everything below the child, we could have some more nodes below 9 as well, will remain attached to the tree and only the node being deleted will be detached. Now, we are not losing any other node than the node with value 7. This is my tree after the deletion.

Is this still a binary search tree? Yes, it is. Only the right subtree of node with value 5 has changed. Earlier we had 7 and 9 in right subtree of 5 and now we have 9 which is fine.

What if we were having some more nodes below 9? Here in this tree, I can have a node in left of 9 and the value in this node has to be lesser than 12, greater than 5, greater than 7 and lesser than 9. We are left with only one choice. We can only have 8 here. In right, we can have something lesser than 12 and greater than 5, 7 and 9. All in all, between 9 and 12.

Okay, so if the original tree was this much after deletion, this is how my tree will look like. Okay, so are we good now? Is the tree in right a BST? Well, yes, it is. When we are setting this node with value 9 as right child of the node with value 5, we are basically setting this particular subtree as right subtree of the node with value 5. Now, this subtree is already in right of 5, so value of all nodes in this subtree is already greater than 5 and the subtree itself of course is a binary search tree.

Any subtree in a binary search tree will also be a binary search tree. So, even after deletion, even after the rearrangement, property of the tree that for each node, nodes in left should be lesser and nodes in right should be greater in value is conserved. So, this is what we need to do to delete a node with just one child or a node with just one subtree.

Connect its parent to its only child and then wipe it off from memory. There are only two nodes in this tree that have only one child. Let's try to delete this other one with value 3. All we need to do here is set 1 as left child of 5. Once again, if there were some more nodes below 1, then also there was no issue.Okay, so now we are good for two cases. We are good for leaf nodes and we are good for nodes with just one child. And now we should think about the third case.

What if a node has two children? What should we do in this case? Let's come back to this node with value 15 that we were trying to delete earlier. With two children, we can't do something like connect parent to one of the While trying to delete 15, if we will connect 12 to 13, if we will make 13 the right child of 12, then we will include 13 and anything below 13. That is we will include the left subtree of 15.But we will lose the right subtree of 15. That is 17 and anything below 17. Similarly, if we will make 17 the right child, then we will lose the left subtree of 15.

That is 13 and anything below 13. Actually, this case is tricky. And before I talk about a possible solution, I want to insert some more nodes here.

I want to have some more nodes in subtrees of 13 and 17. The reason I'm inserting some more nodes here is because I want to discuss a generic case. And that's why I want these two subtrees to have more than one node.

Okay, coming back, when I'm trying to delete this node, my intent basically is to remove this value 15 from the tree, my delete function will have signature, something like this, it will take pointer or reference to the root node and value to be deleted as argument. So here, I'm deleting this particular node because I want to remove 15 from the tree. What I'm going to do now is something with which I can reduce case three to either case one or case two, I'll wipe off 15 from this node.And I'll fill in some other value in this node. Of course, I can't fill in any random value. What I'll do is I'll look for the minimum in right subtree of this node.

And I'll fill in that value here. Minimum in right subtree of this node is 17. So I have filled 17 here.We now have two nodes with value 17. But notice that this node has only one child, we can delete this node because we know how to delete a node with one child. And once this node is deleted, my tree will be good.

The final arrangement will be a valid arrangement for my BST. But why minimum in right subtree? Why not value in any other leaf node or any other node with one child? Well, we also need to conserve this property that for each node, nodes in left should have lesser value and nodes in right should have greater value. For this node, if I'm bringing in the minimum from its right subtree, then because I'm bringing in something from its right subtree, it will be greater than the previous value 17 is greater than 15.

So all the elements in left of course will be lesser. And because it's the minimum in right subtree, all the elements in right of this guy would either be greater or equal, we'll have a duplicate that will be equal. Once the duplicate is removed, everything else will be fine.

In a tree or subtree, if a node has minimum value, it won't have a left child. Because if there is a left child, there is something lesser. And this is another property that we are exploiting.

Give this some thought in a tree or subtree node with minimum value will not have a left child, there may or may not be a right child. If we would have a right child like here, we have a right child. So here we are reducing case three to case two.

If there was no child, we would have reduced case three to case one. Okay, so let's get rid of the duplicate. I'll build a link like this.And after deletion, this is what my tree will look like. So this is what we need to do in case three, we need to find the minimum in right subtree of the targeted node, then copy or fill in this value. And finally, we need to delete the duplicate or the node with minimum value from right subtree.

There was another possible approach here. And I must talk about it. Instead of going for minimum in right, we could also go for maximum in left subtree.

Maximum left subtree would of course be greater than or equal to all the values in left. Maximum in left subtree of node with value 15 is 14. I'm copying 14 here.

Now all the nodes in left are lesser than or equal to 14. And because we are picking something from left subtree, it will still be lesser than the value being deleted 14 is less than 15. So all the nodes in this right subtree will still be greater.And if we are picking maximum in a tree or subtree, then that node will not have a right child because if we have something in right, we have something greater. So the value can't be maximum, the node may have a left child. In this case, node with value 14 doesn't have a left child.So we are basically reducing case three to case one, I'll simply get rid of this node. So we are looking good even after deletion. In case three, we can apply any of these methods.And this is all in logic part. Let's now write code for this logic. I'll write c++ and we will use recursion.

If you're not very comfortable applying recursion on trees, then make sure you watch earlier lessons in this series, you can find link to them in description of this video. In my code here, I've defined node as a structure with three fields, we have one field to store data and we have two fields that are pointers to node to store addresses of left and right children. And I want to write a function named delete that should take pointer to root node and the data to be deleted as argument.

And this function should return pointer to root node because the root may change after deletion. What we are passing to delete function is only a local copy of roots address. If the address is changing, we need to return it back.

To delete a given value or data, we first need to find it in the tree. And once we find the node containing that data, we can try to delete it. Remember, the only identity of tree that we pass to functions is address of the root node.And to perform any action on the tree, we need to start at root. So let's first search for the node with this data. First, I'll cover a corner case.

If root is null, that is if the tree is empty, we can simply return, I can say return root or return null here, they will mean the same because root is null. Else, if the data that we are looking for is less than the data in root, then it's in the left subtree. The problem can be reduced to deleting the data from left subtree, we need to go and find the data in left subtree.

So we can make a recursive call to delete function passing address of the left child and the data to be deleted. Now the root of the left subtree that is the left child of this current node may change after deletion. But the good thing is delete function will return address of the modified root of the left subtree.

So we can set the return as left child of the current node. Now if data that we are trying to delete is greater than the data in root, we need to go and delete the data from right subtree. And if the data is neither greater nor lesser, that is if it's equal, then we can try deleting the node containing that data.

Now let's handle the three cases one by one. If there is no child, we can simply delete that node. What I'll do here is I'll first wipe off the node from memory.And this is how I'll do it. What we have in root right now is address of the node to be deleted. I'm using delete operator here that's used to deallocate memory of an object in heap.

In C, you would use free function. Now root is a dangling pointer because the object in heap is deleted, but root still has its address. So we can set root as null.

And now we can return root reference of this node in its parent will not be fixed here. Once this recursive call finishes, then somewhere in these two statements in any of these two statements, in any of these two else ifs, the link will be corrected. I hope this is making sense.

Okay, now let's handle other cases. If only the left child is null, then what I want to do is I first want to store the address of current node that I'm trying to delete in a temporary pointer to node. And now I want to move the root this pointer named root to the right child.

So the right child becomes the root of this subtree. And now we can delete the node that is being pointed to by temp, we will use delete operator. In C, we would be using free function.

And now we can return root. Similarly, if the right child is null, I'll first store the address of current root in a temporary pointer to node, then I'll make the left child new root of the subtree. So we'll move to the left child.

And then I'll delete the previous route, whose address I have in temp. And finally, I'll return root. Actually, we need to return root in all cases.

So I'll remove this return root statement from all these if and else if and write one return root after everything. Let's talk about the third case. Now, in case of two children, what we need to do is we need to search for minimum element in right subtree of the node that we are trying to delete.

Let's say this function find min will give me address of the node with minimum value in a tree or subtree. So I'm calling this function find min and I'm collecting the return in a pointer to node named temp. Now I should set the data in current node that I'm trying to delete as this minimum value.

And now the problem is getting reduced to deleting this minimum value from the right subtree of current node. With this much code, I think I'm done with delete function. This looks good to me.

Let's quickly run this code on an example tree and see if this works or not. I have drawn a binary search tree here. Let's say these values outside these nodes are addresses of the nodes.

Now I want to delete number 15 from this tree. So I'll make a call to delete function passing address of the root which is 200 and 15, the value to be deleted. In delete function for this particular call, control will come to this line.

A recursive call will be made. Execution of this call delete 200 comma 15 will pause and it will resume only after this function below. Delete 350 comma 15 returns.

Now for this call below, we will go inside the third else in case three. Here we will find the node with minimum value in right which is 17 which is 400. The value is 17.

Address is 400. First we will set the data in node at 350 as 17. And now we are making a recursive call to delete 17 from right subtree of 350.

We have only one node in right subtree of 350. Here we have case one. In this call, we will simply delete the node at 400 and return null.

Remember root will be returned in all calls in the end. Now delete 350 comma 15 will resume and in this resumed call, we will set address of right child of node at 350 as null. As you can see the link in parent is being corrected when the recursion is unfolding and the function call corresponding to the parent is resuming.

And now this guy can return. And now in this call, we will resume at this line. So right child of node at 200 will be set as 350.

It already it's already 350. But it will be written again. And now this call can also finish.

So I hope you got some sense of how this recursion is working. You can find link to all the source code and code to test the delete function in description of this video. This is it for this lesson.


ds-16

