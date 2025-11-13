
## 26、二叉树

在上节课中，我们介绍了树形数据结构。我们将树作为一种逻辑模型进行了讨论，并简要介绍了树的一些应用。在本节课中，我们将更详细地讨论二叉树。正如我们在上节课中所看到的，二叉树是一种具有以下特性的树：树中的每个节点最多可以有两个子节点。我们将首先讨论二叉树的一些一般特性，然后可以讨论一些特殊类型的二叉树，比如二叉搜索树，它是一种非常高效的结构，用于存储有序数据。在二叉树中，正如我们所说，每个节点最多可以有两个子节点。

在我画的这棵树中，节点要么没有子节点，要么有两个子节点，但也可以有一个子节点。我在这里又添加了一个节点，现在就有了一个只有一个子节点的节点。由于二叉树中的每个节点最多只能有两个子节点，我们把其中一个子节点称为左子节点，另一个称为右子节点。

对于根节点来说，这个特定节点是左子节点，而这个则是右子节点。一个节点可以同时拥有左子节点和右子节点，这四个节点就都同时具备左右子节点；或者一个节点可以仅拥有左子节点或右子节点中的一种。比如这个节点拥有左子节点，但没有右子节点。

我会在这里再添加一个节点。现在，这个节点有一个右子节点但没有左子节点。在程序中，我们会将左子节点的引用或指针设置为空。因此，我们可以说对于这个节点，左子节点为空；同样地，对于这个节点，我们可以说右子节点为空。对于所有没有子节点的叶子节点（即零个子节点的节点称为叶子节点），我们可以说这些节点的左右子节点都为空。

根据属性，我们将二叉树分为不同的类型。我在这里再画一些二叉树。如果一棵树只有一个节点，那么它也是二叉树。这种结构也是二叉树。这也是二叉树。记住，唯一的条件是节点不能有两个以上的子节点。如果每个节点要么有两个子节点，要么没有子节点，那么这棵二叉树就被称为严格二叉树或真二叉树。我现在展示的这棵树不是严格二叉树，因为有两个节点只有一个子节点。我去掉这两个节点后，现在这就是一棵严格二叉树了。

我们称一棵二叉树为完全二叉树，当且仅当除了最后一层外，其他所有层都被完全填满，并且所有节点都尽可能地向左靠拢。除了最后一层外，其他所有层无论如何都会被填满。因此，如果最后一层没有被完全填满，那么该层的节点必须尽可能地向左靠拢。

目前，这棵树还不是一棵完全二叉树。深度可以称为同一层级的节点。树中的根节点深度为0。节点的深度定义为从根节点到该节点的路径长度。在这张图中，假设深度为0的节点是第0层节点。我可以简单地用L0表示第0层。现在这两个节点位于第1层。这四个节点位于第2层，最后这两个节点位于第3层。树中任何节点的最大深度为3。树的最大深度也等于树的高度。

如果我们像L0、L1、L2这样对树中的各个层级进行编号，那么在某一层级i上，我们最多可以拥有的节点数等于2的i次方。在层级0，我们可以有1个节点。2的0次方是1。然后在层级1，我们最多可以有2个节点。在层级2，我们最多可以有2的2次方个节点，也就是4个。因此，一般来说，在任何层级i上，我们最多可以有2的i次方个节点。

你应该能很清楚地看到这一点，因为每个节点可以有两个子节点。所以，如果我们某一层有x个节点，那么这x个节点中的每一个都可以有两个子节点。因此，在下一层，我们最多可以有2x个子节点。在这个二叉树中，第二层有4个节点，这是第二层的最大值。现在，这些节点中的每一个都可能有两个子节点。我在这里只是画出了箭头。因此，在第三层，我们最多可以有2乘以4，也就是8个节点。

现在，对于一个完全二叉树来说，所有层级都必须完全填满。我们可以对最后一层或最深层给予例外。它不必完全填满，但节点必须尽可能靠左。我这里展示的这棵特定树并不是一个完全二叉树，因为左边有两个空缺位置。我将对这个结构稍作调整。现在，这就是一棵完全二叉树了。

我们可以在第三层拥有更多节点，但左侧不应有空缺位置。我在这里又添加了一个节点，这仍然是一棵完全二叉树。如果所有层都完全填满，这样的二叉树也可以称为完美二叉树。在完美二叉树中，所有层级都会被完全填满。如果h是完美二叉树的高度，记住二叉树的高度是从根节点到任意叶节点的最长路径的长度，或者更准确地说，是从根节点到任意叶节点的最长路径中的边数。二叉树的高度也等于最大深度。

对于这棵二叉树，其高度或最大深度为3。一棵高度为h的树中，最大节点数等于：第0层有2的0次方个节点，第1层有2的1次方个节点，以此类推直到高度h层，最深层将有2的h次方个节点。这个总和等于2的h+1次方减1（这里的h+1代表层数）。也可以表述为2的层数次方减1。在本例中，总层数为4（从L0到L3）。

因此，节点数量，最大节点数量将是2的4次方减1，也就是15。所以，一个完美二叉树在给定高度下会拥有最大可能的节点数量，因为所有层级都会被完全填满。嗯，我应该说是高度为h的二叉树中的最大节点数量。好的，我也可以问你这个问题。一个具有n个节点的完美二叉树的高度是多少？假设n是完美二叉树中的节点数。为了求出高度，我们需要解这个方程：n等于2的h加1次方减1。因为如果高度是h，节点数就是2的h加1次方减1。我们可以解这个方程，结果将是这样的。

记住，这里的n代表节点数量。具体数学推导就留给你自己理解了。树的高度等于以2为底的对数（n加1）再减1。在我展示的这个完美二叉树中，节点数是15。因此n等于15。n加1等于16。那么h就等于以2为底的16的对数减1。以2为底的16的对数是4。所以最终结果是4减1等于3。一般来说，对于完全二叉树，我们也可以用取整函数计算高度：以2为底的n的对数取整数部分。完美二叉树也属于完全二叉树的一种。

这里n是15。以2为底15的对数是3.906891。如果我们取整数部分，那么就是3。我不会深入探讨为什么完全二叉树的高度是以2为底n的对数。我们稍后会尝试理解这一点。所有这些数学知识在我们分析二叉树各种操作的成本时将非常有用。树操作的许多时间复杂度取决于树的高度。例如，在二叉搜索树（一种特殊的二叉树）中，搜索、插入或删除元素的时间复杂度与树的高度成正比。因此，在这种情况下，我们希望树的高度尽可能小。

如果树的密度较高，或者接近完美二叉树或完全二叉树，那么树的高度就会较低。当树是完全二叉树时，具有n个节点的树的最小高度可以是log n（以2为底）。如果我们有这样的排列方式，那么树的高度将达到最大值。对于n个节点，最小可能高度是log₂n的向下取整或整数部分，而最大可能高度是n-1，此时我们会得到一个像这样的稀疏树，几乎等同于链表。现在，思考一下。如果我说某个操作所需的时间与树的高度成正比，或者换句话说，如果某个操作的时间复杂度是O(H)，其中H是二叉树的高度，那么对于一个完全或完美二叉树，我的时间复杂度将是O(log₂n)，而对于这种稀疏树的最坏情况，我的时间复杂度将是O(n)。O(log n)的阶数几乎是可能的最佳运行时间。

当n高达2的100次方时，以2为底的对数log n仅为100。如果采用时间复杂度为n的算法，即便n等于2的100次方，即使使用人类制造过的最强大计算机，我们也需要数年时间才能完成计算。这就是问题的关键所在。我们常常希望将二叉树的高度保持在最小可能值，或者更常见地说，我们努力保持二叉树的平衡。如果对于每个节点，其左右子树的高度差不超过某个数k（通常k为1），我们称该二叉树为平衡二叉树。因此可以说，对于每个节点，其左右子树的高度差不应超过1。关于树的高度，我想补充一点：之前我们将高度定义为从根节点到叶子节点的最长路径上的边数。

仅有一个节点的树的高度为0，该节点本身即为叶节点。我们可以将空树定义为没有节点的树，并认为空树的高度为-1。人们常常将高度计算为从根节点到叶节点的最长路径上的节点数量。在此图中，我绘制了一条从根节点到叶节点的最长路径。这条路径上有3条边。因此，树的高度为3。如果计算路径上的节点数，高度则为4。这看起来非常直观，我在很多地方都见过这种高度的定义。如果计算节点数，仅有一个节点的树的高度等于1，那么我们可以说空树的高度为0。但这不是正确的定义，我们不会采用这种假设。我们将认为空树的高度为-1，而仅有一个节点的树的高度为0。节点左右子树的高度差可以计算为左子树高度减去右子树高度的绝对值，在这个计算中，子树的高度也可以是-1。

对于图中这个叶子节点，其左右子树均为空。因此，左子树高度h_left和右子树高度h_right均为-1。但整体差值仍为0。在完美二叉树中，所有节点的平衡因子差值均为0。我已移除该树的部分节点，现在每个节点旁标注了其平衡因子diff值。这仍是一棵平衡二叉树，因为所有节点的最大平衡因子差值为1。让我们再移除一些节点，此时这棵树不再平衡，因为其中一个节点的平衡因子差值达到2。对于这个特定节点，其左子树高度为1，而右子树高度为-1（因为右子树为空）。

因此，差值的绝对值为2。我们尽量保持树的平衡，以确保其紧张度和高度最小化。如果高度最小化，那么依赖于高度的各种操作的成本也会最小化。好的，接下来我想简单谈谈如何在内存中存储二叉树。

在我们之前的课程中，最常见的一种实现方式是动态创建节点，并通过指针或引用相互连接。对于整型的二叉树，在C或C++中，我们可以这样定义一个节点：数据类型为整型，因此有一个字段用于存储数据，还有两个指针变量，一个用于存储左子节点的地址，另一个用于存储右子节点的地址。这当然是最常见的方式——节点在内存中随机位置动态创建，并通过指针相互链接。但在某些特殊情况下，我们也会使用数组。数组通常用于完全二叉树。我这里画了一棵完美二叉树。

假设这是一棵整数树。我们可以做的是，从根节点开始，按从左到右的顺序逐层为这些节点编号，从0开始。因此，我们会得到0、1、2、3、4、5和6这样的编号。现在，我可以创建一个包含7个整数的数组，这些编号就可以作为这些节点的索引。所以，在第0个位置我会填入2，第1个位置填入4，第2个位置填入1，以此类推。我们已经填满了数组中的所有数据，但如何存储关于链接的信息呢？我们如何知道根节点的左子节点值为4，右子节点值为1？

那么，在完全二叉树的情况下，如果我们这样给节点编号，那么对于索引为i的节点，其左子节点的索引将是2i加1，右子节点的索引将是2i加2。请记住，这仅适用于完全二叉树。对于节点0，当i等于0时，左子节点是2i加1即1，右子节点是2i加2即2。对于节点1，左子节点位于索引3，右子节点位于索引4。当i等于2时，2i加1将是5，2i加2将是6。当我们讨论一种称为堆的特殊二叉树时，我们将详细讨论我们的实现。

数组用于实现堆。今天就讲到这里。下节课我们将讨论二叉搜索树，它也是一种特殊的二叉树，为我们提供了一种非常高效的存储结构，可以快速搜索和更新数据。本节课就到这里。感谢观看。

## 27、二叉搜索树

在上节课中，我们总体讨论了二叉树。现在，在这节课中，我们将讨论二叉搜索树，这是一种特殊的二叉树，它是一种高效的数据结构，可以快速组织和搜索数据，同时也能快速更新。但在开始讲解二叉搜索树之前，我想请大家思考一个问题：你会使用哪种数据结构来存储一个可修改的集合？假设你有一个集合，它可以是任何数据类型的集合。

集合中的记录可以是任何类型。现在，您希望将这个集合以某种结构存储在计算机的内存中，然后希望能够快速搜索集合中的记录，并且还能够修改该集合。您希望能够向集合中插入一个元素或从集合中移除一个元素。

那么，你会使用哪种数据结构呢？你可以选择数组或链表。这两种众所周知的数据结构都能用来存储集合。现在，如果我们使用数组或链表，这些操作（搜索、插入或删除）的运行时间会是多少呢？我们先来讨论数组，为了简单起见，假设我们要存储的是整数。

要存储一个可修改的整数列表或集合，我们可以创建一个足够大的数组，并将记录存储在数组的某一部分。我们可以标记列表的末尾。在我展示的这个数组中，我们有从0到3的整数，记录从0到3，数组的其余部分是可用的空间。

现在，要在集合中搜索某个x，我们必须从索引0开始扫描数组直到末尾，在最坏的情况下，可能需要查看列表中的所有元素。如果n是列表中元素的数量，所需时间将与n成正比，换句话说，我们可以说这个操作的时间复杂度为O(n)。好的，那么插入的成本是多少呢？假设我们想在这个列表中插入数字5。如果有可用的空间，这些黄色的单元格都是可用的，我们可以通过增加这个标记end来添加一个单元格，并填入要添加的整数。

此操作所需时间将是恒定的。运行时间不会依赖于集合中的元素数量。因此，我们可以说时间复杂度为O(1)。好的，那删除操作呢？假设我们想从集合中删除1。我们需要做的是将所有位于1右侧的记录向左移动一位，然后就可以减少end的值。在最坏情况下，删除操作的成本再次为O(n)。在最坏情况下，我们需要移动n-1个元素。如果数组中有可用空间，插入操作的成本将为O(1)。因此，数组必须足够大。如果数组填满了，我们可以创建一个更大的新数组。通常，我们会创建一个大小为已填满数组两倍的数组。

因此，我们可以创建一个新的更大的数组，然后将已填满的数组内容复制到这个新的大数组中。复制操作的时间复杂度为O(n)。我们在之前的课程中已经多次讨论过动态数组这个概念。所以，如果数组未填满，插入操作的时间复杂度为O(1)；如果数组已填满，插入操作的时间复杂度则为O(n)。目前，我们暂且假设数组总是足够大。现在，如果我们使用链表，让我们来讨论这些操作的成本。如果使用链表，我在这里画了一个整数链表。

数据类型可以是任何类型。搜索操作的成本再次为O(n)，其中n是集合中的记录数或链表中的节点数。在最坏的情况下进行搜索时，我们将不得不遍历整个列表。我们需要查看所有节点。在链表中，头部插入的成本是O(1)，尾部插入的成本是O(n)。我们可以选择在头部插入以降低成本。因此，插入的运行时间可以说是O(1)，换句话说，我们将花费恒定的时间。删除操作同样会是O(n)。我们首先需要遍历链表并搜索记录，在最坏的情况下，可能需要查看所有节点。好的，这就是如果我们使用数组或链表进行操作的成本。

插入操作确实很快，但对于像搜索这样的操作，O(n)的时间复杂度表现如何呢？你怎么看？如果我们要搜索一条记录x，在最坏的情况下，我们不得不将这条记录x与集合中的所有n条记录进行比较。假设我们的机器每秒可以执行一百万次比较。那么可以说，这台机器每秒能进行10的6次方次比较。

因此，一次比较的成本将是10的负6次方秒。当今世界的机器处理的是真正庞大的数据。现实世界的数据完全有可能达到1亿或10亿条记录。世界上有许多国家的人口超过1亿。有两个国家的人口超过10亿。如果我们拥有一个国家所有居民的数据，那么记录数很容易就会达到1亿条。

好的，如果我们假设一次比较的成本是10的负6次方秒，那么当n达到1亿时，所需时间将是100秒。对于搜索这种可能频繁执行的操作来说，100秒显然不合理。我们能否做得更好？能否突破O(n)的时间复杂度？实际上，在数组中，如果数据已排序，我们可以采用二分查找算法——其运行时间为O(log n)，这正是最优的时间复杂度。

我在这里画了这个整数数组。数组中的记录已排序。这里的数据类型是整数。对于其他数据类型，特别是复杂数据类型，我们应该能够根据记录的某个属性或键来对集合进行排序。我们应该能够比较记录的键，而且对于不同的数据类型，比较逻辑也会有所不同。例如对于一个字符串集合，我们可能希望记录按字典或字母顺序排序。因此，我们将进行比较，看看哪个字符串在字典顺序中排在前面。现在，这就是我们对二分查找的要求。数据结构应该是一个数组，并且记录必须是有序的。

好的，如果我们使用排序数组，搜索操作的成本可以降到最低，但在插入或删除时，我们必须确保数组之后仍然保持有序。在这个数组中，如果我想在这个阶段插入数字5，我不能简单地把5放在索引6的位置。我需要做的是，首先找到在有序列表中插入5的位置。我们可以使用二分查找在log n的时间复杂度内找到这个位置。

我们可以通过二分查找来找到列表中第一个大于5的整数。这样，我们就能快速定位到该位置。在这个例子中，索引是2，但之后我们需要将所有从该位置开始的记录向右移动一位，然后才能插入5。因此，尽管我们可以在O(log n)的时间内快速找到记录应该插入的位置，但在最坏情况下，这种移位操作会消耗O(n)的时间。所以，插入操作的总体运行时间将是O(n)，同样地，删除操作的成本也是O(n)，因为我们也需要移动一些记录。

好的，当我们使用排序数组时，搜索操作的成本被最小化了。在二分查找中，对于n条记录，最多需要进行以2为底的对数n次比较。因此，如果我们每秒能进行百万次比较，那么当n等于2的31次方（即超过20亿）时，我们只需要31微秒。因为以2为底的2的31次方的对数就是31。

好的，我们现在已经解决了查找的问题。对于任何实际的n值，我们都能很好地处理。但是插入和删除操作呢？它们的时间复杂度仍然是O(n)。我们能否在这方面做得更好呢？如果我们使用一种叫做二叉搜索树的数据结构（我简写为BST），那么这三个操作的时间复杂度都可以降低到O(log n)。在平均情况下，所有操作的时间复杂度都是O(n)。在最坏情况下，但我们可以通过确保树始终保持平衡来避免最坏情况的发生。

我们在之前的课程中讨论过平衡二叉树。二叉搜索树只是二叉树的一种特殊形式。为了确保这些操作的时间复杂度始终为O(log n)，我们应该保持二叉搜索树的平衡。我们稍后会详细讨论这个问题。首先，让我们看看什么是二叉搜索树，以及当我们使用二叉搜索树时，这些操作的成本是如何被最小化的。二叉搜索树是一种二叉树，其中对于每个节点，左子树中所有节点的值都小于该节点的值，右子树中所有节点的值都大于该节点的值。我在这里将二叉树画成一个递归结构。我们知道，在二叉树中，每个节点最多可以有两个子节点。我们可以将其中一个子节点称为左子节点。

如果我们将树视为一种递归结构，左子节点就是左子树的根节点，同理右子节点就是右子树的根节点。那么，对于一棵被称为二叉搜索树的二叉树来说，左子树中所有节点的值都必须更小。或者为了处理重复值，我们可以说小于或等于，而右子树中所有节点的值都必须更大，并且这一规则必须适用于所有节点。因此，在这个递归结构中，左右子树也必须都是二叉搜索树。我将画一个整数的二叉搜索树。现在，我已经在这里画好了一个整数的二叉搜索树。

让我们看看这个性质是否成立：对于每个节点，左子树中所有节点的值都必须小于或等于该节点，而右子树中所有节点的值都必须大于该节点。首先来看根节点。左子树中的节点值为10、8和12。所以，它们都小于15，而在右子树中，我们有17、20和25。它们都大于15。因此，根节点的情况是良好的。现在，让我们来看这个值为10的节点。在左边，我们有8，它较小。在右边，我们有12，它较大。

所以，我们没问题。这个节点值为20也没问题，我们不需要担心叶子节点，因为它们没有子节点。所以，这是一棵二叉搜索树。那么，如果我把这个值12改成16，现在这还是二叉搜索树吗？对于值为10的节点来说，没有问题。值为16的节点位于其右侧。所以，这不是问题。但是，对于根节点来说，现在左子树中有一个值更高的节点。因此，这棵树不是二叉搜索树。

我会重新调整，把数值改回12。刚才我们说到，在平均情况下，二叉搜索树的搜索、插入或删除操作时间复杂度为O(log n)。这究竟是怎么实现的呢？我们先来谈谈搜索操作。如果我在树中的这些整数存在于一个已排序的数组中，我们就可以进行二分查找。那么，二分查找是怎么做的呢？假设我们要在这个数组中查找数字10。在二分查找中，我们首先将整个列表定义为我们的搜索空间。

这个数字只能存在于搜索空间内。我将用这两个指针——起始和结束——来标记搜索空间。现在，我们将要搜索的数字或元素与搜索空间的中间元素或中位数进行比较。并且，如果正在搜索的记录中，被搜索的元素较小，我们就在左半边继续搜索；否则，就在右半边继续搜索。如果相等，就说明找到了该元素。在这个例子中，10比15小。

因此，我们将向左继续搜索。现在我们的搜索空间缩小了一半。再次与中间元素进行比较，这次我们找到了匹配项。在二分查找中，我们从包含n个元素的搜索空间开始。如果中间元素不是我们要找的元素，就将搜索空间缩小为n/2，并继续将搜索空间减半，直到找到目标记录或搜索空间只剩下一个元素为止。在整个缩小过程中，如果搜索空间从n缩小到n/2，再到n/4，再到n/8，依此类推，那么总共需要进行以2为底的对数log₂n次步骤。

如果我们进行k步操作，那么n除以2的k次方将等于1，这意味着2的k次方等于n，而k将等于以2为底的n的对数。因此，这就是为什么二分查找的运行时间是O(log n)。现在，如果我们使用这个二叉搜索树来存储整数，搜索操作将会非常相似。假设我们要查找数字12。我们将从根节点开始，然后将要查找的值（即整数）与根节点的值进行比较。如果相等，则查找完成。

如果值更小，我们就知道需要去左子树查找，因为在二叉搜索树中，左子树的所有元素都更小，右子树的所有元素都更大。现在，我们将查看值为15的节点的左子节点。我们知道要找的数字12只能存在于这个子树中，除此之外的其他子树都可以被排除。因此，我们将搜索范围缩小到仅剩这三个节点，其值分别为10、8和12。现在，我们再次将12与10进行比较，发现它们并不相等。

12更大。因此，我们知道需要在这个值为10的节点的右子树中继续查找。现在，我们的搜索范围缩小到仅剩一个节点。再次比较该节点的值，我们发现匹配成功。因此，在二叉搜索树中查找元素本质上就是这样的遍历过程：在每一步中，我们会选择向左或向右移动，从而在每一步中舍弃其中一个子树。如果树是平衡的，我们称一棵树为平衡树的条件是：对于所有节点，左右子树的高度差不超过1。因此，如果树是平衡的，我们将从n个节点的搜索空间开始，当我们舍弃其中一个子树时，我们将舍弃n/2个节点。

因此，我们的搜索空间将先缩小到n/2，然后在下一步缩小到n/4。我们会持续这样缩小搜索空间，直到找到目标元素，或者搜索空间缩小到仅剩一个节点时结束。所以，这里的搜索也是一种二分查找，这也是为什么它被称为二叉搜索树。我现在展示的这棵树是平衡的。

事实上，这是一棵完美的二叉树，但由于记录相同，我们可能会得到这样一棵不平衡的树。这棵树具有与之前结构相同的整数值，也是一棵二叉搜索树，但它是不平衡的。这几乎等同于一个链表。

在这棵树中，没有任何节点拥有右子树。每一步都只有一个节点。从搜索空间中的n个节点开始，我们会依次访问n-1个节点、n-2个节点，直到最后访问1个节点，总共需要n步。在二叉搜索树中，平均情况下，搜索、插入或删除操作的时间复杂度是O(log n)，而最坏情况下（也就是我现在展示的这种排列方式），运行时间会达到O(n)。我们总是试图通过保持二叉搜索树的平衡来避免最坏情况的发生。

树中相同的记录可以有多种排列方式。对于这棵树中的这些整数，另一种排列是这样的。对于所有节点，在搜索时左子树中没有需要丢弃的内容。这是另一种排列方式。这仍然是平衡的，因为对于所有节点来说，左右子树的高度差不超过1。但当我们有一个完美二叉树时，这是最佳的排列方式。在每一步，我们都会有正好n除以2的节点被丢弃。

好的，现在要在二叉搜索树中插入一些记录，我们首先需要找到可以插入的位置，我们可以在O(log n)的时间内找到这个位置。假设我们想在这棵树中插入19。我们要做的是从根节点开始。如果要插入的值小于或等于当前节点，且没有左子节点，则作为左子节点插入；否则向左移动。如果值大于当前节点且没有右子节点，则作为右子节点插入；否则向右移动。在本例中，19大于当前节点，因此我们将向右移动。

现在我们位于20。19更小，且左子树不为空。我们有一个左子节点，所以我们将向左移动。现在我们位于17。19比17大，所以它应该放在17的右侧。17没有右子节点。因此，我们将创建一个值为19的节点，并将其作为右子节点链接到这个值为17的节点上。因为我们在这里使用的是指针或引用，就像链表一样，所以不需要进行任何移位操作。与数组类似，创建一个链接只需要常数时间。

因此，总体而言，插入操作和搜索操作的成本是相似的。删除操作也需要先搜索节点。搜索的时间复杂度仍然是O(log n)，而删除节点只需调整一些链接。因此，删除操作的平均时间复杂度也将与搜索类似，为O(log n)。二叉搜索树在插入和删除过程中可能会失去平衡。因此，我们经常在插入和删除操作后恢复其平衡性。

有几种方法可以实现这一点，我们将在后面的课程中详细讨论所有这些内容。下一节课，我们将详细讨论二叉搜索树的实现。这节课就到这里。感谢观看。

## 28、二叉搜索树的实现

在上一课中，我们了解了什么是二叉搜索树。现在，我们将在这节课中实现二叉搜索树。我们将编写一些关于二叉搜索树的代码。学习本课程的前提是，你必须理解C/C++中指针和动态内存分配的概念。如果你已经跟随这个系列，并看过我们关于链表的课程，那么实现二叉搜索树或一般的二叉搜索树不会有太大不同。

我们这里也会有节点和链接。好的，让我们开始吧。二叉搜索树（BST）是一种二叉树，其中对于每个节点，左子树中所有节点的值都小于或等于该节点的值，而右子树中所有节点的值都大于该节点的值。我们可以将二叉搜索树（BST）表示为这样的递归结构：左子树中所有节点的值必须小于或等于当前节点，右子树中所有节点的值必须大于当前节点，并且这一性质必须适用于所有节点，而不仅仅是根节点。因此，在这个递归定义中，左右子树本身也必须是二叉搜索树。

我在这里画了一个整数的二叉搜索树。现在的问题是，我们如何在计算机的内存中创建这种非线性逻辑结构。在我们讨论二叉树时，我已经简要地提到过这一点。最流行的方法是动态创建节点，并使用指针或引用将它们相互连接。就像我们处理链表的方式一样。因为在二叉搜索树或一般的二叉树中，每个节点最多可以有两个子节点。

我们可以将节点定义为一个包含三个字段的对象，就像我在这里展示的这样。我们可以用一个字段来存储数据，另一个字段来存储左子节点的地址或引用，还有一个字段来存储右子节点的地址或引用。如果节点没有左子节点或右子节点，引用可以设置为空。在C或C++中，我们可以这样定义节点。有一个字段用于存储数据。

这里的数据类型是整数，但它可以是任何类型。有一个字段是指向节点的指针。Node asterisk 表示指向节点的指针。这个用于存储左子节点的地址，另一个用于存储右子节点的地址。这个节点的定义与双向链表中节点的定义非常相似。还记得在双向链表中，每个节点也有两个链接。

一个指向前一个节点，另一个指向下一个节点。但双向链表是一种线性排列。这个节点的定义是针对二叉树的。我们也可以将其命名为BST节点。但节点这个名称也没问题。我们就用节点吧。

现在在我们的实现中，就像链表一样，所有的节点都将通过C语言中的malloc函数或C++中的new操作符在应用程序内存的动态内存或堆区中创建。在C++中我们也可以使用malloc函数。众所周知，任何在应用程序内存的动态内存或堆区中创建的对象都不能拥有名称或标识符。

必须通过指针来访问它。malloc或new运算符会返回指向堆中创建的对象的指针。如果你想复习这些动态内存分配的概念，可以查看本视频描述中的课程链接。理解应用程序内存中栈和堆的概念非常重要。对于链表来说，我们始终需要记住的是头节点的地址。只要知道头节点，我们就可以通过链接访问所有其他节点。

对于树结构，我们始终保存的信息是根节点的地址。只要知道根节点，我们就可以通过链接访问树中的所有其他节点。要创建一棵树，首先需要声明一个指向二叉搜索树节点的指针。我宁愿在这里称它为二叉搜索树节点。BST即二叉搜索树。因此，要创建一棵树，我们首先需要声明一个指向BST节点的指针，该指针将始终存储根节点的地址。

我在这里声明了一个指向节点的指针，命名为rootptr，ptr代表指针。在C语言中，你不能直接写BST node星号rootptr。你必须写成struct空格BST node星号。这里你也必须写上struct。我打算在这里写C++代码。不过无论如何，现在我在尝试解释逻辑。

我们不必纠结于具体实现的新细节。在我展示的这个树形逻辑结构中，每个节点如你所见包含三个字段，即三个单元格。最左侧的单元格用于存储左子节点的地址，最右侧的单元格则用于存储右子节点的地址。假设根节点在内存中的地址是200，其他节点的地址我也随机假设一下。现在我可以为每个节点的左右单元格填入左右子节点的地址。在我们的定义中，数据是第一个字段，但在这个逻辑结构中，我把数据显示在中间。

好的，对于每个节点，我已经填写了左子节点和右子节点的地址。如果没有子节点，地址为零或空。正如我们之前所说，树的标识是根节点的地址。我们需要有一个指向节点的指针，以便在其中存储根节点的地址。我们必须有一个类型为指向节点的指针变量来存储根节点的地址。所有这些带有三个单元格的矩形都是节点。

它们是通过malloc或new操作符创建的，并存在于应用程序内存的堆区。我们无法为它们命名或标识，总是通过指针来访问。这个根指针（root ptr）必须是一个局部或全局变量。稍后我们会更详细地讨论这一点。通常，我们喜欢将这个根指针命名为root。我们可以这样做，但不要混淆，这是指向根的指针，而不是根本身。正如我所说的，要创建一个二叉搜索树（BST），我们首先需要声明这个指针。最初，我们可以将这个指针设为null，表示树为空。没有节点的树可以称为空树，对于空树，根指针应设为null。我们可以在程序的主函数中进行此声明并将根指针设为null。

其实，我们直接在真正的编译器里写这段代码吧。我这里在写C++。如你在main函数中所见，我声明了这个指向节点的指针，它将始终存储我的树的根节点地址，我最初将其设为null，表示树是空的。通过这段代码，我们创建了一棵空树，但空树有什么意义呢？我们需要往里面添加一些数据。所以，我现在想做的是编写一个函数来向树中插入节点。我将编写一个名为insert的函数，该函数将接收根节点的地址和要插入的数据作为参数，并在树的适当位置插入一个包含该数据的节点。在主函数中，我将调用这个insert函数，传入根节点的地址和要插入的数据。假设我先要插入数字15，然后是数字10，接着是数字20。我们还可以插入更多数据，但让我们先编写insert函数的逻辑。

在我为插入函数编写逻辑之前，我想先编写一个函数来在动态内存或堆中创建一个新节点。这个获取新节点的函数应该接收一个整数（即要插入的数据）作为参数，使用new或malloc在堆中创建一个节点，并返回这个新节点的地址。我在这里使用new操作符来创建新节点。

操作员将返回新创建节点的地址，我正在将其收集到这个指向BST节点的指针变量中。在C语言中，我们需要使用malloc而不是new操作符。在C++中也可以使用malloc。C++只是C的一个超集。在这里使用malloc也是可以的。现在，动态内存或堆中的任何内容总是通过指针来访问。现在，通过这个新节点的指针，我们可以访问新创建节点的字段。

我必须使用星号运算符来解引用这个指针。所以我写了星号new node，现在可以访问这些字段了。节点中有三个字段：data和两个指向左右节点的指针。我已经在这里设置了数据。我们可以使用这种替代语法，而不是写星号新节点点数据。我们可以简单地写新节点箭头数据，这表示相同的意思。

我们在链表课程中大量使用了这种语法。现在对于新节点，我们可以将左右子节点设为null，最后返回新节点的地址。好的，回到插入函数。插入操作可能会出现几种情况。首先，树可能是空的。当我们第一次插入数字15时，树将是空的。

如果树为空，我们可以简单地创建一个新节点并将其设为根节点。通过这条语句 root 等于获取新节点，我将根节点设置为新节点的地址，但这里有些地方不太对劲。这个根节点是插入函数的局部变量，其作用域仅限于该函数内部。我们希望修改的是主函数中的根节点。这个根节点是主函数的局部变量。有两种方法可以实现这一点：我们可以返回新根节点的地址，这样插入函数的返回类型将是指向二叉搜索树节点的指针，而不是void。

在主函数中，我们需要编写类似"root等于插入函数及其参数"这样的语句。因此，我们必须收集返回值并在主函数中更新root。另一种方法是，我们可以将主函数中这个root的地址传递给插入函数。这个root已经是一个指向节点的指针，所以它的地址可以被收集在一个指向指针的指针中。因此，在插入函数中，第一个参数将是一个指向指针的指针，这里我们可以传递地址。


我们将使用“&根”来传递地址。我们可以将这个参数命名为root，也可以命名为root ptr，或者随便取什么名字。现在我们需要做的是，用星号运算符解引用这个指针，以访问main函数中root的值，同时也可以设置main函数中root的值。因此，通过这条语句，我们设置了值，现在返回类型可以是void。这个指向指针的指针有点复杂。

我会选择前一种方法。实际上，还有另一种方式。我们不必在main函数中将root声明为局部变量，而是可以将其声明为全局变量。众所周知，全局变量必须在所有函数之外声明。如果root是全局变量，那么所有函数都可以访问它，我们就不需要传递存储在其中的地址作为参数了。

无论如何，回到插入的逻辑，正如我们所说的，如果树是空的，我们可以简单地创建一个新节点并将其设置为根节点。在这个阶段，我们想要插入15。如果我们调用插入函数，根的地址是0或null。Null只是0的宏定义，第二个参数是要插入的数字。

在这个插入函数的调用中，我们将调用获取新节点的函数。假设我们在地址200处获得了这个新节点。获取新节点函数将返回给我们地址200，我们可以在这里将其设置为根节点。但这个根节点是一个局部变量。我们将把这个地址200返回给主函数，在主函数中，我们实际上是将这个根节点等同于插入操作。因此，在主函数中，我们正在建立这个链接。

好的，接下来在主函数中我们要插入数字10。此时，根节点是200。根节点的地址是200，要插入的值是10。现在，树不是空的。那么，我们该怎么做呢？如果树不为空，基本上有两种情况。如果要插入的数据小于或等于根节点的值，我们需要将其插入到根节点的左子树中；如果要插入的数据大于根节点的值，则需要将其插入到根节点的右子树中。

因此，我们可以以一种自相似的方式，即递归的方式，来减少这个问题。在处理树结构时，递归是我们几乎随时都会用到的一种方法。在这个函数中，我会说如果要插入的数据小于或等于根节点的数据，那么就递归调用将数据插入左子树。

左子树的根节点将成为左孩子。因此，在这个递归调用中，我们传递左孩子的地址和数据作为参数，当数据插入左子树后，左子树的根节点可能会发生变化。插入函数将返回左子树新根节点的地址，我们需要将其设置为当前节点的左孩子。

在这个示例树中，目前左右子树均为空。我们正尝试插入数字10。因此，我们调用了这个插入函数。从主函数中，我们通过传递地址200和数值（或数据）10来调用插入函数。现在，由于10小于15，程序控制将转到这一行，并调用在左子树中插入数据的函数。

现在，左子树为空。因此，左子树的根地址为0。传递的数据，要插入的数据作为参数传递的是10。现在，第一个插入调用将等待下面的插入完成并返回。在最后一次插入调用中，根节点为空。假设我们在地址150处获得了这个节点。现在，这个插入调用将返回150，第一个插入调用的执行将在此行恢复，现在这个特定地址将被设置为150。

所以，我们将建立这个链接，现在这个插入调用可以结束了。它可以返回当前的根节点。实际上，这个返回根节点的操作应该适用于所有情况。所以，我把它取出来，在所有这些条件之后我得到了它。当然，我们这里还有一个else。如果数据更大，我们需要插入到右子树中。在插入函数的第三次调用中，我们插入数字20。这次，我们将执行else语句。假设else语句中的新节点地址为300。所以，这个节点会返回300。对于这个200的节点，右子节点会被设置为300，现在这个插入调用就可以结束了。返回值将是200。

好的，在这个阶段，如果调用插入数字25的操作会怎样。我们现在处于根节点，也就是地址为200的节点。25更大，所以我们需要向右子树插入。这次右子树不为空。因此，对于这次调用，我们再次来到这个else分支，最后的else，因为25大于20。现在，在这次调用中，我们将进入第一个if语句。将创建一个节点。假设我们在堆中得到了这个节点，地址为500。

这个特定的调用插入025将返回500并结束。现在，对于节点300，其右子节点将被设置为500。因此，这个链接将被建立。现在，这个人将返回300。这个子树的根没有改变，第一次调用插入也将结束。它将返回200。所以，在所有情况下我们都表现得很好。这个插入函数适用于所有情况。我们也可以在不使用递归的情况下编写这个插入函数。

我鼓励你这样做。你需要使用一些临时指针和循环。递归在这里非常直观，而且在我们处理树结构的几乎所有操作中，递归都很直观。因此，我们真正理解递归是非常重要的。好的，我现在再写一个函数来在二叉搜索树中搜索一些数据。在主函数中，我又进行了一些插入操作的调用。

现在，我想写一个名为search的函数，该函数应以根节点的地址和要搜索的数据作为参数。如果数据存在于树中，该函数应返回true，否则返回false。我们再次需要考虑几种情况。如果根节点为空，则可以返回false。如果根节点的数据等于我们要查找的数据，那么我们可以返回true。否则，我们有两种情况。要么我们需要去左子树中搜索，要么我们需要去右子树中搜索。

所以，我在这里再次使用了递归。在这两种情况下，我对搜索函数进行了递归调用。如果你理解了之前的递归，那么这个非常相似。现在来测试这段代码。我在这里所做的是让用户输入一个要搜索的数字，然后调用这个搜索函数。如果函数返回true，我就打印“找到”，否则打印“未找到”。让我们运行这段代码，看看会发生什么。

我把多条插入语句放在一行是因为这里空间有限。假设我们要搜索数字8。8被找到了，现在假设我们要搜索22。22没有被找到。所以，我们进展顺利。我就讲到这里。你可以查看本视频描述中的链接获取所有源代码。在接下来的课程中，我们将深入探讨树结构的更多应用。下节课中，我们会更深入一些，看看应用程序内存各个部分中的数据是如何移动的——当我们执行这些函数时，数据如何在内存的栈区和堆区之间流动。这会让你对概念有更清晰的认识。本节课就到这里。感谢观看。

## 29、栈和堆的内存分配

在之前的课程中，我们编写了一些关于二叉搜索树的代码。我们编写了在二叉搜索树中插入和搜索数据的函数。现在在这节课中，我们将更深入一些，尝试理解当这些函数执行时，应用程序内存的各个部分是如何变化的，这将让你更加清晰明了。

这将让你大致了解程序执行过程中内存的管理方式，以及树结构中频繁使用的递归机制是如何运作的。本节课要讨论的概念在我们之前的课程中已经有所涉及，但在实现树结构时重温这些概念会很有帮助。那么，以下是我们之前编写的代码。

我们有这个函数get new node来在动态内存中创建一个新节点，然后有这个函数insert来在树中插入一个新节点，接着有这个函数来在树中搜索一些数据，最后这是主函数。你可以查看这个视频的描述以获取源代码的链接。

现在在主函数中，我们有一个名为root的BST节点指针，用于存储树的根节点地址。我最初将其设置为null以创建一个空树，然后调用插入函数向树中插入一些数据，最后要求用户输入一个数字，并调用搜索函数在树中查找这个数字。

如果搜索函数返回true，我就打印“找到”，否则打印“未找到”。让我们看看程序执行时内存会发生什么变化。在典型架构中，分配给程序或应用程序执行的内存可以分为以下四个部分。程序中有一个称为文本段的部分，用于存储所有的指令。这些指令会被编译成机器语言。还有另一个部分用于存储所有的全局变量。

在所有函数之外声明的变量称为全局变量。所有函数都可以访问它。接下来的段栈基本上是函数调用执行的暂存空间。所有局部变量，即在函数内部声明的变量，都存储在栈中。而第四部分——堆（也称为自由存储区）则是动态内存，可以根据我们的需求增长或缩减。其他所有段的大小都是固定的，这些段的大小在编译时就已经确定。但堆可以在运行时增长，而且我们无法在运行时控制其他段的内存分配或释放，但可以控制堆中的内存分配和释放。

我们在动态内存分配的课程中已经详细讨论过所有这些内容。你可以在描述中查看相关链接。现在我要做的是将栈和堆区域画成这两个矩形容器。我正在放大这两个部分。现在我将向大家展示当这个程序执行时，应用程序内存中这两个部分的数据是如何移动的。当程序开始执行时，首先会调用主函数。

每当调用一个函数时，都会从栈中分配一定量的内存用于其执行。这块被分配的内存称为函数调用的栈帧。所有局部变量以及函数调用的执行状态都将存储在函数调用的栈帧中。在主函数中，我们有一个局部变量root，它是指向BST节点的指针。因此，我在这里的栈帧中展示root。我们将按顺序执行指令。

在main函数的第一行，我们声明了root并对其进行初始化，将其设置为null。Null只是地址0的宏定义。因此，在这个图中，我将root中的地址设置为0。接下来的一行，我们调用了insert函数。此时，main函数的执行将在此处暂停，并为insert函数的执行分配一个新的栈帧。

主程序将等待上述插入操作完成并返回。一旦插入调用完成，主程序将在第2行恢复执行。在插入函数中，我们有两个局部变量root和data用于接收参数。对于这次插入函数的调用，由于root为空，我们将进入这里的第一个if条件分支。在这一行，我们将调用获取新节点的函数。因此，这个插入调用的执行将再次暂停，并为执行获取新节点函数分配一个新的栈帧。在获取新节点函数中，我们有两个局部变量：用于收集参数的data，以及一个名为new node的指向BST节点的指针。

现在在这个函数中，我们使用new操作符在堆中创建一个二叉搜索树节点。假设我们在地址200处获得了一个新节点。new操作符将返回给我们这个地址200。因此，这个地址将被设置在这里的新节点中。这样我们就有了这个链接，现在使用这个新节点的指针，我们正在设置节点的这三个字段的值。假设第一个字段是用来存储数据的。

因此，我们在这里将值设为15，假设第二个单元格用于存储左子节点的地址。这里将其设为null，右子节点的地址同样设为null。现在，get_new_node函数将返回新节点的地址并结束执行。每当一个函数调用完成时，分配给它的栈帧就会被回收。

调用插入函数将在此行恢复执行，并将获取新节点地址200的返回值赋给这个根节点，该根节点是插入调用的局部变量。现在，插入函数这次特定的调用将返回根节点的地址，即当前存储在该变量根中的地址200，然后结束。此时，主函数将在此行恢复执行，主函数的根节点将被设为200。这次插入调用（插入根节点15）的返回值将在此处设置。

现在，在执行主函数时，控制权将转到下一行，我们调用插入函数来插入数字10。再次，主函数的执行将被暂停，并为插入函数的执行分配一个栈帧。这一次对于插入调用，根节点不为空。所以我们不会进入第一个if语句。我们将在insert函数中使用名为root的指针访问地址200处该节点的数据字段，并将其与值10进行比较。10小于15，因此我们将执行这一行代码，现在在这里进行递归调用。

递归是指函数调用自身，而函数调用自身与函数A调用另一个函数B并无本质区别。因此，这里会发生的情况是：当前这个特定的insert函数调用会被暂停，系统会为执行另一个insert函数调用分配一个新的栈帧。传递给这个新insert调用的参数是：局部变量root中的地址0（即地址200处节点的左子节点为空）。所以我们传入的root参数是0，数据参数是10。对于这个特定的insert调用，程序会首先进入第一个if条件判断，然后执行这一行的get_new_node函数调用。

因此，执行这个插入操作会暂停，我们将转到这里的获取新节点函数。我们正在堆中创建一个新节点。假设我们在地址150处获得了这个新节点。现在，获取新节点将返回150并结束。对这个插入调用的执行将在这行代码处恢复。获取新节点的返回值将在这里设置，现在这个插入调用将返回地址150并结束。

在下面的插入操作将从此行继续执行，现在在这个插入调用中，地址为200的该节点的左子节点将被设置为前一个插入调用的返回值150。最后，这个插入调用将完成。控制权将返回到主程序的这一行，根节点将被重写为200。但之前它已经是200了。

它没有变化。接下来在主函数中，我们调用了插入数字20的操作。我不打算展示这个操作的模拟过程。再次说明，栈中分配的内存会增长和收缩。最后，当控制权返回到主函数，在这次插入调用结束后，我们会在堆中有一个值为20的节点，设置为这个地址200节点的右孩子。假设我们得到了这个值为20的新节点，地址为300。

如你所见，地址200处的节点其右子节点地址被设置为300。接下来要插入数字25。这个例子很有趣，让我们看看会发生什么。主程序将暂停，我们会进入这次插入调用。对于该调用而言，本地根节点接收到的地址参数是200。

我们已经在数据中传递了数字25。现在这里的25大于地址200处该节点的值。因此，我们将进入最后一个else条件，需要在右子树中插入。于是会再次调用插入函数，我们将传递地址300作为根节点，传递的数据仍为25。对于这次调用，节点300中的值再次进行比较。这次调用的根节点300小于2525，而25大于20。

因此，我们再次来到最后的 else 部分，并对右子树进行递归插入调用。这次右子树是空的。所以对于这次顶层的插入调用，这里的根地址将是零。因此，对于这次调用，我们将进入第一个 if 条件并调用 get new node。假设这个新节点返回给我们地址为 100 的节点。我空间不够了。

所以我并没有展示全部内容，现在这里有一个新的节点栈帧，我们将返回到顶部的这个插入调用。现在这条路径被设置为新创建节点的100地址。现在这个插入调用将完成，我们将回到下面的这个插入调用。

然后这个插入操作将在最后一个else中的这一行恢复执行，地址为300的节点的右子节点将被设置为100。现在这个插入操作将返回地址300，无论其根节点设置为什么。下面的这个插入操作将在最后一个else中的这一行恢复执行，地址为200的节点的右子节点将被设置为300。

之前也是300。所以即使覆盖后，我们也不会更改。现在这个插入操作即将完成。最后，主程序会从这里恢复，主程序的根节点将被设置为这个插入调用的返回值，它只会被相同的值覆盖。确保主程序中的这个根节点以及所有链接和节点都正确更新是非常重要的。由于代码中的错误，我们经常会丢失一些链接，或者创建一些不必要的链接。

现在，正如你所看到的，我们正在堆中创建所有节点，堆为我们提供了这种灵活性——我们可以在运行时决定节点的创建。而且我们可以控制堆中任何对象的生命周期，任何在堆中申请的内存都必须显式地释放，在C语言中使用free函数，在C++中使用delete操作符，因为堆中的内存在程序运行期间会一直保持分配状态。

如你所见，栈中的内存在函数调用结束时会被释放。主函数中其余的函数调用也将以类似的方式执行。我留给你自己去观察和思考。现在我们在堆中有了这棵树。

从逻辑上讲，内存本身是一个线性结构。而树作为一种非线性结构——在逻辑上确实是非线性的——会以我展示的这种堆中节点随机分布、彼此相连的方式融入其中。希望这个解释能让你更清楚。在接下来的课程中，我们将解决一些关于树的问题。这节课就到这里。感谢观看。

## 30、寻找二叉搜索树中最小和最大的元素

在我们之前的课程中，我们为二叉搜索树编写了一些基础代码。但为了巩固概念，我们需要编写更多代码。所以我挑选了这个简单的问题：给定一个二叉搜索树，我们需要找出其中的最小和最大元素。让我们看看如何解决这个问题。这里我画了一个整数二叉搜索树的逻辑示意图。

众所周知，在二叉搜索树中，对于所有节点而言，左子树节点的值较小，而右子树节点的值较大。在C或C++中，我们可以这样定义一个二叉搜索树的节点：使用一个包含三个字段的结构体，其中一个字段存储数据，另一个存储左子节点的地址，还有一个存储右子节点的地址。正如我们之前在二叉搜索树的实现中所见，树的标识（即我们始终持有并传递给函数的）是根节点的地址。

所以我想做的是，首先编写一个名为find_min的函数，该函数应将根节点的地址作为参数，并返回树中的最小元素。与find_min类似，我们还可以编写另一个名为find_max的函数，它可以返回二叉搜索树中的最大元素。让我们先看看如何找到最小元素。

这里有两种可能的方法：我们可以编写一个迭代解决方案，通过简单的循环找到最小元素；或者我们可以使用递归。首先来看迭代解决方案。如果我们有一个指向根节点的指针，并且想要在二叉搜索树（BST）中找到最小元素，那么我们需要从根节点出发，尽可能沿着左链接前进。

因为在二叉搜索树（BST）中，对于所有节点而言，左侧节点的值都较小，而右侧节点的值都较大。因此，我们需要尽可能地向左遍历。我们可以从指向根节点的临时指针开始，可以将这个指针命名为temp，或者命名为current以表示我们当前正指向该节点。

在我的这个函数中，我声明了一个名为current的BST节点指针，最初我将根节点的地址赋给它。通过这个指针，我们可以用类似current等于current箭头left的语句来访问左子节点，但首先需要检查是否存在左子节点，然后再移动指针。我们可以使用这样的while循环来实现。

如果当前节点的左子节点不为空，我们可以通过语句current = current->left将指针current移动到左子节点。在这个例子中，当前我们指向值为15的节点。它有一个左子节点，因此我们可以移动到值为10的节点。同样，这个节点也有一个左子节点，所以我们可以再次向左移动。

现在这个值为8的节点没有左子节点。因此我们无法继续向左遍历，while循环将在此终止。此时我们指向的节点就是最小值节点，可以直接返回该节点的数据值。不过这个函数还遗漏了一种情况——如果树为空时，我们可以抛出错误或返回一个表示空树的值。比如当确定树中只会存储正数值时，可以返回-1这样的特殊值来标识空树状态。

因此在我的函数中，我添加了这个条件：如果根节点等于null，也就是说如果树是空的，就打印这个错误并返回负一。还有一点，我们不需要在这里使用名为current root的额外BST节点指针，因为root本身就是一个局部变量。我们可以直接使用这个root。所以我们可以这样编写代码：当root的左节点不等于null时，我们可以通过语句root等于root箭头left来向左移动。

最后，我们可以返回根箭头数据，这只是星号根点数据的另一种语法。修改这个局部根不会影响主函数中的根，也不会影响调用这个查找主函数的任何其他函数中的根。因此，这是我们用于查找二叉搜索树中最小元素的迭代解决方案。查找最大值的逻辑类似，唯一的区别在于我们不再向左走，而是一直向右走。这部分就留给你们来实现。

现在让我们看看如何用递归找到最小元素。如果我们想以递归的方式、自相似的方式来解决这个问题，那么我们可以说，如果左子树不为空，那么我们可以将问题简化为在左子树中寻找最小值。如果左子树为空，我们已经知道最小值了，因为右子树中不可能有更小的值。

我们可以写出以下递归逻辑：根节点为空是一种边界情况。如果根节点为空，即树为空，我们可以抛出错误。否则，如果根节点的左子节点为空，我们可以返回根节点中的数据。否则，如果左子节点不为空，换句话说，如果左子树不为空，我们可以将问题简化为在左子树中搜索最小值。

因此，我们进行这个递归调用来查找最小值，传入左子节点的地址，即左子树的根节点地址，左子节点就是左子树的根节点。第二个 else if 是我们的递归终止条件。如果你已经理解了之前我们写的在二叉搜索树中插入节点的递归方法，那么这个递归对你来说应该不难理解。

这是我们用递归方法在二叉搜索树（BST）中查找最小值的解决方案。要找到最大值，我们只需要在右子树中继续搜索即可。好的，今天就讲到这里。在接下来的课程中，我们将解决更多关于二叉搜索树的有趣问题。感谢观看。本节课我们将编写代码来计算二叉树的高度，也可以称之为二叉树的最大深度。

我们在第一堂关于树的入门课中已经讨论了深度和高度。但在这里我会快速回顾一下。首先，我在这里画了一棵二叉树。我没有在节点中填入任何数据，数据可以是任何东西。正如我们所知，二叉树是一种每个节点最多可以有两个子节点的树。因此，一个节点可以有零个、一个或两个子节点。

我会给这些节点编号以便引用。假设这个根节点是1号。然后我会从左到右逐层编号，依次为二、三、四，依此类推。树的高度定义为从根节点到叶节点的最长路径上的边数。在这个示例树中，四、六、七、八和九都是叶节点。叶节点是指没有子节点的节点，从根节点到叶节点的最长路径上的边数为三。

对于路径中有八条和九条边的情况，从根节点出发的边数都是三。因此，树的高度为三。实际上，我们可以将树中某个节点的高度定义为从该节点到叶节点的最长路径中的边数。所以，树的高度基本上就是根节点的高度。在这个示例树中，节点三的高度是一，节点二的高度是二，节点一的高度是三。

由于这是根节点，所以这也是树的高度，叶节点的高度为零。因此，如果一棵树只有一个节点，那么根节点本身就是一个叶节点，因此树的高度为零。这就是树高的定义。我们还经常讨论深度，并且经常混淆深度和高度。

但这二者是不同的属性。节点的深度被定义为从根节点到该节点的路径上的边数。基本上，深度是到根节点的距离，而高度是到最深可达叶节点的距离。在这个示例树中，节点二的深度为一，高度为二。对于节点九（这是一个叶节点），深度为三，高度为零。根节点的深度为零，高度为三。

树的高度等于树中任意节点的最大深度。因此，高度和最大深度这两个术语可以互换使用。好的，现在让我们看看如何计算二叉树的高度或最大深度。我将编写一个名为find_height的函数，该函数以根节点的引用或地址作为参数，并返回二叉树的高度。计算高度的逻辑可以这样设计：对于任何节点，如果我们能计算出其左子树的高度和右子树的高度，那么该节点的高度就是左右子树高度中的较大值加一。

在这棵树中，根节点的左子树高度为二，右子树高度为一。因此，根节点的高度将是这两个值中的较大者，再加上一（因为根节点与子树之间有一条边连接）。所以，根节点的高度（也就是这棵树的高度）在这里是三。

在我们的代码中，我们可以使用递归来计算左右子树的高度。我将在这里做的事情以及查找高度的函数是，我会首先递归调用查找左子树的高度。我们可以说是查找左子树的高度或查找左孩子的高度。两者意思相同。我将这个递归调用的返回值保存在一个名为left height的变量中。现在我要进行另一个递归调用来计算右子树或右孩子的高度。

现在，树的高度或我们调用此函数的任何节点的高度将是左高度和右高度这两个值中较大的一个加一。现在递归中只缺少一件事，我们需要编写基本或退出条件，我们不能无限递归。我们可以做的是，继续递归直到根节点为空。

如果根节点为空，即树或子树为空，我们可以返回某个值。这里应该返回什么呢？仔细想想。假设我调用函数来查找某个叶节点的高度，比如这个标号为7的节点，那么对于这个节点来说，它的左右子节点都为空。

在调用节点七时，我们将进行两次递归调用，两次调用都传入null。那么我们应该返回什么？应该返回零吗？如果这两个调用都返回零，那么节点七的高度就是一。因为这里的返回语句是说，左子树和右子树高度的最大值加一。

但正如我们之前讨论的，叶节点的高度应为零。因此，如果根节点为空时返回零，这是不正确的。我们可以改为返回负一。当我们返回负一时，这条实际上不存在的通向空节点的边，虽然仍被计算在内，但会通过负一得到平衡。希望这能讲得通。按照惯例，空树的高度也设定为负一。

这是我用来查找二叉树高度的伪代码。有些人将高度定义为从根节点到叶节点的最长路径中的节点数量。我们这里计算的是边的数量。这才是正确的定义。如果你想计算节点的数量，那么对于叶节点来说高度就是一，而对于空树来说高度就是零。所以你只需要在这里返回零就可以了。

以下是计算节点数量的代码。不过我认为正确的定义应该是边的数量。所以这里我会返回减一的结果。这个函数的时间复杂度是大O(n)，其中n是树中的节点数量。我们会为树中的每个节点进行一次递归调用。也就是说，我们基本上会访问树中的每个节点一次，因此运行时间将与节点数量成正比。本节课中我将跳过对运行时间的详细分析。这就是我的查找高度函数在C或C++中的样子。这里的Max是一个函数，它会返回作为参数传递给它的两个值中的较大者。这节课就到这里。感谢观看。



In this lesson, we are going to talk about binary tree traversal. When we are working with trees, we may often want to visit all the nodes in the tree.

Now tree is not a linear data structure like array or linked list. In a linear data structure, there would be a logical start and a logical end. So we can start with a pointer at one of the ends and keep moving it towards the other end.

For a linear data structure like linked list, for each node or element, we would have only one next element. But tree is not a linear data structure. I have drawn a binary tree here, data type is character this time I filled in these characters in the nodes.

Now for a tree at any time, if we are pointing to a particular node, then we can have more than one possible directions, we can have more than one possible next nodes. In this binary tree, for example, if we will start with a pointer at root node, then we have two possible directions. From F, we can either go left to D, or we can go right to J. And of course, if we will go in one direction, then we will somehow have to come back and go into the other direction later.

So tree traversal is not so straightforward. And what we are going to discuss in this lesson is algorithms for tree traversal. Tree traversal can formally be defined as the process of visiting each node in the tree exactly once in some order.

And by visiting a node, we mean reading or processing data in the node. For us in this lesson, visit will mean printing the data in the node. Based on the order in which nodes are visited, tree traversal algorithms can broadly be classified into two categories.

We can either go breadth first, or we can go depth first. breadth first traversal and depth first traversal are general techniques to traverse or search a graph. Graph is a data structure and we have not talked about graph so far in this series.

We will discuss graph in later lessons. For now, just know that tree is only a special kind of graph. And in this lesson, we are going to discuss breadth first and depth first traversal in context of trees.

In a tree, in breadth first approach, we would visit all the nodes at same depth or level before visiting the nodes at next level. In this binary tree that I'm showing here, this node with value f, which is the root node is at level zero, I'm writing L0 here for level zero, depth of a node is defined as number of edges in path from root to that node, the root node would have depth zero, these two nodes D and J are at depth one. So we can say that these nodes are at level one.

Now these four nodes are at level two, these three nodes are at level three. And finally, this node with value h is at level four. So what we can do in breadth first approach is that we can start at level zero, we would have only one node at level zero, the root node.

So we can visit the root node, I'll write the value in the node as I'm visiting it. Now level zero is done. Now I can go to level one and visit the nodes from left to right.So after F, we would visit D and then we would visit J. And now we are done with level one. So we can go to level two, we will go like B, then E, then G and then K. And now we can go to level three, A, C and I and finally I can go to level four. This kind of breadth first traversal in case of trees is called level order traversal.

And we will discuss how we can do this programmatically in some time. But this is the order in which we would visit the nodes, we would go level by level from left to right. In breadth first approach for any node, we visit all its children before visiting any of its grandchildren.

In this tree, first we are visiting F, and then we are visiting D, and then we are not going to any child of D, like B or E along the depth. Next we are going to J. But in depth first approach, if we would go to a child, we would complete the whole sub tree of the child before going to the next child. In this example tree here from F, the root node, if we are going left to D, then we should visit all the nodes in this left sub tree that is we should finish this left sub tree in its complete depth.

Or in other words, we should finish all the grandchildren of F along this path before going to right child of F, J. And once again, when we will go to J, we will visit all the grandchildren along this path. So basically we will visit the complete right sub tree. In depth first approach, the relative order of visiting the left sub tree, the right sub tree, and the root node can be different.

For example, we can first visit the right sub tree, and then the root and then the left sub tree. Or we can do something like we can first visit the root, and then the left sub tree and then the right sub tree. So the relative order can be different.

But the core idea in depth first strategy is that visiting a child is visiting the complete sub tree in that path. And remember, visiting a node is reading processing or printing the data in that node. Based on the relative order of left sub tree, right sub tree and the root, there are three popular depth first strategies.

One way is that we can first visit the root node, then the left sub tree, and then the right sub tree, left and right sub trees will be visited recursively in same manner. Such a traversal is called preorder traversal. Another way is that we can first visit the left sub tree, then the root and then the right sub tree.

Such a traversal is called in order traversal. And if root is visited after left and right sub trees, then such a traversal is called post order traversal. In total, there are six possible permutations for left, right and root.

But conventionally, a left sub tree is always visited before the right sub tree. So these are the three strategies that we use. Only the position of root is changing here.If it's before left and right, then it's preorder. If it's in between, it's in order. And if it's after left and right sub trees, then it's post order.

There is an easy way to remember these three depth first algorithms. If we can denote visiting a node or reading the data in that node with letter D going to the left sub tree as L and going to the right sub tree as R. So if we can say D for data L for left and R for right, then in preorder for each node, we will go DLR. First, we will read the data in that node, then we will go left.

And once the left sub tree is done, we will go right. In in order traversal. First, we will finish the left sub tree, then we will read the data in current node.

And then we will go right. In post order for each node first we will go left. Once left sub tree is done, we will go right.

And then we will read the data in current node. So preorder is data left right in order is left data right. And post order is left right and then data.

preorder in order and post order are really easy and intuitive to implement using recursion. But we will discuss implementation later. Let's now see what will be the preorder in order and post order traversal for this tree that I've drawn here.

Let's first see what will be the preorder traversal for this binary tree, we need to start at root node. And for each node, we first need to read the data, or in other words, visit that node. In fact, instead of DLR, we could have said VLR here V for visit, we can use any of these assumptions V for visit or D for data, I will go with DLR here.

So let's start at the root. For each node, we first need to read the data. I'm writing F here, the data that I just read.

And now I need to go left and finish the complete left sub tree. And once all the nodes in the left sub tree are visited, then only I can go to the right sub tree. The problem here is actually getting reduced in a self similar or recursive manner.

Now we need to focus on this left sub tree. Now we are at D root of this left sub tree of F. Once again, for this node, we will first read the data. And now we can go left, we will go towards E only when these three nodes A, B and C will be done.

Now we are focusing on this sub tree comprising of these three nodes. Now we are at B, we can read the data. And now we can go left to A, there is nothing in left of A. So we can say that for left for A left sub tree is done.

And there is nothing in right as well. So we can say right is also done. Now for B left sub tree is done.

So we can go right to C and left and right of C are null. And now for D left sub tree is done. So we can go right.

Once again for E left and right are null. And now at this stage for F complete left sub tree is visited. So we can go right.Now we need to go left of J and there is nothing left of G so we can go right. And now we can go left of I. For H there is nothing in left and right. Now at this stage left sub tree of I is done.

And right sub tree is null. And now we can go back to J, the left sub tree for J is done. So we can go to its right sub tree.

Finally, we have K here and we are done with all the nodes. This is how we can perform a pre order traversal manually. Actual implementation would be a simple recursion.

And we will discuss it later. Let's now see what will be the in order traversal for this binary tree. In in order traversal, we will first finish visiting the left sub tree, then visit the current node and then go right.

Once again, we will start at the root and we will first go left. Now we will first finish this sub tree. Once again for D we will first go left to B and from B we will go to A. Now for A there is nothing left.

So we can say that for this guy left sub tree is done. So we can read the data. And now we can go to its right.

But there is nothing in right as well. So this guy is done. Now for B left sub tree is done.

So we can read the data. And now for B we can go right. For C once again, there is nothing left.

So we can read the data. And there is nothing in right as well. Now left of D is completely done.

So we can visit it read the data here. Now we can go to its right to E. For E once again left and right are null. At this stage, left sub tree of F is done.

So we can read the data. And now we can go to right of F. If we will go on like this, this finally will be my in order traversal. This tree that I'm showing here is actually a binary search tree.

For each node, the value of nodes in left is lesser and the value of nodes in right is greater. So if we are printing in this order left sub tree and then the current node and then the right sub tree, then we would get a sorted list. In order traversal of a binary search tree would give you a sorted list.

Okay, now you should be able to figure out the post order traversal. This is what we will get for post order traversal. I'll leave it for you to see whether this is correct or not.

I'll stop here now. In next lesson, we will discuss implementation of these tree traversal algorithms. Thanks for watching.

In this lesson, we are going to write code for level order traversal of a binary tree. As we have discussed in our previous lesson in level order traversal, we visit all nodes at a particular depth or level in the tree before visiting the nodes at next deeper level. For this binary tree that I'm showing here, if I have to traverse the tree and print the data in nodes in level order, then this is how we will go.

We will start at level zero and print F and now we are done with level zero so we can go to level one and we can visit the nodes at level one from left to right. From F we will go to D and from D we will go to J. Now level one is done so we can go to level two. So we will go like B, E, G and then K and now we can go to next.


ds-14

