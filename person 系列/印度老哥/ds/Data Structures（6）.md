
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



In our previous lesson, we saw what binary search trees are. Now in this lesson, we are going to implement binary search tree.

We will be writing some code for binary search tree. Prerequisite for this lesson is that you must understand the concepts of pointers and dynamic memory allocation in C C++. If you have already followed this series and seen our lessons on linked list, then implementation of binary search tree or binary search tree in general is not going to be very different.

We will have nodes and links here as well. Okay, so let's get started. Binary search tree or BST as we know is a binary tree in which for each node, value of all the nodes in left subtree is lesser or equal and value of all the nodes in right subtree is greater.

We can draw BST as a recursive structure like this. Value of all the nodes in left subtree must be lesser or equal and value of all the nodes in right subtree must be greater and this must be true for all nodes and not just the root node. So, in this recursive definition here, both left and right subtrees must also be binary search trees.

I have drawn a binary search tree of integers here. Now the question is how can we create this non-linear logical structure in computer's memory. I had talked about this briefly when we had discussed binary trees.

The most popular way is dynamically created nodes linked to each other using pointers or references. Just the way we do it for linked lists. Because in a binary search tree or in a binary tree in general, each node can have at most two children.

We can define node as an object with three fields. Something like what I am showing here. We can have a field to store data, another to store address or reference to left child and another to store address or reference to right child.If there is no left or right child for a node, reference can be set as null. In C or C++, we can define node like this. There is a field to store data.

Here the data type is integer but it can be anything. There is one field which is pointer to node. Node asterisk means pointer to node.

This one is to store the address of left child and we have another one to store the address of right child. This definition of node is very similar to definition of node for doubly linked list. Remember in doubly linked list also each node had two links.

One to previous node and another to next node. But doubly linked list was a linear arrangement. This definition of node is for a binary tree.

We could also name this something like BST node. But node is also fine. Let's go with node.

Now in our implementation just like linked list, all the nodes will be created in dynamic memory or heap section of applications memory using malloc function in C or new operator in C++. We can use malloc in C++ as well. Now as we know any object created in dynamic memory or heap section of applications memory cannot have a name or identifier.

It has to be accessed through a pointer. Malloc or new operator return as pointer to the object created in heap. If you want to revise some of these concepts of dynamic memory allocation, you can check the description of this video for link to a lesson.

It's really important that you understand this concept of stack and heap in applications memory really well. Now for a linked list, if you remember the information that we always keep with us is address of the head node. If we know the head node, we can access all other nodes using links.

In case of trees, the information that we always keep with us is address of the root node. If we know the root node, we can access all other nodes in the tree using links. To create a tree, we first need to declare a pointer to BST node.

I'll rather call node BST node here. BST for binary search tree. So to create a tree, we first need to declare a pointer to BST node that will always store the address of root node.

I have declared a pointer to node here named rootptr, ptr for pointer. In C, you can't just write BST node asterisk rootptr. You will have to write struct space BST node asterisk.

You will have to write struct here as well. I'm gonna write C++ here. But anyway, right now I'm trying to explain the logic.

We will not bother about my new details of implementation. In this logical structure of tree that I'm showing here, each node as you can see has three fields, three cells. Leftmost cell is to store the address of left child and rightmost cell is to store address of right child.

Let's say root node is at address 200 in memory and I'll assume some random addresses for all other nodes as well. Now I can fill in these left and right cells for each node with addresses of left and right children. In our definition, data is first field but in this logical structure, I'm showing data in the middle.

Okay, so for each node, I have filled in addresses for both left and right child. Address is zero or null if there is no child. Now as we were saying, identity of the tree is address of the root node.

We need to have a pointer to node in which we can store the address of the root node. We must have a variable of type pointer to node to store the address of root node. All these rectangles with three cells are nodes.

They are created using malloc or new operator and live in heap section of application's memory. We cannot have name or identifier for them. They are always accessed through pointers.This root ptr, root pointer has to be a local or global variable. We will discuss this in little more detail in some time. Quite often, we like to name this root pointer just root.We can do so but we must not confuse. This is pointer to root and not the root itself. To create a BST, as I was saying, we first need to declare this pointer.

Initially, we can set this pointer as null to say that the tree is empty. A tree with no node can be called empty tree and for empty tree, root pointer should be set as null. We can do this declaration and setting the root as null in main function in our program.

Actually, let's just write this code in a real compiler. I am writing C++ here. As you can see in the main function, I have declared this pointer to node which will always store the address of root node of my tree and I am initially setting this as null to say that the tree is empty.

With this much of code, we have created an empty tree but what's the point of having an empty tree? We should have some data in it. So, what I want to do now is I want to write a function to be able to insert a node in the tree. I will write a function named insert that will take address of the root node and data to be inserted as argument and this function will insert a node with this data at appropriate position in the tree.In the main function, I will make calls to this insert function passing it address of root and the data to be inserted. Let's say first I want to insert number 15 and then I want to insert number 10 and then number 20. We can insert more but let's first write the logic for insert function.

Before I write the logic for insert function, I want to write a function to create a new node in dynamic memory or heap. This function get new node should take an integer, the data to be inserted as argument, create a node in heap using new or malloc and return back the address of this new node. I am creating the new node here using this new operator.

The operator will return me the address of the newly created node which I am collecting in this variable of type pointer to BST node. In C instead of new operator, we will have to use malloc. We can use malloc in C++ as well.

C++ is only a superset of C. Malloc will also do here. Now anything in dynamic memory or heap is always accessed through pointer. Now using this pointer new node, we can access the fields of the newly created node.

I will have to dereference this pointer using asterisk operator. So I am writing asterisk new node and now I can access the fields. We have three fields in node, data and two pointers to node left and right.

I have set the data here. Instead of writing asterisk new node dot data, we have this alternate syntax that we could use. We could simply write new node arrow data and this will mean the same.

We have used this syntax quite a lot in our lessons on linked list. Now for the new node, we can set the left and right child as null and finally we can return the address of the new node. Ok, coming back to the insert function.

We can have couple of cases in insertion. First of all tree may be empty. For this first insertion when we are inserting this number 15, tree will be empty.

If tree is empty, we can simply create a new node and set it as root. With this statement root equal get new node. I am setting root as address of the new node but there is something not alright here.

This root is local variable of insert function and its scope is only within this function. We want this root, root in main to be modified. This guy is a local variable of main function.

There are two ways of doing this. We can either return the address of the new root. So, return type of insert function will be pointed to BST node and not void.

And here in the main function, we will have to write statement like root equal insert and the arguments. So, we will have to collect the return and update our root in main function. Another way is that we can pass the address of this root of main to the insert function.

This root is already a pointer to node. So, its address can be collected in a pointer to pointer. So, insert function, in insert function, first argument will be a pointer to pointer and here we can pass the address.

We will say ampersand root to pass the address. We can name this argument root or we can name this argument root ptr. We can name this whatever.

Now, what we need to do is we need to dereference this using asterisk operator to access the value in root of main and we can also set the value in root of main. So, here with this statement, we are setting the value and the return type now can be void. This pointer to pointer thing gets a little tricky.

I'll go with the former approach. Actually, there is another way. Instead of declaring root as a local variable in main function, we can declare root as global variable.

Global variable as we know has to be declared outside all the functions. If root would be global variable, it would be accessible to all the functions and we will not have to pass the address stored in it as argument. Anyway, coming back to the logic for insertion, as we were saying if the tree is empty, we can simply create a new node and we can simply set it as root.At this stage, we wanted to insert 15. If we will make call to the insert function, address of root is 0 or null. Null is only a macro for 0 and the second argument is the number to be inserted.

In this call to insert function, we will make call to get new node. Let's say we got this new node at address 200. Get new node function will return us address 200 which we can set as root here.

But this root is a local variable. We will return this address 200 back to the main function and in the main function, we are actually doing this root equal insert. So, in the main function, we are building this link.

Okay, our next call in the main function was to insert number 10. At this stage, root is 200. The address in root is 200 and the value to be inserted is 10.

Now, the tree is not empty. So, what do we do? If the tree is not empty, we can basically have two cases. If the data to be inserted is lesser or equal, we need to insert it in the left subtree of root and if the data to be inserted is greater, we need to insert it in right subtree of the root.

So, we can reduce this problem in a self-similar manner, in a recursive manner. Recursion is one thing that we are going to use almost all the time while working with trees. In this function, I'll say that if the data to be inserted is less than or equal to the data in root, then make a recursive call to insert data in left subtree.

The root of the left subtree will be the left child. So, in this recursive call, we are passing address of left child and data as argument and after the data is inserted in left subtree, the root of the left subtree can change. Insert function will return the address of the new root of the left subtree and we need to set it as left child of the current node.

In this example tree here, right now both left and right subtree are empty. We are trying to insert number 10. So, we have made call to this function insert.

From main function, we have called insert passing it address 200 and value or data 10. Now, 10 is lesser than 15. So, control will come to this line and a call will be made to insert data in left subtree.

Now, left subtree is empty. So, address of root for left subtree is 0. Data passed, data to be inserted passed as argument is 10. Now, this first insert call will wait for this insert below to finish and return.

For this last insert call, root is null. Let's say we got this node at address 150. Now, this insert call will return back 150 and execution of first insert call will resume at this line and now this particular address will be set as 150.

So, we will build this link and now this insert call can finish. It can return back the current root. Actually this return root should be there for all cases.

So, I am taking it out and I have it after all these conditions. Of course, we will have one more else here. If the data is greater, we need to go insert in right subtree.

The third call in insert function was to insert number 20. Now, this time we will go to this else statement. This statement in else, let's say we got this new node at address 300.

So, this guy will return 300. For this node at 200, right child will be set as 300 and now this call to insert can finish. The return will be 200.

Ok, at this stage what if a call is made to insert number 25. We are at root right now. The node with address 200.

25 is greater so we need to go and insert in right subtree. Right subtree is not empty this time. So, once again for this call also we will come to this else, last else because 25 is greater than 20.

Now, in this call we will go to the first if. A node will be created. Let's say we got this node in heap at address 500.

This particular call insert 025 will return 500 and finish. Now, for the node at 300, right child will be set as 500. So, this link will get built.

Now, this guy will return 300. The root for this subtree has not changed and this first call to insert will also wrap up. It will return 200.

So, we are looking good for all cases. This insert function will work for all cases. We could write this insert function without using recursion.

I encourage you to do so. You will have to use some temporary pointer to node and loops. Recursion is very intuitive here and recursion is intuitive in pretty much everything that we do with trees.So, it's really important that we understand recursion really well. Ok, I'll write one more function now to search some data in BST. In the main function here, I have made some more calls to insert.

Now, I want to write a function named search that should take as argument address of the root node and the data to be searched and this function should return me true if data is there in the tree, false otherwise. Once again we will have couple of cases. If the root is null, then we can return false.

If the data in root is equal to the data that we are looking for, then we can return true. Else, we can have two cases. Either we need to go and search in the left subtree or we need to go in the right subtree.

So, once again I am using recursion here. I am making recursive call to search function in these two cases. If you have understood the previous recursion, then this is very similar.

Let's test this code now. What I have done here is I have asked the user to enter a number to be searched and then I am making call to this search function and if this function is returning me true, I am printing found else I am printing not found. Let's run this code and see what happens.

I have moved multiple insert statements in one line because I am short of space here. Let's say we want to search for number 8. 8 is found and now let's say we want to search for 22. 22 is not found.

So, we are looking good. I will stop here now. You can check the description of this video for link to all the source code.We will do a lot more with trees in coming lessons. In our next lesson, we will go a little deeper and try to see how things move in various sections of application's memory. How things move in stack and heap sections of memory when we execute these functions.It will give you a lot of clarity. This is it for this lesson. Thanks for watching.

In our previous lesson, we wrote some code for binary search tree. We wrote functions to insert and search data in BST. Now in this lesson, we will go a little deeper and try to understand how things move in various sections of application's memory when these functions get executed and this will give you a lot of clarity.

This will give you some general insight into how memory is managed for execution of a program and how recursion which is so frequently used in case of trees works. The concepts that I am going to talk about in this lesson have been discussed earlier in some of our previous lessons but it will be good to go through these concepts again when we are implementing trees. So, here is the code that we had written.

We have this function get new node to create a new node in dynamic memory and then we have this function insert to insert a new node in the tree and then we have this function to search some data in the tree and finally this is the main function. You can check the description of this video for link to this source code. Now in main function here, we have this pointer to BST node named root to store the address of root node of my tree and I am initially setting it as null to create an empty tree and then I am making some calls to insert function to insert some data in the tree and finally I am asking user to input a number and I am making call to search function to find this number in the tree.

If the search function is returning me true, I am printing found else I am printing not found. Let's see what will happen in memory when this program will execute. The memory that is allocated to a program or application for its execution in a typical architecture can be divided into these four segments.

There is one segment called text segment to store all the instructions in the program. The instructions would be compiled instructions in machine language. There is another segment to store all the global variables.

A variable that is declared outside all the functions is called global variable. It is accessible to all the functions. The next segment stack is basically scratch space for function call execution.

All the local variables, the variables that are declared within functions live in stack and finally the fourth section heap which we also call the free store is the dynamic memory that can grow or shrink as per our need. The size of all other segments is fixed. The size of all other segments is decided at compile time but heap can grow during run time and we cannot control allocation or deallocation of memory in any other segment during run time but we can control allocation and deallocation in heap.

We have discussed all of this in detail in our lesson on dynamic memory allocation. You can check the description for a link. Now what I am going to do here is I am going to draw stack and heap sections as these two rectangular containers.

I am kind of zooming into these two sections. Now I will show you how things will move in these two sections of applications memory when this program will execute. When this program will start execution, first the main function will be called.

Now whenever a function is called, some amount of memory from the stack is allocated for its execution. The allocated memory is called stack frame of the function call. All the local variables and the state of execution of the function call would be stored in the stack frame of the function call.

In the main function we have this local variable root which is pointer to BST node. So I am showing root here in this stack frame. We will execute the instructions sequentially.

In the first line in main function, we have declared root and we are initializing it and setting it as null. Null is only a macro for address 0. So here in this figure, I am setting address in root as 0. Now in the next line, we are making a call to insert function. So what will happen is execution of main will pause at this stage and a new stack frame will be allocated for execution of insert.

Main will wait for this insert above to finish and return. Once this insert call finishes, main will resume at line 2. We have these two local variables root and data in insert function in which we are collecting the arguments. Now for this call to insert function, we will go inside the first if condition here because root is null.

At this line, we will make call to get new node function. So once again execution of this insert call will pause and a new stack frame will be allocated for execution of get new node function. We have two local variables in get new node, data in which we are collecting the argument and this pointer to BST node named new node.

Now in this function, we are using new operator to create a BST node in heap. Let's say we got a new node at address 200. New operator will return us this address 200.

So this address will be set here in new node. So we have this link here and now using this pointer new node, we are setting value in these three fields of node. Let's say the first field is to store data.

So we are setting value 15 here and let's say this second cell is to store address of left child. This is being set as null and the address of right child is also being set as null and now get new node will return the address of new node and finish its execution. Whenever a function call finishes, the stack frame allocated to it is reclaimed.

Call to insert function will resume at this line and the return of get new node address 200 will be set in this root which is local variable for insert call and now insert function, this particular call to insert function will return the address of root, the address stored in this variable root which is 200 now and finish. And now main will resume at this line and root of main will be set as 200. The return of this insert call, insert root 15 will be set here.

Now in the execution of main, control will go to the next line and we have this call to insert function to insert number 10. Once again, execution of main will be paused and a stack frame will be allocated for execution of insert. Now this time for insert call, root is not null.

So we will not go inside the first if. We will access the data field of this node at address 200 using this pointer named root in insert function and we will compare it with this value 10. 10 is lesser than 15 so we will go to this line and now we are making a recursive call here.

Recursion is a function calling itself and a function calling itself is not any different from a function A calling another function B. So what will happen here is that execution of this particular insert call will be paused and a new stack frame will be allocated for execution of this another insert call to which the arguments passed are address 0 in this local variable root, left child of node at address 200 is null. So we are passing 0 in root and in data we are passing 10. Now for this particular insert call, control will go inside first if and we will make a call to get new node function at this line.

So execution of this insert will pause and we will go to get new node function here. We are creating a new node in heap. Let's say we got this new node at address 150.

Now get new node will return 150 and finish. Execution of this call to insert will resume at this line. Return of get new node will be set here and now this call to insert will return address 150 and finish.

Insert below will resume at this line and now in this insert call, left child of this node at address 200 will be set as return of the previous insert call which is 150.


ds-13

