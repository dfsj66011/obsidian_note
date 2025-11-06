
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


在上节课中，我们总体讨论了二叉树。


This is it for this lesson. Thanks for watching. In our previous lesson, we talked about binary trees in general.

Now, in this lesson, we are going to talk about binary search tree, a special kind of binary tree which is an efficient structure to organize data for quick search as well as quick update. But before I start talking about binary search trees, I want you to think of a problem. What data structure will you use to store a modifiable collection? So, let's say you have a collection and it can be a collection of any data type.

Records in the collection can be of any type. Now, you want to store this collection in computer's memory in some structure and then you want to be able to quickly search for a record in the collection and you also want to be able to modify the collection. You want to be able to insert an element in the collection or remove an element from the collection.

So, what data structure will you use? Well, you can use an array or a linked list. These are two well-known data structures in which we can store a collection. Now, what will be the running time of these operations, search, insertion or removal if we will use an array or a linked list? Let's first talk about arrays and for sake of simplicity, let's say we want to store integers.

To store a modifiable list or collection of integers, we can create a large enough array and we can store the records in some part of the array. We can keep the end of the list marked. In this array that I am showing here, we have integers from 0 till 3, we have records from 0 till 3 and rest of the array is available space.

Now, to search some x in the collection, we will have to scan the array from index 0 till end and in the worst case, we may have to look at all the elements in the list. If n is the number of elements in the list, time taken will be proportional to n or in other words, we can say that time complexity of this operation will be O of n. Ok, now what will be the cost of insertion? Let's say we want to insert number 5 in this list. So, if there is some available space, all these cells in yellow are available, we can add one more cell by incrementing this marker end and we can fill in the integer to be added.

The time taken for this operation will be constant. Running time will not depend upon number of elements in the collection. So, we can say that time complexity will be O of 1. Ok, now what about removal? Let's say we want to remove 1 from the collection.

What we'll have to do is, we'll have to shift all records to the right of 1 by one position to the left and then we can decrement end. The cost of removal in worst case once again will be O of n. In worst case, we will have to shift n-1 elements. Here the cost of insertion will be O of 1 if the array will have some available space.So, the array has to be large enough. If the array gets filled, what we can do is, we can create a new larger array. Typically, we create an array twice the size of the filled up array.

So, we can create a new larger array and then we can copy the content of the filled up array into this new larger array. The copy operation will cost us O of n. We have discussed this idea of dynamic array quite a bit in our previous lessons. So, insertion will be O of 1 if array is not filled up and it will be O of n if array is filled up.

For now, let's just assume that the array will always be large enough. Let's now discuss the cost of these operations if we will use a linked list. If we would use a linked list, I have drawn a linked list of integers here.

Data type can be anything. The cost of search operation once again will be O of n where n is number of records in the collection or number of nodes in the linked list. To search in worst case, we will have to traverse the whole list.

We will have to look at all the nodes. The cost of insertion in a linked list is O of 1 at head and its O of n at tail. We can choose to insert at head to keep the cost low.So, running time of insertion we can say is O of 1 or in other words, we will take constant time. Removal once again will be O of n. We will first have to traverse the linked list and search the record and in worst case, we may have to look at all the nodes. Ok, so this is the cost of operations if we are going to use array or linked list.

Insertion definitely is fast but how good is big O of n for an operation like search? What do you think? If we are searching for a record x, then in the worst case, we will have to compare this record x with all the n records in the collection. Let's say our machine can perform a million comparisons in one second. So, we can say that machine can perform 10 to the power 6 comparisons in one second.

So, cost of one comparison will be 10 to the power minus 6 second. Machines in today's world deal with really large data. It's very much possible for real world data to have 100 million or billion records.

A lot of countries in this world have population more than 100 million. Two countries have more than a billion people living in them. If we will have data about all the people living in a country, then it can easily be 100 million records.

Ok, so if we are saying that the cost of one comparison is 10 to the power minus 6 second, if n will be 100 million, time taken will be 100 seconds. 100 seconds for a search is not reasonable and search may be a frequently performed operation. Can we do something better? Can we do better than big O of n? Well, in an array, we can perform binary search if it's sorted and the running time of binary search is big O of log n which is the best running time to have.

I have drawn this array of integers here. Records in the array are sorted. Here the data type is integer.

For some other data type, for some complex data type, we should be able to sort the collection based on some property or some key of the records. We should be able to compare the keys of records and the comparison logic will be different for different data types. For a collection of strings for example, we may want to have the records sorted in dictionary or lexicographical order.So, we will compare and see which string will come first in dictionary order. Now, this is the requirement that we have for binary search. The data structure should be an array and the records must be sorted.

Ok, so the cost of search operation can be minimized if we will use a sorted array but in insertion or removal, we will have to make sure that the array is sorted afterwards. In this array, if I want to insert number 5 at this stage, I can't simply put 5 at index 6. What I'll have to do is, I'll first have to find the position at which I can insert 5 in the sorted list. We can find the position in order of log n time using binary search.

We can perform a binary search to find the first integer greater than 5 in the list. So, we can find the position quickly. In this case, it's index 2 but then we will have to shift all the records starting this position, one position to the right and now I can insert 5. So, even though we can find the position at which a record should be inserted quickly in O of log n, this shifting in worst case will cost us O of n. So, the running time overall for an insertion will be O of n and similarly the cost of removal will also be O of n. We will have to shift some records.

Ok, so when we are using sorted array, cost of search operation is minimized. In binary search for n records, we will have at max log n to the base 2 comparisons. So, if we can compare, if we can perform million comparisons in a second, then for n equal 2 to the power 31 which is greater than 2 billion, we are going to take only 31 microseconds.Log of 2 to the power 31 to base 2 will be 31. Ok, we are fine with search now. We will be good for any practical value of n but what about insertion and removal? They are still big O of n. Can we do something better here? Well, if we will use this data structure called binary search tree, I am writing it in short BST for binary search tree, then the cost of all these three operations can be big O of log n. In average case, the cost of all the operations will be big O of n. In worst case, but we can avoid the worst case by making sure that the tree is always balanced.

We have talked about balanced binary tree in our previous lesson. Binary search tree is only a special kind of binary tree. To make sure that the cost of these operations is always big O of log n, we should keep the binary search tree balanced.

We will talk about this in detail later. Let's first see what a binary search tree is and how cost of these operations is minimized when we use a binary search tree. Binary search tree is a binary tree in which for each node, value of all the nodes in left subtree is lesser and value of all the nodes in right subtree is greater.I have drawn binary tree as a recursive structure here. As we know, in a binary tree, each node can have at most two children. We can call one of the children left child.

If we will look at the tree as a recursive structure, left child will be the root of left subtree and similarly right child will be the root of right subtree. Now, for a binary tree to be called binary search tree, value of all the nodes in left subtree must be lesser. Or we can say lesser or equal to handle duplicates and the value of all the nodes in right subtree must be greater and this must be true for all the nodes.

So, in this recursive structure here, both left and right subtrees must also be binary search trees. I will draw a binary search tree of integers. Now, I have drawn a binary search tree of integers here.

Let's see whether this property that for each node, value of all the nodes in left subtree must be lesser or equal and value of all the nodes in right subtree must be greater is true or not. Let's first look at the root node. Nodes in the left subtree have values 10, 8 and 12.

So, they are all lesser than 15 and in right subtree, we have 17, 20 and 25. They are all greater than 15. So, we are good for the root node.

Now, let's look at this node with value 10. In left, we have 8 which is lesser. In right, we have 12 which is greater.

So, we are good. We are good for this node too having value 20 and we don't need to bother about leaf nodes because they do not have children. So, this is a binary search tree.

Now, what if I change this value 12 to 16. Now, is this still a binary search tree? Well, for node with value 10, we are good. The node with value 16 is in its right.

So, not a problem. But, for the root node, we have a node in left subtree with higher value now. So, this tree is not a binary search tree.

I'll revert back and make the value 12 again. Now, as we were saying, we can search, insert or delete in a binary search tree in O of log n time in average case. How is it really possible? Let's first talk about search.

If these integers that I have here in the tree were in a sorted array, we could have performed binary search. And, what do we do in binary search? Let's say we want to search number 10 in this array. What we do in binary search is, we first define the complete list as our search space.

The number can exist only within the search space. I'll mark search space using these two pointers, start and end. Now, we compare the number to be searched or the element to be searched with mid element of the search space or the median.

And, if the record being searched, if the element being searched is lesser, we go searching in the left half, else we go searching in the right half. In case of equality, we have found the element. In this case, 10 is lesser than 15.

So, we will go searching towards left. Our search space is reduced now to half. Once again, we will compare to the mid element and bingo, this time we have got a match.

In binary search, we start with n elements in search space and then if mid element is not the element that we are looking for, we reduce the search space to n by 2 and we go on reducing the search space to half till we either find the record that we are looking for or we get to only one element in search space and be done. In this whole reduction, if we will go from n to n by 2 to n by 4 to n by 8 and so on, we will have log n to the base 2 steps. If we are taking k steps, then n upon 2 to the power k will be equal to 1 which will imply 2 to the power k will be equal to n and k will be equal to log n to the base 2. So, this is why running time of binary search is O of log n. Now, if we will use this binary search tree to store the integers, search operation will be very similar.

Let's say we want to search for number 12. What we will do is, we will start at root and then we will compare the value to be searched, the integer to be searched with value of root. If it's equal, we are done with the search.

If it's lesser, we know that we need to go to the left subtree because in a binary search tree, all the elements in left subtree are lesser and all the elements in right subtree are greater. Now, we will go and look at the left child of node with value 15. We know that number 12 that we are looking for can exist in this subtree only and anything apart from this subtree is discarded.

So, we have reduced the search space to only these 3 nodes having value 10, 8 and 12. Now, once again we will compare 12 with 10. We are not equal.

12 is greater. So, we know that we need to go looking in right subtree of this node with value 10. So, now our search space is reduced to just one node.Once again, we will compare the value here at this node and we have a match. So, searching an element in binary search tree is basically this traversal in which at each step, we will go either towards left or right and hence at each step, we will discard one of the subtrees. If the tree is balanced, we call a tree balanced if for all nodes, the difference between the heights of left and right subtrees is not greater than 1. So, if the tree is balanced, we will start with a search space of n nodes and when we will discard one of the subtrees, we will discard n by 2 nodes.

So, our search space will be reduced to n by 2 and then in next step, we will reduce the search space to n by 4. We will go on reducing like this till we find the element or till our search space is reduced to only one node when we will be done. So, the search here is also a binary search and that's why the name binary search tree. This tree that I am showing here is balanced.

In fact, this is a perfect binary tree but with same records, we can have an unbalanced tree like this. This tree has got the same integer values as we had in the previous structure and this is also a binary search tree but this is unbalanced. This is as good as a linked list.

In this tree, there is no right subtree for any of the nodes.


ds-12

