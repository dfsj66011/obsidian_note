
In our previous lesson, we introduced you to tree data structure. We discussed tree as a logical model and talked briefly about some of the applications of tree. Now in this lesson, we will talk a little bit more about binary trees.As we had seen in our previous lesson, binary tree is a tree with this property that each node in the tree can have at most two children. We will first talk about some general properties of binary tree and then we can discuss some special kind of binary trees like binary search tree, which is a really efficient structure for storing ordered data. In a binary tree, as we were saying each node can have at most two children.In this tree that I've drawn here, nodes have either zero or two children, we could have a node with just one child. I have added one more node here and now we have a node with just one child. Because each node in a binary tree can have at most two children, we call one of the children left child and another right child.

For the root node, this particular node is left child and this one is right child. A node may have both left and right child and these four nodes have both left and right child or a node can have either of left and right child. This one has got a left child but has not got a right child.

I'll add one more node here. Now, this node has a right child but does not have a left child. In a program, we would set the reference or pointer to left child as null.

So, we can say that for this node, left child is null and similarly for this node, we can say that the right child is null. For all the other nodes that do not have children that are leaf nodes, a node with zero child is called leaf node. For all these nodes, we can say that both left and right child are null.

Based on properties, we classify binary trees into different types. I'll draw some more binary trees here. If a tree has just one node, then also it's binary tree.

This structure is also a binary tree. This is also a binary tree. Remember, the only condition is that a node cannot have more than two children.

A binary tree is called strict binary tree or proper binary tree if each node can have either two or zero children. This tree that I'm showing here is not a strict binary tree because we have two nodes that have one child. I'll get rid of two nodes and now this is a strict binary tree.

We call a binary tree complete binary tree if all levels except possibly the last level are completely filled and all nodes are as far left as possible. All levels except possibly the last level will anyway be filled. So, the nodes at the last level, if it's not filled completely must be as far left as possible.

Right now, this tree is not a complete binary tree.


ds-11

