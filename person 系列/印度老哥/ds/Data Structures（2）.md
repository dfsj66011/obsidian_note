
## 六、C/C++ 实现链表在开头插入节点

在之前的课程中，我们学习了如何将链表的逻辑视图映射到 C 或 C++ 程序中，了解了如何实现两个基本操作：一个是遍历链表，另一个是在链表末尾插入节点。在本节课中，我们将看到一个在链表开头插入节点的运行代码。让我们开始吧。我将在这里编写一个 C 程序。

在我们的程序中，首先要做的是定义一个节点。节点在 C 语言中是一个结构体，它包含两个字段：一个用于存储数据。假设我们要创建一个整数链表，那么我们的数据类型就是整数。如果我们想创建一个字符链表，那么这里的数据类型就应该是字符。

因此，我们将有另一个字段指向节点，该节点将存储下一个节点的地址。我们可以将这个变量命名为“link”，或者有些人也喜欢将其命名为 “next”，因为这样听起来更直观。这个变量将存储链表中下一个节点的地址。在 C 语言中，每当我们需要声明节点或指向节点的指针时，我们必须写 “struct Node” 或“struct Node*”。而在 C++中，我们只需要写“Node*”。这就是一个区别。好的，这就是我们节点的定义。

现在要创建一个链表，我们需要创建一个变量，这个变量将指向节点，并存储链表中第一个节点的地址，也就是我们所说的头节点。所以我将在这里创建一个指向节点的指针，struct Node*，我们可以随意命名这个变量，但为了便于理解，我们通常将其命名为 head。现在我已经将这个变量声明为全局变量，我没有在任何函数内部声明这个变量。稍后我会解释为什么要这样做。

现在我来编写主方法，这是我的程序入口。首先我要做的是将 head 设为 NULL，这意味着这个指针变量不指向任何地方。所以现在，这个链表是空的。到目前为止，我们在代码中所做的是创建了一个名为 head 的全局变量，它是一个指向节点的指针类型，而这个指针变量的值为 null。因此到目前为止，这个链表是空的。现在，我想在程序中做的是要求用户输入一些数字。

我想把这些数字都插入到链表中。所以我会打印类似“要输入多少个数字”的提示，假设用户想输入 n 个数字。我会把这个数字收集到变量 `n` 中。然后我会定义另一个变量 `i` 来运行循环。所以我在这里运行一个循环。如果是 C++，我可以在循环内部直接声明这个整数变量。现在我要像这样写一个打印语句。

我将定义另一个变量 `x`，每次我都会从用户那里获取这个变量 `x` 作为输入。现在，我将通过调用`Insert` 方法将这个特定的数字 `x`、这个特定的整数 `x` 插入到链表中。然后每次插入后，我们将打印链表中所有节点的值，即链表中所有节点的数值。通过调用一个名为 `Print` 的函数，这个函数 `Print` 将没有任何参数。当然，我们需要实现这两个函数 `Insert` 和 `Print` 。让我先写下这两个函数的定义。

那么让我们来实现这两个函数：`Insert` 和 `Print`。首先我们来实现 `Insert` 函数，它将在链表的开头插入一个节点。在 `Insert` 函数中，我们需要做的第一件事是在 C 语言中创建一个节点，我们可以使用 `malloc` 函数来创建节点，我们之前已经讨论过这个问题，`malloc` 返回一个指向内存块起始地址的指针，这里我们需要进行类型转换，因为 `malloc` 返回的是一个通用指针，而我们需要的是一个指向节点的指针变量。

然后，只有当我们使用星号进行解引用时，才能访问节点的字段。因此，数据部分将是 x，并且对于这个特定的语法，我们有一个替代语法，我们可以简单地写成类似于 temp 和这个箭头的形式，它将表示相同的意思。在插入函数中，这两行代码更常见，我们所做的只是创建一个节点，假设我们得到了这个节点，并假设我们为这个节点得到的地址是 100。现在有一个变量 temp，我们用它来存储地址。每当我们创建一个节点时，可以做一件事：我们可以将数据设置为我们想要的值，并最初将链接字段设置为null。如果需要，我们可以修改链接字段。所以我会再写一条语句：temp.next 等于 null。

记住，temp 在这里是一个指针变量，我们通过解引用这个指针变量来修改特定节点的值。temp也会在内存中占用一些空间。这就是为什么我为指针变量 head 和 temp 都画了这个矩形块。节点包含两个部分，一部分用于指针变量，另一部分用于数据。所以这部分，也就是链接部分，我们现在可以在这里写 null，或者也可以这样写，意思是一样的。从逻辑上讲，它们表示的含义相同。

现在，如果我们想在链表的开头插入这个节点，可能会出现两种情况。一种是链表为空，就像这种情况，那么我们唯一需要做的就是将头指针指向这个特定的节点，而不是指向空。因此，我会写一个语句，比如 head = temp。现在头指针中的值将是地址 100。这就是我们所说的指针变量指向特定节点的含义——我们存储该节点的地址。这就是我们的链表。

在我们插入第一个节点之后，现在让我们看看如果链表不为空（就像我们现在的情况），如何在开头插入一个节点。再次，我们可以创建一个节点，在这里填入作为参数传递的值 x。最初，我们可以将链接字段设置为 null。假设这个节点在内存中的地址是 150。我们有一个变量 temp，通过它来引用这个特定的内存块。现在，与之前的情况不同，如果我们仅仅将 head 设置为等于 temp，这是不够的，因为我们还需要建立这个链接，我们需要将新创建的节点的 next 或 link 设置为之前的 head。

所以我们可以这样做：如果头节点不等于空，或者说链表不为空，首先将 temp 的下一个节点指向头节点。这样我们先建立这个链接，这里的地址会是 100。然后我们将头节点设置为 temp。所以我们切断这个链接，并将头指针指向这个新创建的节点。这就是我们在链表开头插入第二个节点后修改过的链表。现在最后一点，这一行代码，第三行的 temp.next = null，这仅在链表为空时才会用到。

如果你看到列表为空时，头节点已经是 null。因此我们可以避免写两条语句，只需简单地写这一条语句：temp.next = head。这样也能覆盖列表为空的情况。现在这个程序中唯一剩下的工作就是实现这个打印函数了。

让我们来实现这个打印函数。接下来我要做的是创建一个名为 temp 的节点指针局部变量。我需要在这里写结构体节点，我在 C 单元里总是漏掉这个，你需要这样写。我想把这个设为头节点的地址。所以这个全局变量保存了头节点的地址。现在我要遍历这个链表。所以我会写一个这样的循环。当 temp 不等于 null 时，我会一直用这个语句 temp=temp.next 来访问下一个节点。

在每个步骤中，我都会打印该节点中的值，即 temp.data。现在，我要再写两个打印语句，一个放在这个 while 循环之外，另一个放在这个 while 循环之后，用来打印一个换行符。那么，为什么我们要使用一个临时变量呢？因为我们不想修改 head，否则我们会丢失第一个节点的引用。首先，我们将头部的地址存储在另一个临时变量中。然后，我们通过使用 temp=temp.next 来遍历列表，修改这个临时变量中的地址。

现在让我们运行这个程序，看看会发生什么。这是在询问你想在列表中插入多少个数字。假设我们想插入五个数字。最初，列表是空的。假设我们要插入的第一个数字是 2。在每个阶段，我们都会打印列表。现在列表的第一个元素是 2，最后一个元素也是 2，我们将插入另一个数字。现在列表的开头插入了 5。同样，我们插入了 8，并且在列表的开头也插入了 8。好的，现在让我们插入数字 1，列表现在是 1 8 5 2。最后，我插入了数字 10。所以最终的列表是 10 1 8 5 2。这看起来是可行的。

现在，如果我们在 C++ 中编写这段代码，我们可以做几件事：我们可以编写一个类并以面向对象的方式组织代码，我们也可以用 `new` 运算符代替 `malloc` 函数。

现在我们回到刚才的话题，我们已经将这个头节点声明为全局变量。如果它不是全局变量，而是作为局部变量在 main 函数内部声明，会怎么样呢？那么我将移除这个全局声明。现在，这个头节点在其他函数中将无法访问。因此，我们需要将第一个节点的地址作为参数传递给其他函数，包括打印和插入这两个函数。对于这个打印方法，我们会传递一个参数，假设我们将这个参数命名为 head，我们也可以将其命名为 a、temp 或其他任何名称。如果我们把这个参数命名为 head，那么在打印函数中的这个 head 将是该函数的局部变量，而不是主函数中的这个 head，这两个 head 是不同的，这两个变量也是不同的。

当主函数调用 Print 并传递其 head 时，主函数中这个特定 head 的值会被复制到 Print 函数中的另一个 head。因此，在 Print 函数中，我们可能不需要使用这个临时变量 temp，我们可以直接使用这个 head 变量本身来遍历链表。这样做是可行的，因为我们并没有在主函数中修改这个 head。与打印函数类似，在 Insert 函数中我们需要传入第一个节点的地址。而这个头节点同样只是一个副本，它也是一个局部变量。因此，在我们修改链表后，主方法中的头节点也应被修改。有两种方法可以实现：一种是从这个方法返回指向节点的指针。因此，在主方法中，插入函数将接受另一个参数 head，我们需要将返回值再次赋给 head，以便对其进行修改。现在这段代码就能正常运行了。哎呀，我忘记在这里写一个返回语句了，应该返回头节点。我们可以像之前一样运行这个程序，输入所有值后可以看到链表正确地构建起来。

其实还有另一种方法，不需要让这个 Insert 函数返回头节点的地址，我们可以通过引用传递这个特定的变量 head。也就是说，我们可以传递插入函数的参数为 &head（head 本身已经是一个指向节点的指针）。因此在插入函数中，我们需要接收一个指向指针的节点参数（即 Node**）。为了避免混淆，这次我们给这个变量换个名字，就叫它 pointerToHead 吧。

因此，要获取头节点，我们需要编写类似这样的代码：必须解引用这个特定的变量，并在各处使用星号指针指向头节点。返回类型将是 void。有时我们希望将这个变量的名称改为 head，这个局部变量叫 head 无关紧要，但我们必须确保正确使用它。现在这段代码也能正常运行了。如你所见，我们可以插入节点。看起来进展很顺利。如果你不理解这些作用域的概念，可以参考本视频简介中的补充资料。以上就是如何在链表开头插入节点的操作。感谢观看。

## 七、


In our previous lesson, we had written code to insert a node at the beginning of the linked list. Now in this lesson, we will write program to insert a node at any given position in the linked list. So let me first explain the problem in a logical view.

Let's say we have a linked list of integers here. There are three nodes in this linked list. Let us say they are addresses 200, 100 and 250 respectively in the memory and we have a variable head and that is pointer to node that stores the address of the first node in the list.

Now let us say we number these nodes, we number these positions on a one based index. So this is the first node in the list and this is the second node and this is the third node and we want we want to write a function insert that will take the data to be inserted in the list and the position at which we want to insert this particular data. So we will be inserting a node at that particular position with this data, there will be a couple of scenarios, the list could be empty.

So this variable head will be null, or this argument being passed to the insert function, the position n could be an invalid position. For example, five is an invalid position here. For this linked list, the maximum possible position at which we can insert a node in this list will be four.

If we want to insert at position one, we want to insert at the beginning and if we want to insert at position four, we want to insert at end. So our insert function should gracefully handle all these scenarios. Let us assume for the sake of simplicity, for the sake of simplifying our implementation that we always give a valid position, we will always give a valid position so that we do not have to handle the error condition in case of invalid position.The implementation logic for this function will be pretty straightforward. We will first create a node. Let's say in this example, we want to insert a node with value eight at third position in the list.

So I'll set the data here in the node, the data part is eight. Now to insert a node at nth position, we will first have to go to the n-1th node. In this case, n is equal to three, so we will go to the second node.

Now the first thing that we will have to do is, we will have to set the link field of this newly created node equal to the link field of this n-1th node. So, we will have to build this link. Let's say the address that we get for this newly created node is 150.

Once we build this link, we can break this link and set the link of this newly created node as address of this, set the link of this n-1th node as address of this newly created node. We may have special cases in our implementation, like the list may be empty, or maybe we may want to insert a node at the beginning. Let's say we will take care of special cases, if any, in our actual implementation.

So, now let's move on to implement this particular function in our program. In my C program, the first thing that I need to do is, I want to define a node. So, node will be a structure, we have seen this earlier.

So, node has these two fields, one data of type integer and another next of type pointer to node. Now to create a linked list, the first thing that I need to create is a pointer to node that will always store the address of the first node or the head node in the linked list. So, I will create struct node star, let's name this variable head.

And once again, I have created this variable as a global variable. To understand linked list implementation, we need to understand what goes where, what variable sits in what section of the memory and what is the scope of these variables, what goes in the stack section of the memory and what goes in the heap section of the memory. So, this time as we write this code, we will see what goes where.

In the main method, first I'll set this head as null to say that initially the list is empty. So, let us now see what has gone where so far in our program, in what section of the memory. The memory that is allocated to our program or application is typically divided into these four parts or these four sections.

We have talked about this in our lesson on dynamic memory allocation. There is a link to our lesson on dynamic memory allocation in the description of this video. I'll quickly talk about what these sections are.

One section of the application's memory is used to store all the instructions that need to be executed. Another section is allocated to store the global variables that live for the entire lifetime of the program of the application. One section of the memory which is called stack is used to store all the information about function call executions to store all the local variables and these three sections that we talked about are fixed in size.

Their size is decided at compile time. The last section that we call heap or free store is not fixed and we can request memory from the heap during runtime and that's what we do when we use malloc or new operator. Now, I have drawn these three sections of the memory stack, heap and the section to store the global variables.In our program, we have declared a global variable named head which is pointer to node. So, it will go and sit here and this variable is like anyone can access it. Initially, value here is null.

Now, in my program what I want to do is, I first want to define two functions insert and this function should take two arguments data and the position at which I want to insert a node and insert that particular node at that position, insert data at that position in the list and another function print that will simply print all the numbers in the linked list. Now, in the main method, I want to make a sequence of function calls. First, I want to insert number two.

The list is empty right now. So, I can only insert at position one. So, after this insert list will be having this one number, this particular number two and let's say again I want to insert number three at position two.

So, this will be our list after this insertion and I will make two more insertions and finally I'll print the list. So, this is my main method. I could have also asked a user to input a number and position but let's say we go this way this time.

Now, let us first implement insert. I'll move this print above. So, the first thing that I want to do in this method is I want to create a node.

So, I will make a call to malloc. In C++ we can simply write a new node for this call to malloc and this looks a lot cleaner. Let's go C++ way this time.Now, what I can do is I can first set the data field and set the link initially as null. I have named this variable temp1 because I want to use another temp variable in this function. I'll come to that in a while.

We first need to handle one special case when we want to insert at the head, when we want to insert at the first position. So, if n is equal to one, we simply want to set the link field of the newly created node as whatever the existing head is and then adjust this variable to point to the new head which will be this newly created node and we will be done at this stage. So, we will not execute any further and return from this function.If you can see, this will work even when the list is empty because the head will be null in that case. I'll show a simulation in the memory in a while. So, hold on till then.

Things will be pretty clear to you after that. Now, for all other cases, we will first need to go to the n-1th node as we had discussed in our logic initially. So, what I'll do is I'll create another pointer to node, name this variable temp2 and we will start at the head and then we will run a loop and go to the n-1th node, something like this.

We will run the loop n-2 times because right now we are pointing to head which is the first node. So, if we do this temp2 equal temp2.next n-2 times, we will be pointing temp2 to n-1th node and now the first thing that we need to do is set the next or the link field of newly created node as the link field of this n-1th node and then we can adjust the link of this n-1th node to point to our newly created node. And now I am writing this print here.

I have written this print here. We have used a temporary variable, a temporary pointer to node, initially pointed it to head and we have traversed the whole list. Ok, so let us now run this program and see what happens.

We are getting this output which seems to be correct. The list should be 4, 5, 2, 3 in this order. Now, I have this code.

I will run through this code and show you what's happening in the memory. When the program starts execution, initially the main method is invoked. Some part of the memory from the stack is allocated for execution of a function.

All the local variables and the state of execution of this function is saved in this particular section. We also call this stack frame of a function. Here in this main method we have not declared any local variable.

We just set head to null which we have already done here. Now, the next line is a call to function insert. So, the machine will set the execution of this particular method main on hold and go on to execute this call to insert.

So, insert comes into this stack and insert has couple of local variables. It has two arguments, data and this variable n. This stack frame will be a little larger because we will have couple of local variables and now we create this another local variable which is a pointer to node temp1 and we use the new operator to create a memory block in the heap and this guy temp1 initially stores the address of this memory block. Let's say this memory block is at address 150.So, this guy stores the address 150. When we request some memory to store something on the heap using new or malloc, we do not get a variable name and the only way to access it is through a pointer variable. So, this pointer variable is the remote control here kind of.

So, here when we say temp1.data is equal to this much, through this pointer which is our remote we are going and writing this value 2 here and then we are saying temp.next equal null. So, null is nothing but address 0. So, we are writing address 0 here. So, we have created a node and in our first call n is equal to 1. So, we will come to this condition.

Now, we want to set temp1.next equal head. temp1.next is this section, this second field and this is already equal to head. Head is null here and this is already null.

Null is nothing but 0. The only reason we set temp.next equal head will work for empty cases because head would be null and now we are saying head is equal to temp1. So, head guy now points to this because it stores address 150 like temp1. And in this first call to insert, after this we will return.

So, the execution of insert will finish and now the control returns to the main method. We come to this line where we make another call to insert. With different arguments this time we pass number 3 to be inserted at position 2. Now, once again memory in the stack frame will be allocated for this particular call to insert.The stack frame allocation is corresponding to a particular call. So, each time the function execution finishes, all the local variables are gone from the memory. Now, once again in this call we create a new node.

We keep the address initially in this temporary variable temp1. Let's say we get this node at address 100 this time. Now, n is not equal to 1. We will move on to create another temporary variable temp2.

Now, we are not creating a new node and storing the temp2 here. We are saying temp2 is initially equal to head. So, we store the address 150.

So, initially we make this guy point to the head node and now we want to run this loop and want to go keep going to the next node until we reach n-1th node. In this case n is equal to 2. So, this loop will not execute this statement even once. n-1th node is the first node itself.Now, we execute these two lines. The next of the newly created node will be set first. So, we will build this link.

Oops! No! temp2.next is 0 only. So, even after reset this will be 0. And now we are setting temp2.next as temp1. So, we are building this link.

And now this call to insert will finish. So, we go back to the main method. So, this is how things will happen for other calls also.

ds-3