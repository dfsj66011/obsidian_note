
## 十一、递归法反转链表

在之前的课程中，我们学习了如何用递归遍历链表。我们编写了代码，用递归方式正向和反向打印链表的元素。实际上我们并没有反转链表，只是按逆序打印了元素。在这节课中，我们将使用递归来反转链表。这是另一个著名的编程面试题。如果我们有一个像这样的输入链表，这里有一个整数链表。

链表中有四个节点。这里的每个矩形块代表一个节点，分为两部分：第一部分用于存储数据，第二部分用于存储下一个节点的地址。当然，我们还会有一个变量来存储第一个节点（即头节点）的地址，这个变量我命名为 head，当然你也可以随意命名，我习惯称它为 head。

这是我们的输入列表。反转后，输出应该像这样。这个变量 head 应该存储原始列表中最后一个节点的地址。原始列表中的最后一个节点位于地址 250。然后我们会从 250 到 150，150 到 200，200 到 102。Null 其实就是地址零。

在之前的课程中，我们已经学习了如何用迭代方法反转链表。现在，让我们来看看如何用递归来解决这个问题。在我们的解决方案中，必须通过反转链接来调整链表，而不是通过移动数据或其他方式。那么让我们首先理解可以在递归方法中使用的逻辑。如果你还记得我们之前的课程，在那里我们使用递归来反向打印列表，即按相反顺序打印元素，那么递归为我们提供了一种在 C 或 C++ 程序中以编程方式向后遍历列表的方法，节点将是这样的结构。

因此，让我们首先看一下之前课程中的函数，即用于反向打印列表的递归函数，我们向这个函数传递一个节点的地址，最初我们传递的是头节点的地址。我们有一个退出条件，如果传入的地址为空，则直接返回；否则进行递归调用并传入下一个节点的地址。主方法通常会调用 reverse print 并传入头节点的地址。然后这个函数会先进行递归调用。然后当这个递归调用完成时，才会打印。所以我用 RP 作为 reverse print 的缩写。递归会像这样继续下去。

当它到达这个特定的调用时，如果参数为 null，就会返回。因此这个调用会结束，控制权会再次回到这个调用，参数是地址 250。现在我们正在打印地址 250 处节点的值，这个值将是 4。然后这个人完成了。接着我们继续打印 5。同样地，我们接着打印 6 和 2。

递归为我们提供了一种先正向遍历列表，再反向遍历列表的方法。现在让我们看看如何利用递归实现反转功能。为了简化和实现方便，假设 head 是一个全局变量。因此，所有功能都可以访问。现在我们将实现一个名为 reverse 的函数，该函数将接收一个节点的地址作为参数。最初，我们将把头节点的地址传递给这个函数。

现在我想在我的递归中做类似的事情，我想一直走到最后，我想一直进行递归调用，直到我到达最后一个节点。对于最后一个节点，链接部分将为 null。所以这是我退出递归的条件。这个退出条件将阻止我们在递归中无限进行下去。我在这里做的事情非常简单。一旦到达最后一个节点，我就会修改头指针，让它指向这个节点。所以递归会这样工作：从主方法中，我们将调用反转函数，并传入头节点的地址（地址 100）。

我们会来检查这个条件：如果 p.next 等于 null，不，对于地址为 100 的节点来说，它等于 200。因此，这个递归会一直进行，直到我们到达这个调用，即传递地址 250 给 reverse 函数的调用。然后我们会往下走，现在我们已经到达了这个退出条件，此时 head 将被设置为 P，链表将会变成这样。现在，reverse 250 的调用将结束，我们将返回到 reverse 150。在这个递归调用 reverse 函数之后，这里没有任何语句；如果这里有其他语句，那么在我们从 reverse 250 返回后，它们现在就会为 reverse 150 执行。这就是我们实际上如何以逆序遍历列表的方式。

如果你看到当反转 250 完成后，节点直到 250 已经被反转，因为头节点指向这个节点，并且这个节点的链接部分被设置为 null。所以直到 250 我们已经完成了反转。现在当我们来到 150 时，我们可以确保在完成反转 150 的执行后，列表直到 150 已经被反转。要实现这样的写法，我们需要做两件事：首先必须切断这个节点，并让这个指针指向另一个指针。因此我们要建立这个链接，同时必须切断原有链接，让这个指针指向空值。这样当我们完成这次调用后，地址150之前的节点就会被反转。

所以我在函数中写了这三行代码，它们会在递归调用之后执行。也就是说，它们会在递归展开时执行，此时我们正以相反的方向遍历列表。因此，当我们执行 reverse 150 并在递归后返回到这里时，我们就处在这一特定的代码行。

```c
if(p->next == NULL){
	head = p;
	return;
}
struct Node* q = p->next;
q->next = p;                   // 两句合并为一句  p->next->next = p;
p->next = NULL;
```

所以 P 会是 150，而 q 会是 p 的下一个。因此 q 会是250。这个家伙是 P，P，而这个家伙是 q。我们正在说设置 q->next=p。所以我们会将这个特定的字段设置为 100。所以我们正在建立这个链接并切断这个链接。现在我们可以看到将 p.next 设为 null。因此，我们正在构建这个链接，使 p.next 为 null，现在对 reverse 150 的调用完成了。当这次调用完成后，列表直到 150 的部分会被反转。如你所见，头部是 250。所以从 250 开始，我们将前往 150，然后从 150 我们将前往null。

所以在 150 之前，我们有一个反转后的列表。这就是当对 200 的反转调用完成时的情况，直到 200 为止，我们有一个反转后的列表。然后，我们再次回到对 100 的反转执行。而当 reverse 100 函数执行完毕并返回到主函数时，最终的内存状态就会呈现如此模样。我们在前一课中已经了解到递归执行时内存中的运作机制。在递归过程中，所有函数调用的执行状态都被保存在内存的栈区。在这个函数中，我们基本上就是在递归过程中将节点的地址存储在一个结构中。然后我们首先处理最后一个节点，使其成为反转链表的一部分。接着我们再次回到前一个节点，并不断重复这个过程。

观看上一课，详细了解递归在内存中如何运作的模拟过程。这里还有几点需要注意。其中一点是，与其写这两行代码，我可以将它们合并为一行，比如写成 P->next->next = P，这样表达的意思是一样的，只不过这种写法更加晦涩难懂。

还有一点我们假设的是 head 是一个全局变量。如果 head 不是全局变量，这个反转函数就需要返回修改后的 head 的地址。我把这个留作练习给你们去做。以上就是使用递归反转链表的方法。感谢观看。

## 十二、双向链表

大家好。在本系列课程中，我们已经对链表进行了较多讨论，了解了如何创建链表以及如何对链表执行各种操作。正如我们所知，链表是由称为节点的实体组成的集合。到目前为止，在我们所有的实现中，创建的链表每个节点都包含两个字段：一个用于存储数据，另一个用于存储下一个节点的地址。假设我们这里有一个整数链表。

所以我会在每个节点的数据字段中填入一些值。假设这些节点的地址分别是 200、250 和 350。我还会在每个节点中填入地址字段，第一个节点的地址字段将是第二个节点的地址，也就是 250。第二个节点中的地址字段将是第三个节点的地址，即 350。而第三个节点中的地址部分将为零或空。我们始终保留的链表的标识是头节点的地址或对头节点的引用。

假设我们有一个名为 head 的变量，仅用于存储头节点的地址。请记住，这个名为 head 的变量只是一个指向头节点的指针。理想情况下，我们应该将其命名为类似 head pointer 的名称，它只是指向头节点，而不是头节点本身。头节点就是这个家伙，链表中的第一个节点。好的，现在在我们展示的这个链表中，每个节点只有一个链接，即指向下一个节点的链接。在实际程序中，我在这里展示的链表节点将这样定义。

到目前为止，我们在所有课程中都是这样定义节点的：这里有两个字段，一个是整数类型用于存储数据，另一个是指向节点结构体的指针类型（struct node*）。我将这个字段命名为next。当我们默认说链表时，指的是这种我们也可以称之为单链表的列表。我们这里有一个单向链表。

本节课要讨论的是双向链表的概念。双向链表的原理其实很简单。在双向链表中，每个节点会有两个链接，一个指向下一个节点，另一个指向前一个节点。在编程实现上，这是我们在 C 或 C++ 中为双向链表定义节点的方式。我这里还有一个字段，它同样是一个指向节点的指针，这样我就可以存储一个节点的地址，通过这个字段我可以指向一个节点。

这个字段将用于存储前一个节点的地址。在逻辑表示中，我会这样绘制我的节点。现在，我有一个字段用于存储数据，一个用于存储前一个节点的地址，还有一个用于存储下一个节点的地址。假设我想创建一个整数的双向链表。我在这里创建了三个节点，假设这些节点的地址分别是 400、600 和 800。我会填充一些数据，假设每个节点中间的单元格用于存储数据。

最右边的单元格，假设是用来存储下一个节点的地址。因此，对于第一个节点，这个字段将是 600，这意味着我们有一个这样的链接。对于第二个节点，这个字段将是 800。对于第三个节点，此字段将为零。对于第一个节点，没有前一个节点。因此，这个最左边的单元格（本应包含前一个节点的地址）将为零或空。

第二个节点的前一个节点将是 400。第三个节点的前一个节点是地址为 600 的节点。当然，我们还会有一个变量来存储头节点的地址。好的，现在我们有一个包含三个节点的整数双向链表。既然已经了解了双向链表的基本概念，如果你之前实现过单向链表，那么实现双向链表应该不会太难。

一个显而易见的问题是，我们为什么需要创建一个双向链表？双向链表有哪些优势或重要应用场景？首先，其优势在于，现在我们如果有一个指向任意节点的指针，就可以进行正向和反向的查找。仅凭一个指针，我们就能访问当前节点、下一个节点以及前一个节点。我这里展示了一个名为temp的指针。如果temp是指向某个节点的指针，那么temp.next就是指向下一个节点的指针，即下一个节点的地址；而temp.previous，更准确地说，temp->previous，实际上是(\*temp).prev的语法糖。

所以，这个 temp->prev 指的是前一个节点，或者简单来说就是指向前面节点的指针。在这个例子中，temp 当前存储的值是 600，temp.next 是 800，而 temp.prev 是 400。在单链表中，你无法仅通过一个指针来查看前一个节点，必须额外使用一个指针来跟踪前一个节点。

在很多情况下，能够查看前一个节点会让我们的工作变得更轻松。甚至某些操作（比如删除）的实现也会变得简单许多。在单链表中要删除一个节点，你需要两个指针：一个指向要删除的节点，另一个指向前一个节点。但在双向链表中，我们只需使用一个指针（指向待删除节点的指针）就能完成这一操作。总而言之，这种能在链表中进行反向查找的能力非常实用，它让我们能够双向遍历链表。

双向链表的缺点是需要为指向前一个节点的指针使用额外的内存。以一个整数链表为例，假设在典型的架构中整数占四个字节，指针变量也占四个字节，那么在单链表中，每个节点将占用八个字节，其中四个字节用于数据，四个字节用于指向下一个节点的链接。而在双向链表中，每个节点将占用十二个字节，其中四个字节用于数据，八个字节用于链接（指向前后节点的指针）。

对于整数链表，链接所占用的空间将是数据的两倍。使用双向链表时，在插入或删除时需要更小心地重置链接，比单向链表需要重置更多的链接。因此我们更容易出错，我们将在下一课的 C 程序中实现双向链表，编写遍历、插入和删除等基本操作。这节课就到这里。感谢观看。

## 十三、C/C++ 实现双向链表

在上一课中，我们了解了什么是双向链表。在本节课中，我们将用 C 语言实现双向链表，编写插入、遍历和删除等简单操作。正如上节课所讲，双向链表的每个节点包含三个字段——我已在此绘制了双向链表的逻辑示意图：一个字段存储数据，一个存储下一节点地址，另一个存储前一节点地址。在 C 或 C++ 程序中，整型链表的节点定义将采用这种形式。

在逻辑表示中，我会在每个节点中填入一些数据。假设这些节点的地址分别是 400、600 和 800。我也会填入 next 和 prev 字段。此外，我们还需要一个指向头节点的指针变量。通常我们将这个指针变量命名为 head。在我的实现中，我会编写这些函数：一个用于在链表开头（即头部）插入节点的函数 `InsertAtHead(x)`，该函数将接收一个整数作为参数；另一个用于在链表尾部插入节点的函数 `InsertAtTail(x)`；以及一个在从头到尾遍历链表时打印链表元素的函数 `Print()`。

我将再写一个函数，在从尾到头遍历链表时逆序打印元素 `ReversePrint()`。这个反向打印函数会验证每个节点的反向链接是否正确创建。现在让我们在一个实际的 C 程序中编写这些函数。在我的 C 程序中，我将节点定义为一个包含三个字段的结构体。第一个字段是整型，用于存储数据。第二个字段是指向节点的指针类型，用于存储下一个节点的引用。第三个字段是一个指向节点的指针，用于存储前一个节点的引用。我定义了一个名为head的变量，它同样是一个指向节点的指针。并且我在全局作用域中定义了这个变量，因此head是一个全局变量。当我们在函数内部定义一个变量时，它被称为局部变量。局部变量的生命周期就是函数调用的生命周期。

```c++
void InsertAtPosition_n(int data, int n){  
    Node* temp = head;  
    if(n == 1){  
        InsertAtHead(data);  
        return;  
    }  
    for(int i=0; i<n-2; i++){  
        temp = temp->next;  
    }  
    if(temp->next == nullptr){  
        InsertAtTail(data);  
        return;  
    }  
    Node* temp1 = new Node();  
    temp1->data = data;  
    temp1->next = temp->next;  
    temp->next->prev = temp1;  
    temp->next = temp1;  
    temp1->prev = temp;  
}
```

它是在函数调用执行期间创建的，并在函数调用执行完成后从内存中清除。但全局变量在应用程序的整个生命周期中都存在于内存中，直到程序执行完毕。全局变量可以在所有函数中随处访问，而局部变量除非通过指针访问，否则无法随处访问。在我们之前的实现中，我们大多将 head 声明为全局变量。好的，现在让我们来编写这些函数。我想写的第一个函数是在头部插入，这个函数将接收一个整数作为参数。

我们在这里首先要做的是创建一个节点，我们可以像这样声明一个节点。就像声明任何其他变量一样，我们可以说 struct node。然后我们可以给它一个标识符或名称。现在，在我创建的节点中，我可以填写所有字段。但这里的问题是，当我像这样创建一个节点时，我是将其作为局部变量创建的。当函数调用结束时，它将被从内存中清除。

局部变量存在于我们称之为应用程序内存的栈区，我们无法控制其生命周期。当函数调用结束时，它就会被从内存中清除。这不是我们想要的，我们的要求是除非我们显式移除节点，否则它应该一直保留在内存中。这就是为什么我们要在动态内存（也就是我们所说的堆内存区）中创建一个节点。堆中的内容除非我们显式释放，否则不会被清除。在 C 语言中，我们使用 malloc 函数在堆中创建节点，而在 C++ 中则使用 new 操作符。

malloc 函数的作用仅仅是在堆中预留一些内存。这段内存可以用来存储任何内容，任何变量，任何对象。访问这段内存总是通过指针变量来实现。我们在之前的课程中已经多次讨论过这个概念。但我还是要反复强调，因为它确实非常重要。通过这条语句，我在动态内存（即堆）中创建了一个节点，可以通过一个指向该节点的指针变量来引用它。我将这个变量命名为temp。现在，我可以使用这个指针变量来填充节点的各个字段的值。我需要用星号运算符来解引用这个指针变量。

然后我就可以访问各种字段，比如data、prev或next。对于这个星号temp点data，还有一种替代语法，我们可以简单地写成temp箭头data。同样地，我也可以访问其他字段。所以，要访问pref字段，我可以说temp箭头prev。让我们把这个设置为null。同时，我们也将next字段设置为null。

如果你想了解或复习内存中栈和堆的概念，可以查看本视频描述中提供的动态内存分配课程链接。好的，在我的insert at head函数中，我在内存的堆区创建了一个节点。我正在使用名为temp的指针变量来引用该节点。“Temp”这个名字不太有意义。我们可以使用像“new node”或“new node pointer”这样的名称。我想把这部分节点创建的逻辑，也就是这几行创建节点的代码，单独提取到一个函数中。

我在这里写了一个名为get_new_node的函数，它会接收一个整数作为参数，创建一个节点，将数据字段填充为x，并将前驱和后继指针都设为null。这个函数会返回一个指向节点的指针。因此，我将从这里返回new_node。

我之所以单独写一个函数，是因为通过将创建节点的功能独立封装，可以避免代码重复——因为在稍后编写的"头部插入"函数和"尾部插入"函数中都需要创建新节点。现在在头部插入函数里，我只需调用这个get_new_node函数并传入x值即可。该函数会返回指向新创建节点的指针，我将用这个名为temp的节点指针变量来接收返回值（当然这个变量也可以命名为new_node）。

在头部插入的新节点与获取新节点中的新节点不同。这些都是局部变量。这个新节点是头部插入的局部变量，而这个新节点是获取新节点的局部变量。现在，在头部插入时会有两种情况。链表可能为空。因此，头节点将等于null。

在这种情况下，我们可以简单地将头指针设置为新节点的地址并返回或退出。如果我也从逻辑视图展示所有内容，事情就会变得清晰。目前，我的链表在这里是空的。在我展示的这个逻辑视图中，假设我调用了一个在头部插入数字2的操作。获取新节点的函数会给我一个新节点。假设一个新节点在地址400处被创建。通过这条语句"头指针等于新节点"，我们正在将新节点变量中存储的地址设置到头指针中。

NULL 不过是地址 0。一旦这个在头部插入的函数执行完毕，这个变量 new node 就会从内存中清除。但节点本身不会被清除。如果我们像这样创建节点，struct node new node，在这个声明中，new node 不是指向节点的指针，它本身就是节点，而且我们没有使用 struct node*。

所以，如果我们像这样创建节点，节点也会被清除。好了，回到这个函数，让我们写下剩下的逻辑。当链表不为空时插入一个节点，我会这样做。现在，我正在调用一个在头部插入数字4的操作。一旦新节点创建完成，我首先会将现有头节点的previous字段设置为这个新节点的地址。这样，我就在建立这个链接。接着，我会将新节点的next字段设置为当前头节点的地址。

现在我可以断开这个链接并建立这个链接。因此，我将把head设置为新节点的地址。最终的情况会是这样的。让我们也快速看看应用程序内存的各个部分实际上是如何移动的。分配给程序的内存通常分为这四个段。我们在之前的课程中已经多次看到这个图表。代码或文本段存储所有待执行的指令。有一个段用于存储全局变量。还有一个我们称为栈的部分，它就像便签或白板一样用于函数调用，执行栈则是所有局部变量的存放位置。

不仅仅是局部变量，所有关于函数调用执行堆的信息也是我们所说的动态内存。我这里将堆栈、堆和全局区分别展示出来。在我们的程序中，我们将head声明为一个全局变量。对于一个空链表，我们最初会将head设置为null或零。假设我们会在主函数中完成这一操作。现在，当在此阶段调用在头部插入时，假设我传入数字2作为参数进行调用。

假设我们从主函数调用在头部插入的操作。程序开始执行时，首先调用的是主函数。每当调用一个函数时，都会从栈中分配一部分内存用于执行该函数。这部分内存称为该函数的栈帧。该函数的所有局部变量都存在于其栈帧中。当函数调用执行完毕时，栈帧会被回收。

当主函数调用在头部插入的函数时，主函数的执行将在调用行暂停，系统会为执行头部插入操作分配一个栈帧。为了节省空间，我将"insert at head"简写为"I a H"。头部插入函数的所有参数和局部变量都将存储在这个栈帧中。

我们正在创建一个名为new_node的变量，它是一个指向节点的指针，作为局部变量。然后我们调用get_new_node函数，此时insert_at_head函数的执行将暂停，转而执行get_new_node函数。我们可以这样编写get_new_node函数：这里我在栈上创建一个节点。x是一个局部变量，get_new_node也是。接着我创建一个节点，将数据填充为x的值（即2），并将previous和next字段设置为null或0。

然后，因为我需要返回一个指向节点的指针，所以我在这里使用了取地址运算符。使用取地址运算符可以获取变量的指针。假设我们在get_new_node函数的栈帧中创建的这个新节点的地址是50。通过这个返回，当get_new_node函数执行完毕时，insert_at_head函数中的这个新节点的值将是50。请注意，根据这段代码，get_new_node函数中的这个新节点是struct node类型，而insert_at_head函数中的这个新节点是指向struct node的指针类型。所以它们是不同的类型。我们可以返回这个地址50，这是没问题的。

但是一旦函数执行完毕，get new node的栈帧就会被回收。所以现在即使你有地址50，那里也没有节点。我们无法控制栈上内存的分配和释放。这是自动发生的。这就是为什么我们使用堆上的内存。如果我使用这段代码来创建新节点，那么我所做的是将这个变量new node声明为struct node指针，而不是struct node。

那是指向节点的指针。我使用malloc在堆区创建实际的节点。假设这个节点的地址是400。对于堆区中的某块内存，我们无法直接命名。访问堆区中内容的唯一方式是通过指针。如果我们丢失了这个指针，就会丢失这个节点。好的，现在我们正在使用这个指针new node，它是get new node函数的局部变量。我们正在访问这个节点，填充数据，填写地址字段。现在我们要返回这个地址400。

当获取新节点的操作即将完成时，我将返回的地址400存储在这个局部变量new_node中。现在我们返回到insert_at_head函数，此时head的值仍为null。于是我们执行head = new_node的赋值操作——请注意head是全局变量，它在整个应用程序生命周期中都不会被清除。随着insert_at_head函数的栈帧即将被清除，new_node这个局部变量也将随之释放。

这就是我们最终得到的结果，当我们再次调用头部插入操作时。执行函数时又会分配新的栈帧，并创建相应的链接。因此我们的链表会相应地进行修改。希望这些内容能让大家理解。当再次调用头部插入操作并完成后，控制权返回到主程序时，链表可能呈现这样的结构。假设我得到了一个存储在600地址的节点。右侧单元格用于下一个节点，右侧单元格存储下一个节点的地址，左侧单元格存储上一个节点的地址。这就是我们将要实现的。现在让我们继续编写其余的函数。

打印函数将与单链表的打印功能相同。我们将创建一个临时节点指针，初始时将其设置为头节点，然后使用语句temp等于temp.next来移动到下一个节点，并持续打印数据。在反向打印时，我们将首先使用next指针到达链表的末端节点，然后通过语句temp等于temp->prev来向后遍历链表。

因此，我们将使用前一个指针，并在向后遍历时打印数据。好的，现在让我们测试一下到目前为止我们编写的所有这些函数。在主函数中，我将头指针设置为null，表示链表最初为空，现在我正在编写几个插入语句。我正在调用几次在头部插入的函数，每次调用后我都会打印链表，包括正向和反向遍历。让我们运行这段代码看看输出结果。这就是我得到的结果，我认为这符合预期。

我正在调用几次插入头节点的函数，每次调用后，我都会以正向和反向两种方式打印链表。让我们运行这段代码看看输出结果。这就是我得到的结果，我认为这符合预期。我之前说过要写一个在尾部插入的函数。如果你已经理解了前面的内容，写这个在尾部插入的函数对你来说应该不会太难。我把这个留作你的练习。今天就到这里。如果你想获取这段源代码，可以查看视频描述中的链接。在接下来的课程中，我们将讨论循环链表，并探讨链表上更多有趣的问题。

## 十四、栈

在本节课中，我们将向大家介绍栈数据结构。我们知道，数据结构是计算机中存储和组织数据的方式。在本系列课程中，我们已经讨论了一些数据结构，比如数组和链表。现在，在这节课中，我们将要讨论栈，并且我们将栈作为一种抽象数据类型（ADT）来讲解。

当我们把数据结构视为抽象数据类型时，我们只讨论该数据结构可用的特性或操作，而不涉及具体的实现细节。因此，我们基本上只将数据结构定义为一个数学或逻辑模型。

我们将在后续课程中深入讲解栈的实现。本节课我们仅讨论栈的抽象数据类型（ADT），因此只会关注栈的逻辑结构。计算机科学中的栈数据结构与现实世界中物品堆叠的方式并无本质区别。以下是现实生活中的堆叠实例：第一张图展示的是一叠餐盘的堆叠方式。

第二幅图展示的是一个名为“河内塔”的数学谜题。该游戏包含三根柱子（或称为三根钉），以及多个圆盘。游戏的目标是将一叠圆盘从一根柱子移动到另一根柱子，但有一个限制条件：不能将较大的圆盘放在较小的圆盘上面。第三幅图则展示了一堆网球。堆栈本质上是一种具有特定性质的集合：堆栈中的元素必须从同一端（我们称之为栈顶）插入或移除。

事实上，这不仅仅是一个属性，而是一种约束或限制。只有栈顶是可访问的，任何项目都必须从栈顶插入或移除。栈也被称为后进先出的集合。栈中最晚添加的物品必须最先取出。在第一个例子中，你总是从栈顶拿起餐盘，如果需要将盘子放回栈中，你也总是会将其放回栈顶。你可能会争辩说，我可以不实际移除顶部的盘子，而从中间抽出一个盘子。

所以“我必须总是从顶部取出盘子”这一限制并未被严格执行。为了讨论方便，这样是可以的。你可以这么说。在另外两个例子中，当我们在一个柱子上有圆盘，或者在这个只能从一侧打开的盒子里有网球时，你不可能从中间取出一个物品。任何插入或移除都必须从顶部进行。你无法从中间滑出一个物品——你可以取出一个物品，但为此你必须移除该物品上方的所有物品。

现在让我们正式将栈定义为一种抽象数据类型。栈是一种列表或集合，其限制在于插入和删除只能在一端进行。我们称这一端为栈顶。现在让我们定义栈ADT可用的接口或操作。栈有两个基本操作。插入操作称为压栈（push）。

压入操作可以将某个项目x插入或压入堆栈。另一个操作，第二个操作称为弹出。弹出是从堆栈中移除最近的项目，从堆栈中移除最近的元素。压入（push）和弹出（pop）是最基本的操作，此外还可以有少数其他操作。通常有一个称为top的操作，它只返回栈顶的元素；还可以有一个操作用于检查栈是否为空。如果栈为空，这个操作将返回true，否则返回false。

因此，push操作是将一个元素压入栈顶，而pop操作则是从栈顶移除一个元素。我们一次只能push或pop一个元素。这里列出的所有操作都可以在常数时间内完成，换句话说，时间复杂度为O(1)。请记住，最后被压入或插入栈中的元素会最先被弹出或移除。因此，栈被称为后进先出结构。最后进入的元素最先出来。后进先出简称为LIFO。从逻辑上讲，栈可以用一个三边的图形来表示，就像一个只有一侧开口的容器。这是一个空栈的表示。我们把这个栈命名为S。假设这个图形代表一个整数栈。

现在栈是空的。我将执行压入和弹出操作来向栈中插入和移除整数。我会先在这里写下操作，然后向你展示逻辑表示中会发生什么。我们先执行一个压入操作。我想把数字2压入栈中。现在栈是空的，所以我们无法弹出任何元素。

压入后，栈看起来会是这样。栈中只有一个整数，所以它当然位于栈顶。让我们再压入一个整数。这次我想压入数字10。现在假设我们要执行一个弹出操作。当前位于栈顶的整数是10。执行弹出操作后，它将被从栈中移除。让我们再执行几次压入操作。

我刚把7和5压入栈中。此时如果调用top操作，会返回数字5。isEmpty会返回false。此时执行pop操作会从栈中移除5。正如你所见，最后进入的整数会最先出来。这就是为什么我们称栈为后进先出（LIFO）的数据结构。我们可以不断执行pop操作直到栈为空。再执行一次pop操作栈就会变空。以上就是栈数据结构的基本概念。现在一个显而易见的问题是：在实际场景中栈能帮助我们解决什么问题？

让我们列举一些栈的应用。栈数据结构用于程序中函数调用的执行。我们在动态内存分配和链表的课程中已经多次讨论过这一点。我们也可以说栈用于递归，因为递归也是一系列函数调用。只不过所有这些调用都是针对同一个函数。想了解更多这方面的应用，你可以查看本视频描述中提供的链接，那里有我课程学校关于动态内存分配的课程内容。

栈的另一个应用是可以用来实现编辑器中的撤销操作。我们可以在任何文本编辑器或图像编辑器中执行撤销操作。现在我正在按Ctrl+Z，正如你所看到的，我写的一些文本正在被清除。你可以使用栈来实现这一点。栈在许多重要算法中都有应用，例如编译器会使用栈数据结构来验证源代码中的括号是否匹配。对于源代码中的每一个左花括号或左括号，必须在适当的位置有一个右括号与之对应。

如果源代码中的括号没有正确放置，如果它们不对称，编译器应该抛出错误。而这一检查可以使用栈来完成。我们将在接下来的课程中详细讨论其中一些问题。作为入门介绍，这些内容已经足够。在下一课中，我们将讨论栈的实现。本节课就到这里。感谢观看。


In our previous lesson, we introduced you to stack data structure. We talked about stack as abstract data type or ADT.

As we know when we define a data structure as abstract data type, we define it as a mathematical or logical model. We define only the features or operations available with the data structure. And do not bother about implementation.

Now in this lesson, we will see how we can implement stack data structure. We will first discuss possible implementations of stack and then we'll go ahead and write some code. Okay, so let's get started.

As we had seen, a stack is a list or collection with this restriction with this constraint, that insertion and deletion that we call push and pop operations in a stack must be performed one element at a time, and only from one end that we call the top of stack. So if you see, if we can add only this one extra property, only this one extra constraint to any implementation of a list, that insertion and deletion must be performed only from one end, then we can get a stack. There are two popular ways of creating lists.

We have talked about them a lot in our previous lessons, we can use any of them to create a stack, we can implement stacks using a arrays and be linked lists. Both these implementations are pretty intuitive. Let's first discuss array based implementation.

Let's say I want to create a stack of integers. So what I can do is I can first create an array of integers, I'm creating an array of 10 integers here, I'm naming this array A. Now I'm going to use this array to store a stack. What I'm going to say is that at any point, some part of this array, starting index zero till an index marked as top will be my stack, we can create a variable named top to store the index of top of stack.

For an empty stack, top is set as minus one. Right now in this figure top is pointing to an imaginary minus one index in the array. And insertion or push operation will be something like this, I will write a function named push that will take an integer x as argument.

In push function, we will first increment top and then we can fill in integer x at top index. Here we are assuming that A and top will be accessible to push function even when they're not passed as arguments. In C, we can declare them as global variables or in an object oriented implementation.

All these entities can be members of a class. I'm only writing pseudocode to explain the implementation logic. Okay, so for this example array that I'm showing here right now, top is set as minus one, so my stack is empty.Let's insert something onto the stack, I will have to make call to push function. Let's say I want to insert number two onto the stack. In a call to push first top will be incremented.And then the integer passed as argument will be written at top index. So two will be written at index zero. Let's push one more number.

Let's say I want to push number 10. This time, once again top will be incremented 10 will now go at index one. With each push the stack will expand towards higher indices in the array.

To pop an element from the stack, I'm writing a function here for pop operation. All I need to do is decrement top by one with a call to pop. Let's say I'm making a call to pop function here, top will simply be decremented, whatever cells are in yellow in this figure are part of my stack, we do not need to reset this value before popping.

If a cell is not part of stack anymore, we do not care what garbage lies there. Next time when we will push we will modify it anyway. So let's say after this pop operation, I want to perform a push, I want to insert number seven onto the stack.

So top once again will be incremented and value at index two will be overwritten, the new value will be seven. These two functions push and pop that I have written here will take constant time, we have simple operations and these two functions and execution time will not depend upon size of stack. While defining stack ADT, we had said that all the operations must take constant time or in other words, the time complexity should be big O of one.

In our implementation here, both push and pop operations are big O of one. One important thing here, we can push onto the stack only till array is not exhausted only till some space is left in the array, we can have a situation where stack would consume the whole array. So top will be equal to highest index in the array.

A further push will not be possible because it will result in an overflow. This is one limitation with array based implementation. To avoid an overflow, we can always create a large enough array.

For that we will have to be reasonably sure that stack will not grow beyond a certain limit. In most practical cases, large enough array works. But irrespective of that, we must handle overflow in our implementation.There are a couple of things that we can do in case of an overflow. Push function can check whether array is exhausted or not. And it can throw an error in case of an overflow.So push operation will not succeed. This will not be a really good behavior. We can do another thing, we can use the concept of dynamic array, we have talked about dynamic array in initial lessons in this series.

What we can do is in case of an overflow, we can create a new larger array, we can copy the content of stack from older filled up array into new array. If possible, we can delete the smaller array, the cost of copy will be big O of n, or in simple words, time taken to copy elements from smaller array to larger array will be proportional to number of elements in stack or the size of the smaller array. Because anyway, stack will occupy the whole array.

There must be some strategy to decide the size of larger array. Optimal strategy is that we should create an array twice the size of smaller array. There can be two scenarios in a push operation.

In a normal push, we will take constant time in case of an overflow, we will first create a larger array twice the size of smaller array, copy all elements in time proportional to size of the smaller array. And then we will take constant time to insert the new element. The time complexity of push with this strategy will be big O of one in best case, and big O of n in worst case in case of an overflow time complexity will be big O of n. But we will still be big O of one in average case, if we will calculate the time taken for n pushes, then it will be proportional to n. Remember n is the number of elements in stack.Big O of n is basically saying that time taken will be very close to some constant times n. In simple words, time taken will be proportional to n. If we are taking c into n time for n pushes to find out average, we will divide by n average time taken for each push will be a constant. Hence, big O of one in average case. I will not go into all the mathematics of why it's big O of n, I will just show you how to calculate the time taken for each push.

ds-5