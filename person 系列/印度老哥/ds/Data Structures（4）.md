
## 十六、链表实现栈

在之前的课程中，我们学习了如何用数组实现栈。现在这节课，我们将学习如何用链表实现栈。在本课程中，我假设您已经了解了栈和链表栈。正如我们目前讨论所知，栈被称为后进先出（LIFO）的数据结构。最后进入栈的元素会最先出来。它是一种有特定限制的列表，插入和删除操作只能在一端进行，我们称之为栈顶。在栈中，插入操作称为压栈（push），删除操作称为弹栈（pop）。

要实现一个栈，我们只需在任何列表实现中强制规定插入和删除操作只能在一端进行，我们可以称该端为栈顶。在链表中强制执行这种行为非常简单。这里我画了一个整数链表。这是链表的逻辑表示。链表是由我们称为节点的实体组成的集合。每个节点包含两个字段，一个用于存储数据，另一个用于存储下一个节点的地址。

这是链表的逻辑表示。链表是由我们称为节点的实体组成的集合。每个节点包含两个字段：一个用于存储数据，另一个用于存储下一个节点的地址。假设这些节点的地址分别为100、200和400。因此，我也会填写地址部分。链表的标识是第一个节点的地址，我们也称之为头节点。

变量存储着头节点的地址，我们通常将这个变量命名为head。与数组不同，链表的大小不是固定的，链表中的元素也不存储在一个连续的内存块中。通过之前的课程，我们已经了解了如何创建链表，以及如何在链表中插入和删除元素。我只是简单回顾一下。在链表中插入一个元素时，我们首先要创建一个新节点，本质上就是占用一部分内存来存储我们的数据。在这个例子中，假设我的新节点获取的地址是350。

我们可以将链表的数据部分设置为我想在列表中添加的任何值。然后，我需要修改一些现有节点的地址字段，以在实际列表中链接这个节点。对于栈来说，我们希望插入和删除始终发生在同一端，这时可以使用链表来实现栈。

如果我们总是在同一端插入和删除节点，我们有两种选择：可以从链表的末端（也称为尾部）或链表的起始端（称为头部）进行插入或删除。如果你还记得我们之前的课程，在链表的末端插入节点并不是一个恒定时间的操作。如果我们讨论链表末端插入和删除的时间复杂度，它是O(n)。而在栈的定义中，我们要求压入（push）和弹出（pop）操作应该花费恒定时间，或者说它们的时间复杂度应该是O(1)。

但如果我们在链表末尾进行插入和删除操作，时间复杂度会是O(n)。要在链表末尾插入一个新节点，我们需要遍历到最后一个节点，并将该节点的地址部分设置为指向新节点。为了遍历链表并到达最后一个节点，我们应该从头节点或第一个节点开始。从第一个节点，我们可以获取第二个节点的地址。

所以我们来到第二个节点，从第二个节点获取第三个节点的地址。这就像玩寻宝游戏一样，你找到第一个人，询问第二个人的地址；然后你找到第二个人，询问第三个人的地址，以此类推。现在，一旦我到达了这里的最后一个节点，在我的示例中，我可以设置它的地址部分，使其指向新创建的节点。总的来说，这个操作所需的时间与链表中的元素数量成正比。要从末尾删除一个节点，我们再次需要遍历整个链表，我们需要到达倒数第二个节点，断开这个链接，我们将地址字段设置为零或空。

然后，我们可以简单地从计算机内存中擦除从列表中删除的最后一个节点。再次强调，遍历的成本将是n的大O。因此，在末端或尾部插入和删除对我们来说不是一个选择，因为我们无法在恒定时间内进行push和pop操作。如果我们选择从末端插入和删除，那么从开头插入或删除的成本是大O(1)，即在开头插入节点或从开头删除节点将花费恒定时间。

要在开头插入一个节点，我们必须创建一个新节点。在这个例子中，我又一次创建了一个新节点，假设这个新节点的地址是350。我会在这个节点的第一个字段中插入一些数据。好了，要在开头插入这个节点，我们只需要建立两个链接。首先，我们需要建立这个链接。因此，我们将这里的地址设置为当前头节点的地址。

然后我们可以断开这个链接，并通过在这个名为head的变量中设置其地址，使这个节点成为新的头节点。要删除一个节点。在这个例子中，我们首先需要断开这个链接，然后建立这个链接，这意味着要重置这个变量head中的地址。

然后我们可以释放分配给这个特定节点的内存，从开头删除这个特定节点再次是一个常数时间操作。所以这就是如果我们从开头插入并从开头删除，那么我们所有的条件都得到了满足。因此，用链表实现堆栈是非常直接的。

我们只需要在开头插入一个节点，并从开头删除一个节点。因此，链表的头部实际上就是栈顶，我更愿意在这里将这个变量命名为top。我会快速用C语言写一个基本的实现，我将节点定义为C语言中的一个结构体，我想创建一个整数栈。

因此，节点中的第一个字段是一个整数。另一个字段是指向节点的指针，它将存储下一个节点的地址。我们在之前关于链表的所有课程中都见过这种节点的定义。接下来我要做的是声明一个名为top的变量，它是指向节点的指针。最初，我将其中的地址设置为null。在这里，我使用名为top的变量而不是head。

当 top 为空时，我们的栈是空的。通过将 top 初始化为 null，我表示初始时我的栈是空的。现在让我们编写 push 和 pop 函数。这是我的 push 函数，push 接受一个整数 x 作为参数，该参数必须被插入到栈中。我们在 push 函数中做的第一件事是使用 malloc 创建一个节点。假设在这个例子中，在我展示的这个逻辑表示中，我正在执行 push 操作。

因此，我正在调用 push 函数，并将数字 2 作为参数传递给它。于是，在内存中创建了一个节点，这个节点位于我们所说的动态内存或堆中，假设这个节点的地址是 100。这个变量本质上是一个指向该节点的指针，temp 是一个指向该节点的指针。在下一行，我们正在设置这个节点的数据字段。为此，我们对 temp 进行解引用。然后，我们将这个新创建的节点的链接部分设置为现有的顶部。

所以我们正在构建这个链接。然后我们说top等于temp。所以我们正在构建这个链接。这是在链表开头进行简单插入。我们在这个系列中有一个完整的视频，介绍如何在链表开头插入节点。让我们再推一次。假设这次我想把数字5压入堆栈。再次，将创建一个节点，我们将设置数据。然后我们将首先把这个节点指向现有的顶部。

然后让这个指针变量指向这个新的顶部，假设这个新顶部的地址是250。因此，这个变量top中的地址将被设置为250。在第二次压栈操作后，我的堆栈将呈现这样的状态。这里的top是一个全局变量，所以我们不需要将其作为参数传递给函数，它对所有函数都是可访问的。在面向对象的实现中，它可以是一个私有字段。

我们可以在构造函数中将其设为 null。好的，现在让我们看看 push，抱歉，pop 函数会是什么样子。这是我的 pop 函数。假设在这个例子中，我调用了 pop 函数。如果栈已经是空的，我们可以通过检查 top 是否为 null 来判断栈是否为空。如果 top 是 null，栈就是空的。

在这种情况下，我们可以抛出一些错误并返回，因为在这个例子中栈不为空。栈中有两个整数。我们首先做的是创建一个指向节点的指针temp，并将其指向栈顶节点。然后我们断开这个链接，将栈顶的地址设置为下一个节点的地址。现在，我们使用这个指针变量temp来释放从列表中移除的节点所占用的内存。一旦我退出pop函数，这就是我的栈。

这就是我们实现的核心部分，我鼓励你自己完成剩下的内容。你可以编写诸如top和is empty等操作的代码。链表的栈实现有一些优势。其中一个优点是，与基于数组的实现不同，除非耗尽机器本身的内存，否则我们无需担心溢出问题。每个节点会使用一些额外的内存来存储引用或地址。但关键在于我们按需分配内存并在不需要时释放，这使得压入（push）和弹出（pop）操作更加优雅。所以这是基于链表的栈实现。在接下来的课程中，我们将使用栈来解决一些问题。这节课就到这里。感谢观看。

## 十七、栈反转字符串

在上一节课中，我们学习了如何实现一个栈。我们看到了两种常见的栈实现方式：一种是使用数组，另一种是使用链表。战士不应仅拥有武器，更需懂得何时及如何使用。作为程序员，我们必须清楚在哪些场景下可以运用特定的数据结构。本节课我将探讨栈的一个简单应用案例。栈可以用来反转列表或集合，或者简单地以相反的顺序遍历列表或集合。

我将讨论两个问题：字符串反转和链表反转。我将使用栈来解决这两个问题。让我们先讨论字符串反转。我这里有一个字符数组形式的字符串，这个字符串是 "Hello"，字符串就是字符的序列。在 C 语言中，这是一个 C 风格字符串，必须以空字符结尾。所以最后一个字符是空字符。反转意味着数组中的字符应该重新排列，就像我右边展示的那样，空字符仅用于标记字符串的结束，它不是字符串的一部分。好的，我们有几种高效的方法可以反转字符串。

我们先讨论如何用栈来解决这个问题，然后看看它的效率如何。我们可以创建一个字符栈。这里展示的是栈的逻辑表示。这是一个字符栈，目前是空的。接下来，我们可以从左到右遍历字符串中的字符，并将它们逐个压入栈中。首先，H 被压入栈中，接着下一个字符是 E，然后是 L，之后又有一个 L。最后一个字符是 O。一旦字符串中的所有字符都进入栈中，我们就可以再次从第零个索引开始。现在我们需要将栈顶的字符写入这个索引位置，可以通过调用 top 操作来获取栈顶字符。然后我们可以执行 pop 操作。接着我们移动到下一个索引，填入栈顶的字符并执行 pop 操作。我们可以继续这个过程，直到栈为空。因此，字符数组中的所有位置都将被覆盖。最终，我们在这里实现了字符串的反转。

在栈中，后进先出。所以如果我们把一堆元素压入栈中，一旦所有元素都被压入后，如果我们开始弹出，我们将以相反的顺序得到这些元素，第一个被压入栈的元素将最后出来。让我们快速为这个逻辑编写代码。我将在这里用 C++ 编写，其他语言中的情况会非常相似，所以这并不重要。在我的代码中，我将创建一个字符数组来存储字符串。然后我会让用户输入一个字符串。一旦我输入了字符串，我将调用一个名为 reverse 的函数，传递给它这个数组以及通过调用字符串长度函数获得的字符串长度。最后，我正在打印反转后的字符串。

现在我需要编写反转函数。在反转函数中，我想使用一个栈，一个字符栈，我们已经了解了如何实现栈。在 C++ 中，我们可以创建一个名为 stack 的类，该类将包含一个字符数组和一个名为 top 的整型变量，用于标记数组中栈顶的位置。这些变量可以设为私有。我们可以通过这些公共函数来操作栈。在反转函数中，我们可以简单地创建一个栈对象并使用它。这个类可以是基于数组实现的栈，也可以是基于链表实现的栈，这并不重要。在 C++ 和许多其他语言中，语言库也为我们提供了栈的实现。在这个程序中，我不打算自己编写栈的实现，而是打算使用 C++ 中所谓的标准模板库中的栈。我需要使用这个包含语句：#include \<stack\>。现在，我就有了一个可用的栈类。

要创建这个类的对象，我需要写 stack，并在尖括号内指定我们想要的栈的数据类型，然后空格后跟名称或标识符。通过这一条语句，我就创建了一个字符栈。现在让我们来编写核心逻辑。reverse 函数签名中的 n 表示字符串的字符数量。我们知道，在 C 或 C++ 中，数组总是通过指针以引用方式传递，这个后面跟着方括号的 C 只是 *C 的另一种语法形式。编译器会这样解析它。好的，那么接下来我要做的是运行一个从零开始到 n-1 的循环。签名中 reverse 函数的 n 表示字符串的字符数量。

所以我将从左到右遍历字符串。在遍历字符串的过程中，我会通过调用 push 函数将字符压入栈中，我会使用这样的语句。一旦压入完成，我会再做一个循环进行弹出操作，我将运行一个循环，变量 I 从零开始直到 n-1。我将首先将 C[i] 设置为栈顶。然后我会执行一个弹出操作。如果你想了解更多关于 STL 中栈的可用函数，比如它们的签名和如何使用，你可以查看这个视频的描述以获取一些资源。

这就是我在反转函数中需要做的全部内容。让我们运行这段代码看看会发生什么。我需要输入一个字符串。输入 "Hello" 试试。这就是我得到的输出，看起来是正确的。让我们再运行一次。这次我想输入"mycodeschool"。看起来也没问题。所以我们似乎搞定了。这个函数解决了我的字符串反转问题。

现在让我们看看它的效率如何。我们来分析一下它的时间复杂度。我们知道栈上的所有操作都是常数时间。因此，循环内的这些语句也都是常数时间。第一个循环运行了 n 次，第二个循环也运行了 n 次。第一个循环的时间复杂度是 O(n)，第二个循环的时间复杂度也是 O(n)。这两个循环不是嵌套的，而是一个接一个执行的。因此，在这种情况下，整个函数的复杂度也将是 O(n)。时间复杂度是 O(n)。

但我们在这里使用了一些额外的内存用于栈。我们将字符串中的所有字符一个一个地压入栈中。栈中占用的额外空间将与字符串中的字符数量成正比，即与 n 成正比。所以我们可以说，这个函数的空间复杂度也是 O(n)。简单来说，额外占用的空间与 n 成正比。其实存在一些高效的方法可以在不使用额外空间的情况下反转字符串。

最高效的方法可能是仅用两个变量来标记字符串的起始和结束索引。假设我最初使用变量 i 和 j。在这个例子中，i 初始值为零，j 为四。当 i 小于 j 时，我们可以交换这两个位置的字符。一旦我们完成了交换，就可以递增 i 并递减 j。如果 i小于 j，我们可以再次交换。然后再次递增 i 并递减 j。此时 i 不再小于 j，而是等于 j，我们就可以停止交换，完成操作。这个算法的空间复杂度是 O(1)，因为我们在这里使用了恒定的额外内存。这种方法的时间复杂度再次是 O(n)。我们将进行 n/2 次交换。因此，所需时间将与 n 成正比。显然，由于空间复杂度较低，这种方法比我们的栈方法更好。有时当我们知道输入规模很小且时间空间开销无需过多考虑时，我们会选用特定算法——因其实现简单且直观易懂。当我们使用栈来反转字符串时，显然不是这种情况。

但对于另一个问题，即我们之前提到的用栈来反转链表，这种方法提供了一个简洁直观的解决方案。我在这里画了一个整数链表。众所周知，链表是由我们称为节点的实体组成的集合。每个节点包含两个字段：一个用于存储数据，另一个用于存储下一个节点的地址。在这个例子中，我假设这些节点分别位于地址 100、150、250 和 300。链表的标识是头节点的地址。我们通常将这个地址存储在一个名为 head 的变量中。

在数组中，访问任何元素都需要恒定时间。因此，无论是第一个元素还是最后一个元素，访问它们所需的时间都是恒定的。这是因为数组是作为一个连续的内存块存储的。如果我们知道数组的起始地址，假设这个数组的起始地址是 400，而数组中每个字符元素的大小是一个字节。在这个例子中，每个元素占用一个字节，那么我们就可以计算出任何元素的地址。例如，我们知道第五个元素的地址就是 400 加 4，也就是 404。

但在链表中，节点存储在内存中不相连的位置。要访问任何节点，我们必须从头节点开始。因此，我们无法像数组那样简单地设置起始和结束两个指针来访问元素。在本系列中，我们已经看到了两种可以用来反转链表的方法。一种是迭代解决方案，我们使用一些临时变量在遍历链表时不断反转链接。另一种解决方案是使用递归。

迭代解法的时间复杂度为 O(n)，空间复杂度为 O(1)。在递归解法中，我们不会显式创建栈结构。但递归会利用计算机内存中的调用栈来执行函数调用。在这种情况下，我们说我们使用的是隐式堆栈，堆栈并不是显式创建的，但我们仍然在使用隐式堆栈。我稍后会回过头来详细解释这一点。递归解法的时间复杂度再次是 O(n)，但空间复杂度也是 O(n)。这次，空间复杂度同样是 O(n)。现在，让我们看看如何使用显式堆栈来解决这个问题。

我再次在这里绘制了栈的逻辑表示。目前栈是空的。在程序中，这将是一个指向节点类型的指针栈。我现在要做的是使用一个临时节点指针来遍历这个链表，临时变量最初会指向头节点，当我们到达某个特定节点时，我们会将该节点的地址压入栈中。所以首先100会被压入栈。现在我们将会移动到下一个节点。

现在150将被压入堆栈。接下来我们将处理250。最后是位于300的最后一个节点。这里我们在堆栈中显示的是地址。但实际上我们压入的对象是指向节点的指针，或者说节点的引用。如果在C++中节点是这样定义的，我们就需要使用这些语句来遍历链表并压入所有引用。假设 head 被指定给 node，我假设这是一个全局变量，用于存储头节点的地址。我使用了一个临时变量，最初指向 node，我在这个临时变量中存储了头节点的地址。然后我运行一个循环，遍历链表。

在我遍历时，我会将引用压入堆栈。一旦所有引用都被压入堆栈，我们就可以开始弹出它们，随着我们弹出引用，我们将以相反的顺序获得节点的引用，这就像以相反的顺序遍历列表。在以相反的顺序遍历列表时，我们可以构建反向链接。我要做的第一件事是创建一个临时变量，该变量将指向节点，并存储栈顶地址的地址，目前是 300。现在，我将把 head 设置为这个地址。因此，head 现在变成了 300。然后我将执行弹出操作。我正在通过这个例子向你演示，就像我正在编写代码一样，head 和 temp 现在都是 300。

像这样循环，就像我在这里写的那样。当栈不为空时，如果栈为空，这个函数 empty 返回 true。我在 C++ 中使用标准模板库中的栈。所以当栈不为空时，我要将 temp.next 设置为栈顶的地址。基本上，我使用这个指向节点 temp 的指针来解引用并设置这个特定的地址字段。现在 top 是 250。

所以我正在构建这个反向链接。下一条语句是弹出操作。接着下一条语句中，我说 temp 等于 temp.next，这意味着 temp 现在将指向 250 这个节点。栈不为空。所以循环会再次执行，我们现在在这里写入地址，然后应该弹出，再通过 temp=temp.next 这条语句移动到 150。现在我们正在构建这个链接，弹出，然后哎呀，这里应该是 150。接着通过 temp=temp.next，我们来到这里。尽管我们通过在这里设置这个字段构建了这个链接，但这个节点仍然指向这个家伙。因为栈现在是空的，循环结束后我们将退出循环。

退出循环后，我又写了一行代码 temp.next = null。这样我就将反转后链表的最后一个节点的链接部分设置为空。最后，这是我的反转函数，我假设 head 是一个全局变量，它是指向节点的指针。如果你想查看完整的源代码，可以查看这个视频描述中的链接。在这种情况下使用栈会让我们的工作更轻松。反转链表仍然是一个复杂的问题。尝试仅以相反的顺序打印链表的元素。如果使用栈，这会非常简单。本节课就到这里。如果你想知道我所说的隐式栈是什么意思，可以再次查看本视频描述中的一些资源。本节课就到这里，感谢观看。


## 十八、用栈检查括号匹配

在上节课中，我们看到了栈的一个简单应用，了解到栈可以用来反转列表或集合，或者简单地以相反的顺序遍历列表或集合。在本节课中，我们将讨论另一个可以用栈解决的著名问题。这也是一个常见的编程面试题。问题在于，给定一个以字符串形式呈现的表达式，该表达式包含常量、变量、运算符和括号。当我说括号时，我的定义中还包括大括号和方括号。因此，我的表达式或字符串可以包含大小写字母、运算符符号、以及开闭圆括号、开闭大括号或开闭方括号等字符。

让我们在这里写下一些表达式，我先写一个简单的表达式，这里有一个带有一对开闭括号的简单表达式。在这个表达式中，我们有嵌套的括号。现在给定这样的表达式，我们想编写一个程序来告诉我们表达式中的括号是否平衡。我们所说的平衡括号究竟是什么意思呢？平衡括号的真正含义是指，每一个开括号（无论是圆括号、花括号还是方括号）都必须按正确的顺序对应一个相应的闭括号。这里的这两个表达式是平衡的。然而，接下来的这个表达式则是不平衡的。

这里缺少一个闭合的大括号。接下来的表达式也不平衡，因为这里缺少一个开方括号。下一个也不平衡，因为对应于这个开放的大括号，我们没有闭合的大括号，对应于这个闭合的圆括号，我们没有开放的圆括号。如果我们以一个花括号开始，也应该以一个花括号结束，这两个不会互相抵消。检查括号是否平衡是编译器执行的任务之一。当我们编写程序时，经常会遗漏一个开始或结束的花括号，或者一个开始或结束的圆括号，编译器必须检查这种平衡性，如果符号不平衡，它应该给出一个错误。

在这个问题中，括号内的内容无关紧要，我们不需要检查括号内的任何内容是否正确。因此，在字符串中，除了开括号、闭括号、开花括号、闭花括号、开方括号和闭方括号之外的任何字符都可以忽略。这个问题有时可以更好地表述为这样。给定一个仅由圆括号、花括号或方括号的开闭字符组成的字符串，我们需要检查其是否平衡。因此，只有这些字符及其顺序是重要的。在解析实际表达式时，我们可以直接忽略其他字符。我们只关心这些字符及其顺序。

那么，现在如何解决这个问题呢？一个直接想到的方法是，因为每个开括号、开大括号或开方括号都应该有一个对应的闭括号，我们可以统计这三种类型的开闭符号的数量，它们应该相等。因此，开括号的数量应该等于闭括号的数量。开大括号的数量应该等于闭大括号的数量。方括号的数量也应遵循相同的规则。但这还不够好。这里的表达式有一个开括号和一个闭括号，但括号没有配对。下一个表达式是配对的。

但是，这个表达式虽然每种字符的数量与第二个表达式相同，却并不平衡。因此，这种方法行不通。除了数量相等之外，还必须保留其他一些特性。每个开括号必须在右侧找到对应的闭括号。每个闭括号也必须在左侧找到对应的开括号，这在第一个表达式中并不成立。另一个必须保持的特性是，只有当在它之后打开的所有括号都关闭时，一个括号才能关闭。这个圆括号是在这个方括号之后打开的。因此，除非这个圆括号被关闭，否则这个方括号不能被关闭。任何最后打开的东西都应该最先关闭。嗯，实际上不应该是最后打开的先关闭。在这个例子中，这个是最后打开的。但是之前打开的这一个先关闭了。这是可以的。必须保持的性质是，当我们从左到右扫描表达式时，任何关闭符号都应该对应之前未关闭的圆括号。任何关闭符号都应该对应最后一个未关闭的。

让我们从左到右扫描一些表达式，看看它是如何成立的。让我们扫描最后一个。我们将从左到右进行。第一个字符是一个左方括号。第二个是一个左圆括号。让我们用红色标记未闭合的圆括号。好的，现在我们这里有一个闭合符。第三个字符是一个闭合符。这应该是最后一个未闭合的闭合符。所以这应该是这个右括号的闭合符，这个家伙，这个左括号。现在最后一个未闭合的是这个家伙。下一个字符又是一个左括号。现在在这个阶段我们有两个未闭合的括号，而这个是最新未闭合的。下一个是右括号。所以它应该对应最后一个未闭合的左括号。现在，最后一个未闭合的仍然是左方括号的开头。现在我们有一个闭合符号，它应该与这个左方括号配对。

我们可以用这种方法来解决这个问题。我们可以做的是从左到右扫描表达式，在扫描过程中，随时跟踪所有未闭合的括号。具体来说，每当遇到一个开始符号（如左圆括号、左花括号或左方括号），我们就将其添加到列表中。如果遇到一个结束符号，它应该与列表中最后一个元素相匹配。如果出现不一致的情况，比如列表中最后一个开括号与闭括号类型不匹配，或者因为列表为空而根本没有最后一个开括号，我们就可以停止整个过程，并判定括号不平衡。否则，我们可以移除列表中的最后一个开括号，因为已经找到与之对应的闭括号，并继续整个过程。

通过一个例子来演示，事情会变得更加清晰。我将再次通过最后一个例子。我们将从左到右扫描这个表达式，并维护一个列表来跟踪所有尚未闭合的开放括号。我们将记录所有未闭合的括号，即已打开但未关闭的括号。最初这个列表是空的。我们得到的第一个字符是一个方括号的开始。这个字符将被放入列表中，然后我们将移动到下一个字符。下一个字符是一个左括号。所以，它应该再次被添加到列表中。我们应该总是在列表的末尾插入。下一个字符是一个右括号。现在我们必须查看列表中的最后一个左括号符号，如果它是相同类型的，那么我们就找到了它的对应项，应该将其移除。现在我们继续到下一个字符。这又是一个左括号。它应该被添加到列表的末尾。下一个字符是一个右括号。

因此，我们将查看列表中的最后一个元素。它是一个左括号。所以，我们可以将其从列表中移除，现在转到最后一个字符，它是一个右方括号。我们再次需要查看列表中的最后一个元素。此时列表中只有一个元素，即一个左方括号。因此，我们可以再次将其从列表中移除。现在我们已经完成了列表的扫描，列表再次为空。如果一切正常，如果括号是平衡的，我们最终总会得到一个空列表。如果最终列表不为空，则说明存在未找到对应右括号的左括号，表达式不平衡。

这里值得注意的是，我们总是从列表的同一端插入或移除一个元素。在整个过程中，最后进入列表的元素会最先被移除。有一种特殊的列表强制规定了这种插入和移除必须发生在同一端的规则，我们称之为栈。在栈结构中，我们可以在常数时间内从同一端逐个插入和移除元素。*因此，我们的处理方式是：当扫描列表时遇到开符号，就将其压入栈；当遇到闭符号时，检查栈顶的开符号是否与该闭符号属于同种类型。* 如果是同一类型，我们可以弹出。如果不是同一类型，我们可以直接说括号不平衡。我会很快为这个逻辑写出伪代码。

我将编写一个名为 checkBalancedParenthesis 的函数，该函数将以字符串形式接收表达式作为参数。首先，我会将字符串中的字符数量存储在一个变量中，然后创建一个字符栈。接下来，我将使用循环从左到右扫描表达式。在扫描过程中，如果字符是开括号符号（包括圆括号、花括号或方括号的开括号），就将该字符压入栈中。假设这个函数 push 会将一个字符压入栈 s 中。否则，如果在扫描时表达式 i 或第 i 个位置的字符是三种类型中的任何一种的闭合符号，我们可能会遇到两种情况：栈为空，或者栈顶元素与该闭合符号不匹配。如果我们遇到的是圆括号的闭合符号，那么栈顶元素应该是圆括号的开放符号，而不能是花括号的开放符号。在这种情况下，我们可以得出结论：括号是不平衡的。否则，我们可以执行弹出操作。最后，一旦扫描完成，我们可以检查堆栈是否为空。如果为空，则括号是平衡的。如果不为空，则不平衡。这就是我的伪代码。

让我们通过几个例子来验证这个方法是否适用于所有场景和测试用例。首先来看这个表达式。我们在代码中做的第一件事是创建一个字符栈。我在这里画了一个栈的逻辑示意图。好，现在我们来扫描这个字符串。假设我们有一个从零开始的索引，字符串就是一个字符数组。我们开始扫描，进入循环。这是一个右括号。所以，这个 if 语句不会成立。所以，我们将进入 else 条件。现在我们将进入 else 内部，检查这个条件：栈是否为空，或者栈顶是否与此闭合符号配对。栈是空的。如果栈为空，说明没有与这个闭合符号对应的起始符号。因此，我们将直接返回 false，返回意味着退出函数。所以，我们在这里简单地得出结论：括号不平衡并退出。我们现在来看这个。

首先，有一个左方括号。所以，我们会进入第一个if并压入。下一个是一个左圆括号。同样地，它也会被压入。下一个是一个右方括号。因此，这个 else if 的条件将为真，我们将进入这个 else if。现在，栈顶是一个开圆括号。它本应是一个开方括号，这样我们才能配对。所以，这次我们也必须返回 false 并退出。好的，现在让我们来看这个例子。首先，我们会有一个 push 操作。下一个字符也将是一个压栈操作。现在，下一个字符是一个右括号，它与栈顶的左括号配对。因此，我们将执行一个弹栈操作。接着我们处理下一个字符，这次又是一个左括号，所以会有一个压栈操作。再下一个字符是右括号，而栈顶是左括号，它们配对成功，因此会执行弹栈操作。最后一个字符是右花括号。

所以，我们再次检查栈顶是否是一个左大括号。我们是否有一对匹配的括号呢？是的，我们有一对。因此，将执行一次弹出操作。至此，我们的扫描过程将结束，最终栈应该是空的。确实为空。所以，这里的括号是平衡的。尝试用你选择的语言实现这段伪代码，看看它是否适用于所有测试用例。如果你想看我的实现，可以在视频描述中查看链接。在接下来的课程中，我们将看到更多关于栈的问题。这节课就到这里。感谢观看。


## 十九、中缀、前缀和后缀

大家好。这节课我们要讨论计算机科学中一个非常重要且非常有趣的主题，这个主题会用到栈数据结构，那就是算术和逻辑表达式的求值。那么，我们如何写一个表达式呢？我在这里写了一些简单的算术表达式。一个表达式可以包含常量、变量和符号，这些符号可以是运算符或括号，所有这些组成部分都必须按照一组规则、按照一种语法来排列，我们应该能够根据这种语法来解析和求值表达式。

我在这里写的所有这些表达式都有一个共同的结构。我们在两个操作数之间有一个运算符。根据定义，操作数是执行操作的对象或值。在这个表达式2加3中，2和3是操作数，加号是运算符。在下一个表达式中，a和b是操作数，减号是运算符。在第三个表达式中，这个星号表示乘法运算。

所以，这就是运算符。第一个操作数p是一个变量，第二个操作数2是一个常数。这是编写表达式的常见方式，但不是唯一的方式。这种将运算符写在操作数之间的表达式写法称为中缀表示法。操作数并不总是常数或变量，操作数本身也可以是一个表达式。

在我这里写的第四个表达式中，乘法运算符的一个操作数本身就是一个表达式。另一个操作数是一个常数。我们可以有一个更复杂的表达式。在我这里写的第五个表达式中，乘法运算符的两个操作数都是表达式。我们在这个表达式中有三个运算符。对于第一个加运算符p和q，这些变量p和q是操作数。

对于第二个加号运算符，我们有r和s；而对于这个乘法运算符，第一个操作数是这个表达式p加q，第二个操作数是这个表达式r加s。在计算包含多个运算符的表达式时，运算必须按照特定的顺序进行。就像在第四个例子中，我们必须先进行加法运算，然后才能进行乘法运算。在第五个表达式中，我们必须先进行这两个加法运算，然后才能进行乘法运算。

我们会回到评估部分，但正如你所见，所有这些表达式中，运算符都位于操作数之间。这是我们遵循的语法。这里我必须指出一点，在整个课程中，我们将只讨论二元运算符。需要两个操作数的运算符称为二元运算符。从技术上讲，我们可以有只需要一个操作数或可能超过两个操作数的运算符。但我们这里只讨论二元运算符的表达式。

好的，那么现在让我们来看看需要应用哪些规则来评估以这种我们称为中缀表示法的语法编写的表达式。对于只有一个运算符的表达式，没有问题，我们只需应用该运算符即可。对于像这样有多个运算符且没有括号的表达式，我们需要决定运算符应用的顺序。

在这个表达式中，如果我们先进行加法运算，那么这个表达式将简化为10乘以2，最终结果为20。但如果我们先进行乘法运算，那么这个表达式将简化为4加12，最终结果为16。所以基本上，我们可以从两种角度来看待这个表达式。

可以说加法运算符的操作数是4和6。而乘法运算符的操作数是这个表达式4加6，以及这个常数2。或者也可以说乘法运算符的操作数是6和2。而加法运算的操作数是4和这个表达式6乘以2。这里存在一些歧义。但如果你还记得高中数学知识，这个问题可以通过遵循运算符优先级规则来解决。

在代数表达式中，我们遵循以下运算优先级。首先处理括号内的运算。其次是指数运算。我用这个符号表示指数运算符。例如，如果要表示2的3次方，我会这样书写。当出现多重指数运算符时，我们从右向左依次运算。

所以，如果我有一个这样的表达式，那么最右边的指数运算符会首先被应用。这样，表达式会简化为512。如果你先应用左边的运算符，那么结果会是64。在指数运算之后，接下来优先进行乘法和除法运算。如果表达式中有乘法和除法运算符，我们应该从左到右进行计算。在乘法和除法之后，再进行加法和减法运算。

这里同样是从左到右进行运算。如果有一个像这样的表达式，只包含加法和减法运算符，那么我们会先应用最左边的运算符，因为这些运算符的优先级相同。这样计算的结果就是三。如果你先应用加号运算符，这将被计算为一，这是错误的。在第二个表达式中，我在这里写的四加六乘以二，如果我们应用运算符优先级，那么应该先进行乘法运算。如果我们想先进行加法运算，那么我们需要将这个四加六写在括号内。

现在加法会先执行，因为括号的优先级更高。我再举一个复杂表达式的例子并尝试计算它，以便进一步说明问题。那么，我这里有一个表达式。在这个表达式中，我们有四个运算符：一个乘法、一个除法、一个减法和一个加法。乘法和除法具有更高的优先级。在这两个具有相同优先级的乘法和除法之间，我们会先选择左边的那个。所以，我们首先会这样简化这个表达式。现在我们将进行除法运算。现在只剩下减法和加法了。

因此，我们将从左到右进行运算。最终得到的结果就是我在这里写的这个“从右到左和从左到右”的规则，对于具有相同优先级的运算符来说，更准确的术语是运算符结合性。如果多个运算符具有相同的优先级，我们选择从左到右运算时，就称这些运算符是左结合的。而如果从右到左运算，则称这些运算符是右结合的。在计算中缀表达式时，我们首先需要关注运算符的优先级。

然后，为了解决具有相同优先级的运算符之间的冲突，我们需要考虑结合性。总的来说，为了解析和计算中缀表达式，我们需要做这么多事情，括号的使用变得非常重要，因为这是我们控制运算顺序的方式。括号明确了运算应按此顺序执行的意图，同时也提高了表达式的可读性。我已经修改了第三个表达式，现在这里有一些括号。

而我们通常只使用大量括号来书写这样的中缀表达式。尽管中缀表示法是书写表达式最常见的方式，但在不产生歧义的情况下解析和评估中缀表达式并不十分容易。因此，数学家和逻辑学家研究了这个问题，并提出了另外两种书写表达式的方式，它们无需括号，并且可以在不考虑任何运算符优先级或结合性规则的情况下无歧义地解析。

这两种方法分别是后缀表示法和前缀表示法。前缀表示法早在1924年就由一位波兰逻辑学家提出。前缀表示法也被称为波兰表示法。在前缀表示法中，运算符位于操作数之前。中缀表达式"二加三"在前缀表示法中会写作"加 二 三"，加号运算符被放置在两个操作数二和三之前。"p减q"会写作"减 p q"。

再次强调，前缀表示法中的操作数并不总是必须是常量或变量，操作数本身也可以是一个复杂的前缀表达式。这个中缀表达式“a加b乘以c”在前缀形式下会写成这样。稍后我会回到如何将中缀表达式转换为前缀表达式的问题。首先，来看这个前缀形式的第三个表达式。对于这个乘法运算符，两个操作数是变量B和C。这三个元素采用前缀语法。首先是运算符，然后是两个操作数。加法运算符的操作数是变量A和此前缀表达式星号BC。

在中缀表达式中，我们需要使用括号，因为一个操作数可能与两个运算符相关联，就像这个中缀形式的第三个表达式中的B可以与加法和乘法都相关联。为了解决这种冲突，我们需要使用运算符优先级和结合性规则，或者使用括号来明确指定关联关系。但在前缀形式以及我们稍后将讨论的后缀形式中，一个操作数只能与一个运算符相关联。

所以我们没有这种歧义。在解析和评估前缀和后缀表达式时，我们不需要额外的信息，也不需要所有的运算符优先级和结合性规则。我稍后会回来讨论如何评估前缀表示法。我先来定义后缀表示法。后缀表示法也被称为逆波兰表示法。这种语法是在20世纪50年代由一些计算机科学家提出的。在后缀表示法中，运算符位于操作数之后。从编程角度来看，后缀表达式最容易解析，并且在时间和内存评估方面成本最低。这就是它被发明出来的原因。

前缀表达式也可以在类似的时间和内存条件下求值。但解析和计算后缀表达式的算法确实非常直接且直观。这就是为什么在机器计算中更倾向于使用它。我将为之前写好的表达式转换为后缀形式。以第一个表达式为例，"2加3"的后缀形式是"2 3加"。为了区分操作数，我们可以用空格或逗号等分隔符。这就是在编程时通常将前缀或后缀表达式存储在字符串中的方式。第二个表达式的后缀形式是"p q减"。如你所见，在后缀表达式中，我们把运算符放在操作数之后。

后缀表达式中的第三个表达式将是ABC星号然后加号。对于这个乘法运算符，操作数是变量B和C。而对于这个加法运算，操作数是变量A和此前缀表达式BC星号。我们将在后续课程中学习将中缀表达式转换为前缀或后缀表达式的高效算法。目前，我们暂且不必考虑如何在程序中实现这一点。让我们先快速了解一下如何手动完成这个转换。要将一个表达式从中缀形式转换为其他两种形式中的任意一种，我们需要一步步来，就像我们在求值时那样。

我已选择将中缀表达式 A 加 B 转换为 C 的形式，我们应首先转换需要优先计算的部分。因此我们需要按照运算符优先级顺序处理，也可以先为所有隐式括号添加显式括号。这里我们会先将这个 P 转换为 C。所以首先我们会处理乘法运算符的转换，然后再处理加法运算符的转换——我们会把加法运算提到前面来。因此，表达式的转换过程是这样的，我们可以在中间步骤中使用括号。一旦我们完成了所有步骤，就可以擦除括号。现在让我们对后缀表达式做同样的操作，首先处理乘法运算符的转换。

然后在下一步中，我们将进行加法运算。现在我们可以去掉所有的括号了。括号确实为这些表达式或形式增加了可读性。但如果我们不关心人类可读性，那么对机器而言，实际上我们节省了一些用于存储括号信息的内存。中缀表达式无疑是最易为人类理解的，而前缀和后缀表示法则更适合机器处理。这就是中缀、前缀和后缀表示法。下节课我们将讨论前缀和后缀表示法的求值方法。本节课就到这里。感谢观看。



In our previous lesson we saw what prefix and postfix expressions are but we did not discuss how we can evaluate these expressions. In this lesson we will see how we can evaluate prefix and postfix expressions. Algorithms to evaluate prefix and postfix expressions are similar but I'm going to talk about postfix evaluation first because it's easier to understand and implement and then I'll talk about evaluation of prefix.Okay, so let's get started. I have written an expression in infix form here and I first want to convert this to postfix form. As we know in infix form operator is written in between operands and we want to convert to postfix in which operator is written after operands.

We have already seen how we can do this in our previous lesson. We need to go step-by-step just the way we would go in evaluation of infix. We need to go in order of precedence and in each step we need to identify operands of an operator and we need to bring the operator in front of the operands.What we can actually do is we can first resolve operator precedence and put parenthesis at appropriate places. In this expression we will first do this multiplication, this first multiplication then we'll do the second multiplication then we will perform this addition and finally the subtraction. Okay, now we will go one operator at a time.Operands for this multiplication operator are A and B so this A asterisk B will become AB asterisk. Now next we need to look at this multiplication. This will transform to CD asterisk and now we can do the change for this addition.The two operands are these two expressions in postfix. So I'm placing the plus operator after these two expressions. Finally for this last operator the operands are this complex expression and this variable E. So this is how we will look like after the transformation.

Finally when we are done with all the operators we can get rid of all the parenthesis. They are not needed in postfix expression. This is how you can do the conversion manually.We will discuss efficient ways of doing this programmatically in later lessons. We will discuss algorithms to convert infix to prefix or postfix in later lessons. In this lesson we are only going to look at algorithms to evaluate prefix and postfix expressions.Okay, so we have this postfix expression here and we want to evaluate this expression. Let's say for these values of variables A, B, C, D and E. So we have this expression in terms of values to evaluate. I'll first quickly tell you how you can evaluate a postfix expression manually.

What you need to do is you need to scan the expression from left to right and find the first occurrence of an operator. Like here multiplication is the first operator. In postfix expression, operands of an operator will always lie to its left.

For the first operator, the preceding two entities will always be operands. You need to look for the first occurrence of this pattern operand, operand, operator in the expression and now you can apply the operator on these two operands and reduce the expression. So this is what I'm getting after evaluating 2, 3 asterisk.

Now we need to repeat this process till we are done with all the operators. Once again we need to scan the expression from left to right and look for the first operator. If the expression is correct, it will be preceded by two values.So basically we need to look for first occurrence of this pattern operand, operand, operator. So now we can reduce this. We have 6 and then we have 5 into 4 20.

We are using space as delimiter here. There should be some space in between two operands. Okay so this is what I have now.Once again I'll look for the first occurrence of operand, operand and operator. We will go on like this till we are done with all the operators. When I'm saying we need to look for first occurrence of this pattern operand, operand and operator, what I mean by operand here is a value and not a complex expression itself.

The first operator will always be preceded by two values and if you will give this some thought, you will be able to understand why. If you can see in this expression, we are applying the operators in the same order in which we have them while parsing from left to right. So first we are applying this leftmost multiplication on 2 and 3, then we are applying the next multiplication on 5 and 4, then we are performing the addition and then finally we are performing the subtraction and whenever we are performing an operation, we are picking the last two operands preceding the operator in the expression.So if we have to do this programmatically, if we have to evaluate a postfix expression given to us in a string like this and let's say operands and operators are separated by space, we can have some other delimiter like comma also to separate operands and operator. Now what we can do is we can parse the string from left to right. In each step in this parsing, in each step in this scanning process, we can get a token that will either be an operator or an operand.What we can do is as we parse from left to right we can keep track of all the operands seen so far and I'll come back to how it will help us. So I'm keeping all the operands seen so far in a list. The first entity that we have here is 2 which is an operand so it will go to the list.Next we have 3 which once again is operand so it will go into the list. Next we have this multiplication operator. Now this multiplication should be applied to last two operands preceding it.Last two operands to the left of it because we already have the element stored in this list. All we need to do is we need to pick the last two from this list and perform the operation. It should be 2 into 3 and with this multiplication we have reduced the expression.

This 2 3 asterisk has now become 6. It has become an operand that can be used by an operator later. We are at this stage right now that I'm showing in the right. I'll continue the scanning.

Next we have an operand. We'll push this number 5 on to the list. Next we have 4 which once again will come to the list and now we have the multiplication operator and it should be applied to the last two operands in the reduced expression and we should put the result back into the list.

This is the stage where we are right now. So this list actually is storing all the operands in the reduced expression preceding the position at which we are during parsing. Now for this addition we should take out the last two elements from the list and then we should put the result back.

Next we have an operand. We are at this stage right now. Next we have an operator.This subtraction, we will perform this subtraction and put the result back. Finally when I'm done scanning the whole expression I'll have only one element left in the list and this will be my final answer. This will be my final result.This is an efficient algorithm. We are doing only one pass on the string representing the expression and we have our result. The list that we are using here if you could notice is being used in a special way.We are inserting operands one at a time from one side and then to perform an operation we are taking out operand from the same side. Whatever is coming in last is getting out first. This whole thing that we are doing here with the list can be done efficiently with a stack which is nothing but a special kind of list in which elements are inserted and removed from the same side in which whatever gets in last comes out first.

It's called a last in first out structure. Let's do this evaluation again. I have drawn logical representation of stack here and this time I'm going to use this stack.

I'll also write pseudocode for this algorithm. I'm going to write a function named evaluate postfix that will take a string as argument. Let's name this string expression exp for expression.

In my function here I'll first create a stack. Now for the sake of simplicity let's assume that each operand or operator in the expression will be of only one character. So to get a token or operator we can simply run a loop from 0 till length of expression minus 1. So expression i will be my operand or operator.

If expression i is operand I should push it on to the stack else if expression i is operator we should do two pop operations in the stack. Store the value of the operands in some variable. I'm using variables named op1 and op2.

Let's say this pop function will remove an element from top of stack s and also return this element. Once we have the two operands we can perform the operation. I'm using this variable to store the output.Let's say this function will perform the operation. Now the result should be pushed back on to the stack. If I have to run through this expression with whatever code I have right now then first entity is 2 which is operand so it should be pushed on to the stack.

Next we have 3 once again this will go to the stack. Next we have this multiplication operator. So we will come to this else if part of the code.

I'll make first pop and I'll store 3 in this variable op1. Well actually this is the second operand. So I should say this one is op2 and next one will be op1.

Once I have popped these two elements I can perform the operation. As you can see I'm doing the same stuff that I was doing with the list. The only thing is that I'm showing things vertically.

Stack is being shown as a vertical list. I'm inserting or taking out from the top. Now I'll push the result back on to the stack.Now we will move to the next entity which is operand. It will go on to the stack. Next 4 will also go on to the stack and now we have this multiplication.

So we will perform two pop operations. After this operation is performed result will be pushed back. Next we have addition.So we will go on like this. We have 26 pushed on to the stack now. Now it's 9 which will go in and finally we have this subtraction.26 minus 9, 17 will be pushed on to the stack. At this stage we will be done with the loop. We are done with all the tokens, all the operands and operators.

The top of stack can be returned as final result. At this stage we will have only one element in the stack and this element will be my final result. You will have to take care of some parsing logic in actual implementation.

Operand can be a number of multiple digits and then we will have delimiter like space or comma. So you'll have to take care of that. Parsing operand or operator will be some task.

If you want to see my implementation you can check the description of this video for a link. Okay so this was postfix evaluation. Let's now quickly see how we can do prefix evaluation.Once again I've written this expression in infix form and I'll first convert it to prefix. We will go in order of precedence. I first put this parenthesis.

This two asterisks 3 will become asterisks 23. This 5 into 4 will become asterisks 54 and now we will pick this plus operator whose operands are these two prefix expressions. Finally for the subtraction operator this is the first operand and this is the second operand.In the last step we can get rid of all the parenthesis. So this is what I have finally. Let's now see how we can evaluate a prefix expression like this.

We will do it just like postfix. This time all we need to do is we need to scan from right. So we will go from right to left.Once again we will use a stack. If it's an operand we can push it onto the stack. So here for this example 9 will go onto the stack and now we will go to the next entity in the left.It's 4. Once again we have an operand. It will go onto the stack. Now we have 5. 5 will also be pushed onto the stack and now we have this multiplication operator.

At this stage we need to pop two elements from the stack. This time the first element popped will be the first operand. In postfix the first element popped was the second operand.

This time the second element popped will be the second operand. For this multiplication first operand is 5 and second operand is 4. This order is really important. For multiplication the order doesn't matter but for say division or subtraction this will matter.

Result 20 will be pushed onto the stack and we will keep moving left. Now we have 3 and 2. Both will go onto the stack and now we have this multiplication operation. 3 and 2 will be popped and their product 6 will be pushed.Now we have this addition. The two elements at top are 20 and 6. They will be popped and their sum 26 will be pushed. Finally we have this subtraction.26 and 9 will be popped out and 17 will be pushed and finally this is my answer. Prefix evaluation can be performed in couple of other ways also but this is easiest and most straightforward. Okay so this was prefix and postfix evaluation using stack.

In coming lessons we will see efficient algorithms to convert infix to prefix or postfix. This is it for this lesson. Thanks for watching.

In our previous lesson we saw how we can evaluate prefix and postfix expressions. Now in this lesson we will see an efficient algorithm to convert infix to postfix. We already know of one way of doing this.We have seen how we can do this manually. To convert an infix expression to postfix we apply operator precedence and associativity rules. Let's do the conversion for this expression that I have written here.The precedence of multiplication operator is higher. So we will first convert this part B asterisk C. B asterisk C will become BC asterisk. The operator will come in front of the operands.

Now we can do the conversion for this addition. For addition the operands are A and this postfix expression. In the final step we can get rid of all the parentheses.So finally this is my postfix expression. We can use this logic in a program also but it will not be very efficient and implementation will also be somewhat complex. I'm going to talk about one algorithm which is really simple and efficient and in this algorithm we need to parse the infix expression only once from left to right and we can create the postfix expression.If you can see in infix to postfix conversion the positions of operands and operators may change but the order in which operands occur from left to right will not change. The order of operators may change. This is an important observation.

In both infix and postfix forms here the order of operands as we go from left to right is first we have A then we have B and then we have C but the order of operators is different. In infix first we have plus and then we have multiplication. In postfix first we have multiplication and then addition.In postfix form we will always have the operators in the same order in which they should be executed. I'm going to perform this conversion once again but this time I'm going to use a different logic. What I'll do is I'll parse the infix expression from left to right.

So I'll go from left to right looking at each token that will either be an operand or an operator. In this expression we will start at A. A is an operand. If it's an operand we can simply append it in the postfix string or expression that we're trying to create.At least for A it should be very clear that this is nothing that can come before A. Okay so the first rule is that if it's an operand we can simply put it in the postfix expression. Moving on next we have an operator. We cannot put the operator in the postfix expression because we have not seen it's right operand yet.While parsing we have seen only it's left operand. We can place it only after it's right operand is also placed. So what I'm going to do is I'm going to keep this operator in a separate list or collection and place it later in the postfix expression when it can be placed and the structure that I'm going to use for storage is stack.

A stack is only a special kind of list in which whatever comes in last goes out first. Insertion and deletion happen from the same end. I have pushed plus operator onto the stack here.Moving on next we have B which is an operand. As we had said operand can simply be appended. There is nothing that can come before this operand.The operator in the stack is anyway waiting for the operand to come. Now at this stage can we place the addition operator in the postfix string. Well actually what's after B also matters.

In this case we have this multiplication operator after B which has higher precedence and so the actual operand for addition is this whole expression B asterisk C. We cannot perform the addition until multiplication is finished. So while parsing when I'm at B and I have not seen what's ahead of B I cannot decide the fate of the operator in the stack. So let's just move on.Now we have this multiplication operator. I want to make this expression further complex to explain things better. So I'm adding something at tail here in this expression.Now I want to convert this expression to postfix form. I'm not having any parenthesis here. We will see how we can deal with parenthesis later.Let's look at an expression where parenthesis does not override operator precedence. Okay so right now in this expression while parsing from left to right we are at this multiplication operator. The multiplication operator itself cannot go into the postfix expression because we have not seen its right operand yet and until its right operand is placed in the postfix expression we cannot place it.

The operator that we would be looking at while parsing that operator itself cannot be placed right away. But looking at that operator we can decide whether something from the collection, something from the stack can be placed into the postfix expression that we are constructing or not. Any operator in the stack having higher precedence than the operator that we are looking at can be popped and placed into the postfix expression.

Let's just follow this as rule for now and I'll explain it later. There is only one operator in the stack and it is not having higher precedence than multiplication. So we will not pop it and place it in the postfix expression.Multiplication itself will be pushed. If an element in the stack has something on top of it that something will always be of higher precedence. So let's move on in this expression now.

Now we are at C which is an operand so it can simply go. Next we have an operator subtraction. Subtraction itself cannot go.But as we had said if there is anything on the stack having higher precedence than the operator that we are looking at it should be popped out and should go. And the question is why? We are putting these operators in the stack. We are not placing them in the postfix expression because we are not sure whether we are done with the right operand or not.But after that operator as soon as I'm getting an operator of lower precedence that marks the boundary of the right operand. For this multiplication operator C is my right operand. It's this simple variable.For addition B asterisk C is my right operand. Because subtraction has lower precedence anything on or after that cannot be part of my right operand. Subtraction I should say has lower priority because of the associativity rule.If you remember the order of operation addition and subtraction have same precedence but the one that would occur in left would be given preference. So the idea is anytime for an operator if I'm getting an operator of lower priority we can pop it from the stack and place it in the expression. Here we will first pop multiplication and place it and then we can pop addition and now we will push subtraction onto the stack.

Let's move on now. D is an operand so it will simply go. Next we have multiplication.

There is nothing in the stack having higher precedence than multiplication so we will pop nothing. Multiplication will go onto the stack. Next we have an operand.

It will simply go. Now there are two ways in which we can find the end of right operand for an operator. A is if we get an operator of lesser precedence B if we reach the end of the expression.

Now that we have reached end of expression we can simply pop and place these operators. So first multiplication will go and then subtraction will go. Let's quickly write pseudocode for whatever I have said so far and then you can sit with some examples and analyze the logic.I'm going to write a function named infix to postfix that will take a string exp for expression as argument. For the sake of simplicity let's assume that each operand or operator will be of one character only. In an actual implementation you can assume them to be tokens of multiple characters.

So in my pseudocode here the first thing that I'll do is I'll create a stack of characters named S. Now I'll run a loop starting 0 till length of expression minus 1. So I'm looking at each character that can either be an operand or operator. If the character is an operand we can append it to the postfix string. Well actually I should have declared and initialized a string before this loop.

This is the result string in which I'll be appending. Else if expression I is operator we need to look for operators in the stack having higher precedence. So I'll say while stack is not empty and the top of stack has higher precedence and let's say this function has higher precedence will take two arguments, two operators.

So if the top of stack has higher precedence than the operator that we are looking at we can append the top of stack to the result which is the variable that will store the postfix string and then we can pop that operator. I'm assuming that this S is some class that has these functions stop and pop and empty to check whether it's empty or not. Finally once I'm done with the popping outside this while loop I need to push the current operator.S is an object of some class that will have these functions. Stop, pop and empty. Okay so this is the end of my for loop.At the end of it I may have some operators left in the stack. I'll pop these operators and append them to the postfix string. I'll use this while loop.

I'll say that while the stack is not empty, append the operator at top and pop it and finally after this while loop I can return the result string that will contain my postfix expression. So this is my pseudocode for whatever logic I've explained so far. In my logic I've not taken care of parenthesis.

What if my infix expression would have parenthesis like this? There will be slight change from what we were doing previously. With parenthesis any part of the expression within parenthesis should be treated as independent complete expression in itself and no element outside the parenthesis will influence its execution. In this expression this part A plus B is within one parenthesis.Its execution will not be influenced by this multiplication or this subtraction which is outside it. Similarly this whole thing is within the outer parenthesis. So this multiplication operator outside will not have any influence on execution of this part as a whole.

If parenthesis are nested inner parenthesis is sorted out or resolved first and then only outer parenthesis can be resolved. With parenthesis we will have some extra rules. We will still go from left to right and we will still use stack and let's say I'm going to write the postfix part in right here as I created.Now while parsing, a token can be an operand, an operator or an opening or closing of parenthesis. We will have some extra rules. I'll first tell them and then I'll explain.If it's an opening parenthesis we can push it on to the stack. The first token here in this example is an opening parenthesis so it will be pushed onto the stack and then we will move on. We have an opening parenthesis once again so once again we will push it.Now we have an operand. There is no change in rule for operand. It will simply be appended to the postfix part.Next we have an operator. Remember what we were doing for operator earlier. We were looking at top of stack and popping as long as we were getting operator of higher precedence.Earlier when we were not using parenthesis, we could go on popping and empty the stack but now we need to look at top of stack and pop only till we get an opening parenthesis because if we are getting an opening parenthesis then it's the boundary of the last open parenthesis and this operator does not have any influence after that outside that. So this plus operator does not have any influence outside this opening parenthesis. I'll explain the scenario with some more examples later.Let's first understand the rule. So the rule is if I'm seeing an operator I need to look at the top of stack. If it's an operator of higher precedence I can pop and then I should look at the next top.

If it's once again an operator of higher precedence, I should pop again but I should stop when I see an opening parenthesis. At this stage we have an opening parenthesis at top so we do not need to look below it. Nothing will be popped anyway.Addition however will go on to the stack. Remember after the whole popping game we push the operator itself. Next we have an operand.It will go and we will move on. Next we have a closing of parenthesis. When I'm getting a closing of parenthesis I'm getting a logical end of the last opened parenthesis.For part of the expression within that parenthesis it's coming to the end and remember what we were doing earlier when we were reaching the end of infix expression. We were popping all the operators out and placing them. So this time also we need to pop all the operators out but only those operators that are part of this parenthesis that we are closing.

So we need to pop all the operators until we get an opening parenthesis. I'm popping this plus and appending it. Next we have an opening of parenthesis so I'll stop but as last step I will I will pop this opening also because we are done for this parenthesis.Okay so the rule for closing of parenthesis is pop until you're getting an opening parenthesis and then finally pop that particular opening parenthesis also. Let's move on now. Next we have an operator.

We need to look at top of stack. It's an opening of parenthesis. This operator will simply be pushed.Next we have an operand. Next we have an operator. Once again we will look at the top.

We have multiplication which is higher precedence. So this should be popped and appended. We will look at the top again.

It's an opening of parenthesis so we should stop looking now. Minus will be pushed now. Next we have an operand.Next we have closing of parenthesis. So we need to pop until we get an opening. Minus will be appended.

Finally the opening will also be popped. Next we have an operator and this will simply go. Next we have an operand and now we have reached the end of expression.

So everything in the stack will be popped and appended. So this finally is my postfix expression. I'll take one more example and convert it to make things further clear.I want to convert this expression. I'll start at the beginning. First we have an operand.Then this multiplication operator which will simply go on to the stack. The stack right now is empty. There is nothing on the top to convert.

ds-9

