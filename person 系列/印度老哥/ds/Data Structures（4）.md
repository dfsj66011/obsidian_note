
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


## 十九、

Hello everyone.

In this lesson, we are going to talk about one important and really interesting topic in computer science where we find application of stack data structure and this topic is evaluation of arithmetic and logical expressions. So, how do we write an expression? I have written some simple arithmetic expressions here. An expression can have constants, variables and symbols that can be operators or parenthesis and all these components must be arranged according to a set of rules, according to a grammar and we should be able to parse and evaluate the expression according to this grammar.

All these expressions that I have written here have a common structure. We have an operator in between two operands. Operand by definition is an object or value on which operation is performed.

In this expression 2 plus 3, 2 and 3 are operands and plus is operator. In the next expression, a and b are operands and minus is operator. In the third expression, this asterisk is for multiplication operation.

So, so this is the operator. The first operand p is a variable and the second operand 2 is a constant. This is the common way of writing an expression but this is not the only way.

This way of writing an expression in which we write an operator in between operands is called infix notation. Operand doesn't always have to be a constant or variable. Operand can be an expression itself.

In this fourth expression that I have written here, one of the operands of multiplication operator is an expression itself. Another operand is a constant. We can have a further complex expression.In this fifth expression that I have written here, both the operands of multiplication operator are expressions. We have three operators in this expression here. For this first plus operator p and q, these variables p and q are operands.

For the second plus operator, we have r and s and for this multiplication operator, the first operand is this expression p plus q and the second operand is this expression r plus s. While evaluating expressions with multiple operators, operations will have to be performed in certain order. Like in this fourth example, we will first have to perform the addition and then only we can perform multiplication. In this fifth expression, first we will have to perform these two additions and then we can perform the multiplication.

We will come back to evaluation but if you can see, in all these expressions, operator is placed in between operands. This is the syntax that we are following. One thing that I must point out here, throughout this lesson, we are going to talk only about binary operators.

An operator that requires exactly two operands is called a binary operator. Technically, we can have an operator that may require just one operand or maybe more than two operands. But we are talking only about expressions with binary operators.

Okay, so let's now see what all rules we need to apply to evaluate such expressions written in this syntax that we are calling infix notation. For an expression with just one operator, there is no problem, we can simply apply that operator. For an expression with multiple operators and no parenthesis like this, we need to decide an order in which operators should be applied.

In this expression, if we will perform the addition first, then this expression will reduce to 10 into two and will finally evaluate as 20. But if we will perform the multiplication first, then this expression will reduce to four plus 12 and will finally evaluate to 16. So basically, we can look at this expression in two ways.

We can say that operands for addition operator are four and six. And operands for multiplication are this expression four plus six, and this constant two. Or we can say that operands for multiplication are six and two.And operands for addition operation are four and this expression six into two. There is some ambiguity here. But if you remember your high school mathematics, this problem is resolved by following operator precedence rule.

In an algebraic expression, this is the precedence that we follow. First preference is given to parenthesis or brackets. Next preference is given to exponents.

I'm using this symbol for exponent operator. So, if I have to write two to the power three, I'll be writing it something like this. In case of multiple exponentiation operator, we apply the operators from right to left.

So, if I have something like this, then first this right most exponentiation operator will be applied. So, this will reduce to 512. If you will apply the left operator first, then this will evaluate to 64.After exponents, next preference is given to multiplication and division. And if it's between multiplication and division operators, then we should go from left to right. After multiplication and division, we have addition and subtraction.

And here also we go from left to right. If we have an expression like this, with just addition and subtraction operators, then we will apply the left most operator first, because the precedence of these operators is same. And this will evaluate to three.

If you will apply the plus operator first, this will evaluate as one and that will be wrong. In the second expression, four plus six into two that I have written here, if we will apply operator precedence, then multiplication should be performed first. If we want to perform the addition first, then we need to write this four plus six within parentheses.

And now addition will be performed first because precedence of parentheses is greater. I'll take example of another complex expression and try to evaluate it just to make things further clear. So, I have an expression here.

In this expression, we have four operators, one multiplication, one division, one subtraction and one addition. Multiplication and division have higher precedence. Between these two multiplication and division, which have same precedence, we will pick the left one first.

So, we will first reduce this expression like this. And now we will perform the division. And now we have only subtraction and addition.

So, we will go from left to right. And this is what we will finally get this right to left and left to right rule that I've written here for operators with equal precedence is better termed as operator associativity. If in case of multiple operators with equal precedence, we go from left to right, then we say that the operators are left associative.And if we go from right to left, we say that the operators are right associative. While evaluating an expression in infix form, we first need to look at precedence. And then to resolve conflict among operators with equal precedence, we need to see associativity.All in all, we need to do so many things just to parse and evaluate an infix expression, the use of parentheses becomes really important, because that's how we can control the order in which operation should be performed. Parentheses add explicit intent, that operation should be performed in this order, and also improve readability of expression. I have modified this third expression, we have some parentheses here now.

And most often we write infix expressions like this only using a lot of parentheses. Even though infix notation is the most common way of writing expressions, it's not very easy to parse and evaluate an infix expression without ambiguity. So mathematicians and logicians studied this problem and came up with two other ways of writing expressions that are parentheses free and can be parsed without ambiguity without requiring to take care of any of these operator precedence or associativity rules.

And these two ways are postfix and prefix notations. Prefix notation was proposed earlier in year 1924 by a Polish logician. Prefix notation is also known as Polish notation.

In prefix notation operator is placed before operands. This expression two plus three in infix will be written as plus two three in prefix plus operator will be placed before the two operands two and three p minus q will be written as minus pq. Once again, just like infix notation operand in prefix notation doesn't always have to be a constant or variable operand can be a complex prefix notation itself.This expression a plus b asterisk c in infix form will be written like this in prefix form. I'll come back to how we can convert infix expression to prefix. First have a look at this third expression in prefix form.

For this multiplication operator, the two operands are variables B and C. These three elements are in prefix syntax. First we have the operator and then we have the two operands. The operands for addition operator are variable A and this prefix expression asterisk BC.

In infix expression, we need to use parentheses because an operand can possibly be associated with two operators like in this third expression in infix form B can be associated with both plus and multiplication. To resolve this conflict, we need to use operator precedence and associativity rules or use parentheses to explicitly specify association. But in prefix form, and also in postfix form that we will discuss in some time, an operand can be associated with only one operator.

So we do not have this ambiguity. While parsing and evaluating prefix and postfix expressions, we do not need extra information, we do not need all the operator precedence and associativity rules. I'll come back to how we can evaluate prefix notation.I'll first define postfix notation. Postfix notation is also known as reverse polished notation. This syntax was proposed in 1950s by some computer scientists.

In postfix notation, operator is placed after operands. Programmatically postfix expression is easiest to parse and least costly in terms of time and memory to evaluate. And that's why this was actually invented.

Prefix expression can also be evaluated in similar time and memory. But the algorithm to parse and evaluate postfix expression is really straightforward and intuitive. And that's why it's preferred for computation using machines.

I'm going to write postfix for these expressions that I had written earlier. In other forms, this first expression two plus three in postfix will be two three plus. To separate the operands, we can use a space or some other delimiter like a comma.That's how you would typically store prefix or postfix in a string when you will have to write a program. This second expression in postfix will be pq minus. So as you can see in postfix form, we are placing the operator after the operands.

This third expression in postfix will be ABC asterisk and then plus. For this multiplication operator, operands are variables B and C. And for this addition, operands are variable A and this postfix expression BC asterisk. We will see efficient algorithms to convert infix to prefix or postfix in later lessons.

For now, let's not bother how we will do this in a program. Let's quickly see how we can do this manually. To convert an expression from infix to any of these other two forms, we need to go step by step, just the way we would go in evaluation.

I have picked this expression A plus B into C in infix form, we should first convert the part that should be evaluated first. So we should go in order of precedence, we can also first put all the implicit parentheses. So here we will first convert this P into C. So first we are doing this conversion for multiplication operator and then we will do this conversion for addition operator, we will bring addition to the front.

So this is how the expression will transform, we can use parentheses in in intermediate steps. And once we are done with all the steps, we can erase the parentheses. Let's now do the same thing for postfix, we will first do the conversion for multiplication operator.

And then in next step, we will do it for addition. And now we can get rid of all the parentheses. parentheses surely adds readability to any of these expressions to any of these forms.

But if we are not bothered about human readability, then for a machine, we are actually saving some memory that would be used to store parentheses information. infix expression definitely is most human readable, but prefix and postfix are good for machines. So this is infix prefix and postfix notation for


ds-8