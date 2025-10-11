
## 一、引言

在这一课以及这一系列课程中，我们将向你介绍数据结构的概念。数据结构是计算机科学中最基础、最核心的概念之一。掌握良好的数据结构知识是设计和开发高效软件系统的必备条件。好了，让我们开始吧。我们时时刻刻都在处理数据，而如何存储、组织和分组这些数据至关重要。让我们从日常生活中举几个例子，在这些例子中，以特定结构组织数据对我们大有裨益——比如我们能快速高效地从语言词典中查找到某个单词。

因为字典中的单词是按顺序排列的。如果字典中的单词没有排序，要在数百万个单词中查找一个单词将是不切实际甚至不可能的。因此，字典被组织成一个排序后的单词列表。我们再举一个例子。如果我们有一张城市地图，像地标位置和道路网络连接这样的数据，所有这些数据都是以几何形式组织的，我们在二维平面上以这些几何形式展示地图数据。因此，地图数据需要这样结构化，以便我们有比例尺和方向，从而能够有效地搜索地标并获取从一个地方到另一个地方的路线。

我再举一个例子，比如企业的每日现金收支报表，在会计中我们也称之为现金账簿，以表格形式组织和存储数据是最合理的。如果数据按照这些表格中的列进行组织，就很容易汇总数据和提取信息。因此，需要不同的结构来组织不同类型的数据。

如今，计算机能处理各种类型的数据。无论是文本、图像、视频、关系型数据、地理空间数据，还是地球上几乎任何形式的数据，计算机都能处理。我们如何在计算机中存储、组织和分类数据至关重要，因为计算机处理的数据量极其庞大。即便计算机拥有强大的计算能力，如果我们不使用正确的数据结构、不采用合理的逻辑架构，我们的软件系统就无法高效运行。

数据结构的正式定义是：数据结构是一种在计算机中存储和组织数据的方式，以便高效地使用这些数据。当我们研究数据结构作为存储和组织数据的方式时，我们从两个方面进行研究。可以说，我们讨论数据结构时，*一方面是从数学和逻辑模型的角度来讨论它们*。

当我们把它们作为数学和逻辑模型来讨论时，我们只是从抽象的角度来看待它们。我们只是从高层次上观察，哪些特征和哪些操作定义了那个特定的数据结构。现实世界中抽象视角的例子可以类似于电视这种设备的抽象视角，即它是一种可以开关的电子设备。它可以接收卫星节目的信号并播放节目的音频视频。只要我有这样一个设备，我就不用操心电路是如何嵌入到这个设备中的，或者哪家公司制造了这个设备。所以这是一个抽象的概念。

因此，当我们把数据结构作为数学或逻辑模型来研究时，我们只是定义它们的抽象视图。换句话说，我们有一个术语来描述这一点，即将其定义为抽象数据类型。抽象数据类型的一个例子可以是：我想定义一个称为列表的东西，它应该能够存储一组特定数据类型的元素。

我们应该能够通过元素在列表中的位置来读取它们。我们还应该能够修改列表中特定位置的元素，可以说，存储任意数据类型的给定数量的元素。因此，我们只是在定义一个模型。现在我们可以通过多种方式在编程语言中实现这一点。所以这是一个抽象数据类型的定义，我们也把*抽象数据类型称为 ADT*。如果你注意到，所有高级语言都已经以数组的形式实现了这样的 ADT 的具体实现。

数组为我们提供了所有这些功能。因此，数组是具体实现的数据类型。讨论数据结构的*第二种方式是讨论它们的实现*。实现将是一些具体类型，而不是抽象数据类型。我们可以在同一种语言中以多种方式实现相同的 ADT。例如，在 C 或 C++ 中，我们可以将这个列表 ADT 实现为名为链表的数据结构。

如果你还没有听说过它，我们将会在接下来的课程中频繁讨论它们，特别是链表。好了，让我们正式定义一个抽象数据类型，因为这是一个我们会经常遇到的术语。*抽象数据类型是对数据和操作的定义的实体，但没有具体的实现*。也就是说，它们没有任何实现细节。在这门课程中，我们会讨论很多数据结构，我们会将它们作为抽象数据类型来讨论。

我们还将探讨如何实现这些数据结构。我们将讨论的一些数据结构包括数组、链表、栈、队列、树、图等，还有很多需要学习的内容。在学习这些数据结构时，我们会研究它们的*逻辑视图*，了解可以对它们执行*哪些操作*，并分析这些*操作的成本*，主要是时间成本。当然，我们还会学习如何在*编程语言中实现*这些数据结构。所以我们将在接下来的课程中学习所有这些数据结构。这就是本入门课程的全部内容。

## 二、抽象数据类型-列表

在之前的课程中，我们向大家介绍了数据结构的概念。我们了解到可以从两个角度来探讨数据结构：一是将其视为数学和逻辑模型，也就是我们所说的抽象数据类型（ADT）；二是研究数据结构的具体实现方式。在本节课中，我们将学习一种简单的数据结构，首先会从抽象层面进行定义，将其描述为抽象数据类型。

然后我们将看看可能的实现方式。这种数据结构就是列表，列表是现实世界中常见的实体。列表不过是*同一类型*对象的集合，我们可以有一个单词列表，可以有一个名字列表，也可以有一个数字列表。那么让我们首先将列表定义为一种抽象数据类型。

因此，当我们定义抽象数据类型时，我们只需定义将要存储的数据以及该类型可用的操作，而不涉及具体的实现细节。首先，让我们定义一个非常基础的列表：我需要一个能够存储给定数据类型特定数量元素的列表，这将是一个*静态列表*。列表中的元素数量不会改变，并且在创建列表之前我们就知道元素的数量。此外，我们应该能够写入或修改列表中任意位置的元素。

当然，我们还应该能够读取列表中特定位置的元素。所以，如果我让你实现这样一个列表，而你上过基础的编程入门课程，你可能会说，嘿，这个我知道。数组就能提供所有这些功能，所有这些操作都可以通过数组来实现，我们可以创建任何数据类型的数组。

假设我们要创建一个整数列表，那么我们可以声明数组类型为整数，并在声明时指定大小作为参数(`int A[10];`)。我可以写入或修改特定位置的元素，这些元素从零开始索引，访问方式类似这样，大家对数组都很熟悉。我们还可以读取特定位置的元素，比如第 i 个位置的元素通过 `A[i]` 来访问。因此，数组就是为我们实现这种列表功能的数据结构。

现在我需要一个功能更丰富的列表，它能帮我处理更多场景。因此，我将在这里重新定义这个列表。我不想要一个静态的、固定大小的集合，而是需要一个能根据需求动态扩展的列表。具体来说，我的列表需要具备以下功能：当列表中没有元素时，我会称其为空列表，此时列表的大小为零。然后，我可以向列表中插入元素。

我可以在列表中的任意位置插入元素，也可以在现有列表中移除元素，还能统计列表中元素的数量。此外，我应当能够读取、写入或修改列表中特定位置的元素，并且能够为列表指定数据类型。因此，在创建列表时，我应该能够指定这是一个整数列表、字符串列表、浮点数列表还是其他类型的列表。现在我想要一个实现这种动态列表的数据结构。那么如何实现呢？实际上，我们可以使用数组来实现这样的动态列表。只是我们需要在数组的基础上编写更多的操作来实现所有这些功能。

那么，让我们看看如何使用数组来实现这个特定的列表。为了简化设计，我们假设列表的数据类型为整数。所以我们正在创建一个动态整数列表。我们可以实现这样的列表，方法是声明一个非常大的数组，我们会定义一个最大容量，然后声明一个这个最大容量的数组。现在我们知道，数组中的元素索引是从零开始的，零、一、二，依此类推。

因此，我会定义一个变量来标记这个数组中列表的末尾。如果列表为空，我们可以初始化这个变量，或者将其设为负一，因为最小的可能索引是零。所以，如果 n 等于负一，就表示列表为空。在任何时候，数组的一部分将用来存储这个列表。

```c
int A[MAXSIZE];
int end = -1;
insert(2);        // 默认插入在最后位置
insert(4);
insert(6);
insert(7);
insert(9);
insert(5, 2);     // 指定插入位置
```

好的，假设一开始列表为空时，这个指针 end 指向索引 -1 的位置，这是无效的，不存在的。现在我要往这个数组中插入一个整数。如果我们不指定插入数字的位置，那么这个数字总是会被插入到列表的尾部，也就是列表的末端。

因此，列表的结构将是这样的：位置零会有一个元素，现在 end 的索引是零。所以，在任何时候，end 这个变量标记着数组中列表的末尾。现在，如果我想在列表的特定位置插入某个元素，比如说我想在索引二的位置插入数字五，那么为了在这个特定位置容纳五，我们必须将所有元素向右移动一个单位。从索引二开始的所有元素，我们需要将索引二开始的所有元素向右移动。好的，我刚刚在列表中插入了一些元素。让我也为这些操作编写函数调用，假设我们按照这个顺序进行，先插入二，然后插入四，最后插入五，并且我们还会指定要插入的位置。因此，这个带有两个参数的插入操作就是在特定位置插入元素的调用。经过所有这些操作和插入后，列表将呈现这样的形态。这里的箭头标记了数组中列表的末端。

现在如果我想从特定位置移除一个元素，比如说我调用 remove 函数，我想移除第二个元素。那么我会在这里传入索引 0，我想移除索引 0 处的元素。为此，索引 0 之后的所有元素都会向左或向更低的索引方向移动一个单位。2 会离开。现在这里的 end 变量在我们每次插入后都会进行调整。所以这次插入后，end 会变成零，123之后也是如此。这次移除后，end 又会变成四。好的，看起来我们左边已经有了一个实现这个列表的方法，它被描述为一个抽象数据类型。

当变量 end 等于 -1 时，我们采用一种逻辑将列表视为空。我们可以在列表的特定位置插入元素，也可以移除元素，只不过需要在数组中进行一些移位操作。我们可以统计列表中元素的数量，其值等于变量 end+1。此外，我们还可以读取或修改某个位置上的元素。嗯，这是一个数组。因此我们完全可以读取或修改特定位置的元素。如果我们需要选择数据类型，那只需选择该特定数据类型的数组即可。

这看起来是个很酷的实现，但我们有一个问题：我们说过数组会有某个较大的尺寸，即某个最大尺寸。但什么才是合适的最大尺寸呢？数组总是有可能被耗尽，列表的增长也总是可能耗尽数组，没有一个真正合适的最大尺寸。因此，我们*需要制定一个策略，以应对列表填满整个数组的情况*。

那么在这种情况下我们该怎么办呢？我们需要在设计时考虑到这一点，想直接扩展原有数组是不可能实现的。因此，我们*必须创建一个新的、更大的数组*。当原数组存满时，我们会新建一个更大的数组，并将原数组中的所有元素复制到新数组中。然后我们可以*释放之前数组的内存*。

现在的问题是，*新数组的大小应该增加多少*？创建新数组并将所有元素从旧数组复制到新数组的整个过程在时间上是昂贵的，一个好的设计肯定是避免如此大的成本。因此，我们选择的策略是每次数组满时，创建一个新的更大的数组，其大小是*之前数组的两倍*。而为什么这是最佳策略，我们不会在本课中讨论。因此，我们将创建一个更大的双倍大小的数组，并将元素从之前的数组复制到这个新数组中。这看起来是一个很酷的实现。

数据结构的研究不仅仅涉及这些操作及其实现方式的研究，还包括分析这些操作的成本。因此，让我们来看看动态列表中所有这些操作在时间上的成本是多少。访问这个动态列表中的任何元素，如果我们想通过索引来读取或写入，那么这将花费常数时间，因为我们这里有一个数组。

在数组中，元素被排列在一个连续的内存块中。通过使用内存块的起始地址或基地址，以及元素的索引或位置，我们可以计算出特定元素的地址，并在常数时间内访问它。大 O 表示法用于描述操作的时间复杂度。对于常数时间，用大 O 表示法表示为时间复杂度为 O(1)。如果我们要在数组或列表的末尾插入元素，那仍然是常数时间。但如果我们要在列表的特定位置插入元素，就需要将元素向更高的索引位置移动。在最坏的情况下，当我们在第一个位置插入元素时，需要将所有元素向右移动。

因此，插入所需的时间将与列表的长度成正比，假设列表的长度为 n。换句话说，从时间复杂度来看，插入操作的时间复杂度为 O(n)。如果你不了解大 O 表示法，不用担心，只需明白在特定位置插入元素的时间消耗与列表大小呈线性关系即可。删除一个元素的时间复杂度仍然是 O(n)，所需时间与列表当前的大小成正比，这里的 n 代表列表的大小。

好，现在说到在末尾插入一个元素，我们刚才说这可以在常数时间内完成，但如果数组已满就不是这样了，这时我们需要创建一个新数组。我们把在末尾插入元素称为添加元素，如果列表未满，添加元素只需常数时间；但如果数组已满，所需时间将与数组的大小成正比。

所以在最坏情况下，添加操作的时间复杂度又会回到 O(n)。正如我们所说，当列表已满时，我们会创建一个大小为原数组两倍的新数组，然后将原数组中的元素复制到新数组中。那么乍一看，这种实现方式有什么优点呢？嗯，优点在于我们可以在常数时间内访问任意索引处的元素，这正是数组的特性。

但如果我们需要在中间插入某个元素，或者要从列表中移除元素，那么操作代价就很高。如果列表频繁增减，我们就不得不反复创建新数组，并一次次将旧数组中的元素复制到新数组中。另一个问题是，很多时候数组的大部分空间未被使用，那些内存就被白白浪费了。

而且，使用数组作为动态列表在内存方面效率不高，这种实现在内存方面效率低下。这让我们思考，是否有一种数据结构既能提供动态列表的功能，又能更高效地利用内存？我们确实有一种数据结构可以很好地利用内存，那就是链表。

## 三、链表

在本节课中，我们将向您介绍链表数据结构。在上一节课中，我们尝试使用数组实现动态列表，但遇到了一些问题。就内存消耗而言，这种方法在内存使用效率上并不高。使用数组时存在一些局限性。为了能够理解链表。我们需要理解这些限制。所以我将讲一个简单的故事来帮助你理解这一点。

假设这是计算机的内存，这里的每个分区都是一个字节的内存。现在我们知道，内存中的每个字节都有一个地址，这里我们只展示了一部分内存，所以它向上下延伸。假设地址从下往上递增。那么如果这个字节的地址是 200，下一个字节的地址就是 201。下一个字节的地址是 202，以此类推。我想做的是从左到右水平绘制这段内存，而不是像这样从下到上尝试。这样看起来更好。

假设这里的字节是地址 200。当我们向右移动时，地址会增加。所以这个就是 201。而我们就这样继续下去，202、203，依此类推。内存是从下往上显示还是从左往右显示其实并不重要。这些都只是查看内存的逻辑方式。

回到我们的故事，内存是一种关键资源，所有应用程序都在不断请求使用它。于是计算机先生将管理内存的任务交给了他的一个组件——他称之为内存管理器。这个家伙负责跟踪哪些内存区域是空闲的，哪些内存区域已被分配。任何需要内存来存储东西的人都需要和这个人沟通。阿尔伯特是我们的程序员，他正在开发一个应用程序，需要将一些数据存储在内存中。因此，他需要与内存管理器对话，他可以用高级语言（比如 C 语言）与内存管理器交流，假设他正在用 C 语言与内存管理器对话。

首先，他想在内存中存储一个整数。于是他通过声明一个整型变量来与内存管理器进行沟通，就像这样（`int x;`）。内存管理器看到这个声明后会说，好的，你需要存储一个整型变量。所以我需要给你四个字节的内存空间，因为在典型的架构中，整型变量占用四个字节的存储空间。

在这种架构中，假设它存储在四个字节中。因此，内存管理器在内存中寻找四个字节的空闲空间，并将其分配给变量 `x`。内存块的地址是该内存块中第一个字节的地址。假设这里的第一个字节的地址是 217。变量 `x` 的地址是 217。内存管理器会告诉 Albert："我已经为你的变量 `x` 分配了地址 217，你可以在这里存储任何你想要的数据。(`x=8;`)

"现在，Albert 需要存储一个整数列表，一组数字。他认为这个列表最多可以容纳四个整数。于是他向内存管理器申请了一个名为 A、大小为 4 的整型数组(`int A[4];`)。数组在内存中总是以连续的内存块形式存储。因此，内存管理器会这样处理：好的，我需要为这个变量（即数组 a）寻找一个 16 字节的内存块。于是，内存管理器为这个由四个整数构成的数组 A 分配了起始地址 201、结束地址 216 的内存块。

因为数组是作为一个连续的内存块存储的，内存管理器会传递这个内存块的起始地址。每当阿尔伯特试图访问数组中的任何一个元素时，比如说他试图访问数组中的第四个元素（他用索引 3 来访问），并试图写入某个值，阿尔伯特的应用知道在哪里写入这个特定的值，因为它知道基地址，即数组 A 的起始地址。从基地址出发，利用这里的索引 3，就可以计算出 `a[3]` 的地址。因此，它知道数字 3 对应的地址是 213(=201+3x4)。所以，要访问数组中的任何元素，应用程序只需要恒定的时间。这是数组的一个非常棒的特性：无论数组的大小如何，应用程序都可以在恒定的时间内访问数组中的任何元素。

现在假设阿尔伯特用这个包含四个整数的数组来存储他的列表。那么我会在这些位置上填入一些值。假设这里是八，这里是二，这里是六，这里是五，这里是四。(`A=[6,5,4,2], x=8`)

阿尔伯特在某个时刻觉得，好吧，我需要在这个列表里再加一个元素。他已经声明了一个大小为四的数组，现在想在数组里添加第五个元素。于是他问内存管理器：嘿，我想扩展我的数组 A，有可能做到吗？我想扩展同一个内存块。而内存管理器的回应是：当我为数组分配内存时，并不预期你会要求扩展。所以我用该内存块相邻的可用内存来存储其他变量。在某些情况下，我可能会扩展同一个内存块。

但在这种情况下，我有一个元素和一个变量 `x` 紧邻你的块。所以我不能给你扩展空间。阿尔伯特问，那我还有什么选择？内存管理器说，你可以告诉我新的大小，我可以在某个新地址重新创建一个块。我们必须将之前区块中的所有元素复制到新区块中。于是阿尔伯特说，好吧，我们开始吧。但内存管理器表示，你仍然需要告诉我新区块的大小。

阿尔伯特认为这次他会为新数组或新块分配一个非常大的尺寸，这样就不会填满这个新块。起始地址 224 被分配。阿尔伯特要求内存管理器释放前一个块。而这需要付出一定的代价，他必须将所有元素、所有数字从之前的区块复制到新的区块中。现在他可以在这个列表里再添加一个元素 (3) 了。这次他把数组设置得更大一些，以防万一列表里需要更多的数字。所以阿尔伯特唯一的选择就是把 A 作为一个全新的区块、一个全新的数组来创建。

阿尔伯特仍然感到困扰，因为如果列表太小，他就没有充分利用数组的一部分，导致内存浪费。而如果列表再次增长得太多，他又不得不创建一个新的数组、一个新的块，并且必须将所有元素从之前的块复制到新的块中。阿尔伯特正在拼命寻找解决这个问题的办法。而这个问题的解决方案是一种名为 *链表* 的数据结构。现在，让我们试着理解链表数据结构，看看它是如何解决阿尔伯特的问题的。

阿尔伯特可以做到的是，他不向内存管理器请求一个数组（那将是一大块连续的内存），而是可以每次为一个元素单独请求一个数据单元的内存。我正在这里清理内存。再举个例子，假设阿尔伯特想在内存中存储这四个整数 (6,5,4,2) 的列表。

如果他每次只请求一个整数的内存会怎样。首先，他会向内存管理器请求一些内存来存储数字 6，内存管理器会说好的，你需要空间来存储一个整数。于是你得到了地址 204 处的四字节块。所以 Albert 可以把数字 6 存储在这里。现在 Albert 又发出另一个请求，这次是单独请求数字 5，假设他得到的这个区块起始地址是 217。对于数字 5，由于他是单独发出的请求，他可能得到也可能得不到与数字 6 相邻的内存空间，更大的可能性是他不会获得相邻的内存位置。同样地，阿尔伯特分别请求数字 4 和 2 的内存。假设他分别在地址 232 和 242 处获得了这两个内存块，分别对应数字 4 和 2。

因此，你可以看到，当阿尔伯特为每个整数分别请求内存时，他得到的不是一块连续的内存区域，而是这些分散的、不连续的内存块。因此我们需要在这里存储更多信息，需要记录这是列表中的第一个元素，而这是列表中的第二个元素。所以我们需要以某种方式将这些区块链接起来。

对于数组来说，情况非常简单，我们有一块连续的内存空间。因此，我们可以通过计算元素的地址来定位特定元素，这个计算是基于内存块的起始地址和元素在数组中的位置进行的。但在这里，我们需要存储的信息是：这是第一个存储第一个元素的块，这是第二个存储第二个元素的块，以此类推。

将这些块链接在一起，并存储这是列表中的第一个块，这是列表中的第二个块的信息，我们可以做的是在每个块中存储一些额外的信息。那么，如果我们可以在每个块中有两个部分，就像这样。在块的一部分中，我们存储数据或值，在块的另一部分中，我们存储下一个块的地址。

在这个例子中，第一个块的地址部分将是 217，即存储数字 5 的下一个块的地址。而在下一个块（即第二个块）中，地址部分将是 232。在地址为 232 的块中，我们将存储地址 242，即存储数字 2 的下一个块的地址。地址为 242 的块是最后一个块，之后没有其他块。因此，在地址部分，我们可以用 0表示无效地址，0 可以用来标记这是列表的末尾，在这个特定块之后没有下一个节点或块的链接。

因此，阿尔伯特现在实际上需要向内存管理器申请一块内存，用来存储两个变量：一个是整数变量，用于存储元素的值；另一个是指针变量，用于存储下一个内存块或列表中下一个节点的地址。在 C 语言中，他可以定义一个名为 `node` 的类型，这个类型将包含两个字段：一个用于存储数据，这个字段是整数类型；另一个字段用于存储列表中下一个节点的地址。所以阿尔伯特会申请一个 `node`，他会向内存管理器申请一个 `node` 的内存空间。

```c
struct Node
{
	int data;             // 4 bytes
	Node* next;           // 4 bytes
};
```

内存管理器会这样处理：好的，你需要一个节点，其中需要四个字节来存储整型变量，另外还需要四个字节来存储指针变量（在典型架构中，地址指针变量同样占用四个字节）。因此，内存管理器现在会给我们分配一个八字节的内存块。我们将这个内存块称为 Node。

请注意，Node 结构中的第二个字段是 Node*，它表示指向 Node 的指针。因此，这个字段仅存储链表中下一个节点的地址。如果我们像这样在内存中存储链表，将这些不连续的节点相互连接起来，那么这就是一个链表数据结构。

链表数据结构的逻辑视图可以这样描述：数据存储在这些节点中，每个节点既存储数据，又存储指向下一个节点的链接。因此，每个节点都指向下一个节点。第一个节点也称为头节点。我们始终保留的唯一关于链表的信息是头节点的地址，或者说第一个节点的地址。因此，头节点的地址基本上让我们能够访问整个链表。最后一个节点中的地址是 null 或 0，这意味着最后一个节点不指向任何其他节点。

现在，如果我们想要遍历链表，唯一的方法是从头节点开始，访问第一个节点，然后询问第一个节点下一个节点的地址，即下一个节点的位置，接着我们移动到下一个节点并询问其下一个节点的地址。这是访问链表中元素的唯一方式。

如果我们要在链表中插入一个节点，比如我们想在链表末尾添加数字三，那么我们首先需要独立创建一个节点，它会获得一个内存地址。于是我们创建了这个值为 3 的节点。现在要做的就是正确填写地址，并妥善调整这些链接关系。因此，这个特定节点的地址将被填入值为 2 的节点中。而这个节点的地址部分可以为空，因为它是最后一个节点，不再指向其他任何节点。

让我们也在内存中展示这些节点。我已经在每个节点的顶部用棕色写下了它们的地址。同时，我还填写了每个节点的地址字段。假设值为 3 的节点地址为 252。这就是内存中的情况。而从逻辑上看，链表总是由第一个节点的地址来标识。与数组不同，我们无法在常数时间内访问任何元素。对于数组来说，利用内存块的起始地址和元素在列表中的位置，我们可以计算出元素的地址。但在这种情况下，我们必须从头开始。我们必须向当前元素询问下一个元素的位置，然后继续询问下一个元素“你的后继是谁”，这就像玩寻宝游戏一样——你找到第一个人，他给你第二个人的地址；找到第二个人后，他又会给你第三个人的地址。因此访问元素所需的时间将与链表长度成正比。

假设列表的大小为 n，即列表中有 n 个元素。在最坏的情况下，要遍历到最后一个元素，你需要遍历所有元素。因此，访问元素所需的时间与 n 成正比，换句话说，我们可以说这个操作的时间复杂度是大 $O(n)$。

至于列表的插入操作，我们可以在列表的任何位置插入元素，首先需要创建一个节点，然后正确调整这些链接。比如我想在列表的第三个位置插入 10。那么我们只需要创建一个节点，在数据部分存储值10，就像这样，假设我们在地址 310 处得到了节点 10。然后我们会调整第二个节点的地址字段，使其指向这个地址为 310 的节点。而这个节点将指向值为 4 的节点。现在要插入元素，我们也需要遍历链表并到达特定位置。因此，从时间复杂度来看，这又是 $O(n)$。唯一不同的是，插入操作会很简单，我们不需要像在数组中那样进行所有元素的位移——在数组中插入元素时，必须将所有元素向更高索引的位置移动一位。

同样地，从这个列表中删除某些内容也会很容易。因此，我们可以看到链表的一些优点，它不会额外占用内存。从某种意义上说，虽然有些内存未被使用，但我们确实使用了一些额外的内存来存储地址。但我们有一个优势，那就是我们可以根据需要随时创建节点。我们也可以随时释放节点，而不必像数组那样事先猜测列表的大小。

接下来，我们将讨论链表的所有操作及其成本，并与数组进行比较。在接下来的课程中，我们还将用 C 或 C++ 实现链表。以上就是关于链表的基本介绍。感谢观看。

## 四、数组 vs 链表

在上一课中，我们向大家介绍了链表数据结构。我们看到了链表如何解决数组存在的一些问题。那么现在很明显的问题是，数组和链表哪个更好？

其实，并没有哪一种数据结构比另一种更好的说法。一种数据结构可能非常适用于某种需求，而另一种数据结构则可能非常适用于另一种需求。因此，这完全取决于诸如您希望使用数据结构执行的最频繁操作是什么，或者数据的大小等因素。当然，还可能有其他因素需要考虑。

因此，在本节课中，我们将基于一些参数来比较这两种数据结构，这些参数与我们对这些数据结构进行操作的成本有关。总的来说，我们将比较研究它们的优缺点，并尝试理解在哪种情况下应该使用数组，在哪种情况下应该使用链表。因此，我将在这里画两列，一列用于数组，另一列用于链表。

我们要讨论的 *第一个参数是访问元素的成本*，无论数组的大小如何，访问数组中的元素都需要恒定的时间。这是因为数组是作为一个连续的内存块存储的。因此，如果我们知道这个内存块的起始地址或基地址，假设我们这里有一个整数数组，基地址是 200。该块的第一个字节位于地址 200。那么假设我们要计算索引 $i$ 处元素的地址，它等于 $200 + i \times 4$ (整数的字节大小)。通常，整数的字节大小为 4 字节。因此，地址将是 $200 + i \times 4$。如果零号元素的地址是 200，那么要计算索引为 6 的元素的地址，就是 $200 + 6 \times 4=224$。所以，在数组中知道任何元素的地址只需要这个计算，对于我们的大 O 符号应用来说，常数时间也被称为 $O(1)$。访问数组中的元素在时间复杂度上是 $O(1)$。如果你不了解大 O 表示法，可以查看本视频描述中关于时间复杂度分析的教程。

而在链表中，数据并不是存储在连续的内存块中。如果我们有一个链表，就像这样，假设这里有一个整数链表，那么我们在不同的地址上有多个内存块。链表中的每个块称为一个节点。每个节点有两个字段，一个用于存储数据，另一个用于存储下一个节点的地址。所以我们把第二个字段称为链接字段，关于链表我们唯一保留的信息就是第一个节点的地址，我们也称之为头节点。这就是我们传递给所有函数的内容——头节点的地址，以便访问链表中的某个元素或第一个节点。在第一个节点处，我们需要查看第二个节点的地址，然后我们转到第二个节点，查看第三个节点的地址。

在最坏的情况下，访问链表中的最后一个元素时，我们需要遍历链表中的所有节点。而在平均情况下，我们可能需要访问中间的元素。因此，如果 $n$ 是链表的大小，即链表中的元素数量，那么我们将会遍历 $n/2$ 个元素。因此，平均情况下所花费的时间也与链表中的元素数量成正比。所以我们可以说，平均情况下的时间复杂度是大 $O(n)$。

就访问元素这一参数而言，数组的表现远优于链表。因此，如果你有一个需要频繁访问列表中元素的需求，那么数组无疑是更好的选择。

现在我们要讨论的*第二个参数是内存需求或内存使用情况*。对于数组来说，我们需要在创建之前知道数组的大小，因为数组是作为一个连续的内存块创建的。所以数组的大小是固定的。我们通常的做法是创建一个足够大的数组，数组的一部分用来存储我们的列表，另一部分则保持空缺或为空，以便我们可以在列表中添加更多元素。例如，这里有一个包含七个整数的数组，而列表中只有三个整数。其余四个位置未被使用，那里会有一些垃圾值。

对于链表来说，假设我们有一个整数链表，没有未使用的内存，我们一次只为一个节点申请内存。因此，我们不保留任何预留空间。但我们为指针变量使用了额外的内存。而且，链表中指针变量所需的额外内存不容忽视。在典型的架构中，假设整数占用四个字节，指针变量同样占用四个字节。

所以如果你看到这个包含七个整数的数组的内存需求是 28 字节。而这个链表的内存需求将是 8 乘以3，其中 8 是每个节点的大小，4 字节用于整数，4 字节用于指针变量。所以这也是 24 字节。如果我们在数组中向列表添加一个元素，我们只需要多使用一个位置。而在链表中，我们将创建一个新节点，这将再占用 8 字节。因此，这将达到 32 字节。

链表能让我们充分利用数据的优势，数据部分的体积较大。因此在这种情况下，我们使用了一个整数链表。而整数只占四个字节。如果我们有一个链表，其中数据部分是某种占用 16 字节的复杂类型。那么每个节点将有 4 字节用于链接和 16 字节用于数据，总共 20 字节。一个包含 7 个元素的数组，每个元素占用 16 字节数据，总共将是 112 字节。而一个包含 4 个节点的链表将只有 80 字节。所以这完全取决于列表的数据部分是否占用大量内存，链表肯定会消耗更少的内存。否则，这取决于我们选择什么策略来决定数组的大小以及在任何时候我们保留多少未使用的数组空间。

现在，关于内存分配还有一点需要注意，因为数组是作为一个连续的内存块创建的，有时当我们想要创建一个非常大的数组时，可能没有足够大的连续内存块可用。但如果使用链表，内存可能以多个小块的形式可用。因此，我们会遇到内存碎片的问题，有时我们可能获得许多小内存单元，但无法获得一个大内存块。这种现象可能很少见。但这是一种可能性。因此，这也是链表得分的地方。因为数组有固定的大小，一旦数组被填满，我们需要更多的内存，那么除了创建一个更大的新数组并将内容从旧数组复制到新数组之外，别无选择。因此，这也是链表所不具备的一个成本。所以，当我们需要根据自身需求决定使用哪种数据结构时，必须牢记这些约束条件和要求。

现在，我们要讨论的 *第三个参数是在列表中插入元素的成本*。请注意，当我们在这里讨论数组时，也包括将数组用作动态列表的可能情况。插入操作可能存在三种情况。

第一种情况是当我们需要在列表的开头插入一个元素时。假设我们要在列表的开头插入数字 3。对于数组来说，我们必须将每个元素向更高的索引位置移动一位。因此，所需时间将与列表的大小成正比。所以这将是 $O(n)$，假设 $n$ 是列表的大小，就时间复杂度而言，这将是 $O(n)$。在链表的情况下，在开头插入一个元素仅意味着创建一个新节点并调整头指针和这个新节点的链接。因此，所花费的时间不会取决于列表的大小，它将是恒定的。因此，对于链表来说，在开头插入一个元素的时间复杂度是 $O(1)$。

在列表末尾插入一个元素，对于数组（假设我们讨论的是动态数组，即当数组填满时会创建一个新数组的动态列表），如果在数组中有空间，我们只需写入列表的下一个更高索引处，所以这将是常数时间。所以如果数组未满，时间复杂度就结束了。如果数组已满，我们将不得不创建一个新数组，并将所有之前的内容复制到新数组中，这将花费 $O(n)$ 的时间，其中 $n$ 是列表的大小。在链表的情况下，在末尾添加一个元素意味着遍历整个链表，然后创建一个新节点并调整链接。因此，所需时间将与 $n$ 成正比，我将用这种颜色编码来表示链表。这里的 $n$ 是列表中元素的数量。

第三种情况是当我们想在列表中间的第 $n$ 个位置或第 $i$ 个位置插入元素时。同样，对于数组的情况，我们将不得不移动元素。在平均情况下，我们可能希望在数组的中间位置插入元素。因此，我们将不得不移动 $n/2$ 个元素，其中 $n$ 是列表中元素的数量。因此，在平均情况下，所需时间肯定与 $n$ 成正比。所以时间复杂度将是 $O(n)$。对于链表，我们也需要遍历到那个位置，然后才能调整链接。尽管我们不需要进行任何移位操作，但仍需遍历到那个点，在平均情况下，所需时间与 $n$ 成正比，时间复杂度将是 $O(n)$。

可以看到，删除一个元素也会有这三种情况，而删除操作在这三种情况下的时间复杂度也是相同的。

最后我要讨论的*最后一个参数是哪个更容易使用和实现*。数组绝对比链表更容易使用，特别是在 C 或 C++ 中，链表的实现更容易出现诸如段错误和内存泄漏等错误。使用链表需要格外小心。这就是数组与链表的对比。在下一课中，我们将在 C 或 C++ 中实现链表，亲自动手编写一些实际代码。这节课就到这里。感谢观看。

## 五、C/C++ 实现链表

在我们之前的课程中，我们描述了链表，看到了链表中各种操作的代价，并将链表与数组进行了比较。现在让我们来实现链表。在 C 和 C++ 中的实现将非常相似，我们会讨论一些细微的差异。本课程的先决条件是您应该对 C 和 C++ 中的指针有实际了解。你还应该了解动态内存分配的概念。如果想复习这些概念，可以查看本视频描述中的附加资源。

好了，我们开始吧。我们知道，在链表中，数据存储在多个不连续的内存块中。我们将每个内存块称为链表中的一个节点。那么，我先在这里画一个链表。我们有一个包含三个节点的整数链表。如我们所知，每个节点有两个字段或两部分：一个用于存储数据，另一个用于存储下一个节点的地址，我们也可以称之为链接到下一个节点。

假设第一个节点的地址是 200，第二个节点的地址是 100，第三个节点的地址是 300。对于这个链表来说，这只是链表的逻辑视图。因此，第一个节点的地址部分将是第二个节点的地址 100，而这里我们会得到 300。最后一个节点的地址部分将是 null，这只是地址零的同义词或宏定义，零是一个无效地址。一个等于零或 null 的指针变量，其地址为零或 null，意味着该指针变量没有指向一个有效的内存位置。分配给每个节点的内存块的地址是完全随机的，没有任何关系，不能保证地址会按递增顺序、递减顺序排列，或者彼此相邻。这就是为什么我们需要保留这些链接。

现在，我们始终随身携带的链表标识是第一个节点的地址，也就是我们所说的头节点。因此，我们保留另一个变量，它将是节点指针类型，这个变量将存储第一个节点的地址。我们可以随意命名这个指针变量，假设这个指针变量名为 A，这个指向头节点或第一个节点的特定指针变量的名称也可以被理解为链表的名称。因为这是我们始终持有的链表的唯一标识。

现在让我们看看如何将这个逻辑视图映射到 C 或 C++ 的实际程序中。在我们的程序中，节点将作为一种数据类型，它包含两个字段：一个用于存储数据，另一个用于存储地址。在 C 语言中，我们可以将这种数据类型定义为结构体。因此，假设我们定义了一个名为 Node 的结构体，它包含两个字段。

存储数据的第一个字段，此处数据类型为整数。因此，这将是一个整数链表的节点。如果我们想要一个双精度链表的节点，此处的数据类型将是双精度型。第二个字段将是指向节点结构体 Node* 的指针，我们可以将其命名为 link 或 next 之类的名称。这是 C 语言风格声明 Node* 或指向节点的指针的方式。如果是 C++，我们可以直接写成 Node*，我个人更喜欢用 C++ 这种写法，看起来更顺眼。

```c
struct Node
{
	int data;
	struct Node* link;
};

Node* A;
A = NULL;
Node* temp = (Node*) malloc(sizeof(Node));

// (*temp).data = 2;
// (*temp).link = NULL;
temp->data = 2;
temp->link = NULL;

A = temp;
```

```c++
struct Node
{
	int data;
	Node* link;
};

Node* temp = new Node();
temp->data = 2;
temp->link = NULL;

A = temp;

temp = new Node();
temp->data = 4;
temp->link = NULL;

Node* temp1 = A;
while(temp1->link != NULL){
	temp1 = temp1->link;
}


```

在我们的逻辑视图中，变量 A 的类型是 Node* 或指向 Node 的指针。这三个带有两个字段的矩形都是 Node 类型。而 Node 中的这个字段，第一个字段是整数类型。第二个字段是指向节点的指针类型，或者说 Node*，重要的是在逻辑视图中知道哪个是什么，我们应该在实现链表之前先有这个可视化概念。

好的，现在让我们通过代码来创建这个特定的整数链表。为了实现这一点，我们需要执行两个操作：一个是在链表中插入节点，另一个是遍历链表。

但在那之前，我们首先要做的是声明一个指向头节点的指针，即一个存储头节点地址的变量。为了清晰起见，我会在这里写上“头节点”。因此，我声明了一个名为 A 的节点指针。最初，当链表为空时，即链表中没有任何元素时，这个指针应该不指向任何地方。所以我们写一个类似于 `A =NULL;` 的语句来表达与这两个语句相同的意思，我们所做的是创建了一个名为 A 的指向节点的指针，而这个指针没有指向任何地方。因此，这个列表是空的。

现在假设我们想在这个列表中插入一个节点。所以我们首先创建一个节点，创建节点只不过是在 C 语言中创建一个内存块来存储节点，我们使用 malloc 函数来创建内存块。作为参数，我们传入我们想要的内存块字节数 `malloc(sizeof(Node))`。也就是说，我们需要一个与节点大小相等的内存块。因此，调用 malloc 将创建一个内存块。这是一块在运行时动态分配的内存。处理这类内存的唯一方法是通过指针引用该内存位置。假设这里分配的内存块位于地址 200。现在，malloc 返回一个 void 指针，该指针为我们提供了分配的内存块的地址。因此，我们需要将其收集到某个变量中。

假设我们创建一个名为 temp 的变量，它指向节点。这样我们就可以将 malloc 返回的地址收集到这个特定变量中，这里需要进行类型转换，因为 malloc 返回的是 void 指针，而我们将 temp 作为指向节点的指针。现在，我们已经在内存中创建了一个节点。

现在我们需要做的是在这个特定节点中填入数据并调整链接，这意味着要在变量 A 和新创建节点的链接字段中写入正确的地址。为此，我们将不得不解引用这个刚刚创建的指针变量。我们知道，如果在指针变量前加上星号，就意味着解引用它以修改该特定地址的值。现在在这个例子中，我们有一个包含两个字段的节点。

因此，一旦我们解引用，如果想访问每个字段，我们需要在这里使用类似 `.data` 的方式来访问数据，并使用 `.link` 来写入链接字段。所以我们会写这样的语句来在这里填入值二。现在我们有这个临时变量指向这个。而这个新创建的节点的链接部分应该是空的，因为这是第一个也是最后一个节点。最后我们需要做的是在 A 中写入这个新创建节点的地址。所以我们会写类似 `A = temp;` 这样的语句。

好的，temp 是用来临时存储节点地址的，直到我们正确修复所有链接为止。现在我们可以将 temp 用于其他用途，我们的链表是完整的。现在它有一个节点。

我们在这里写的这两行代码用于解引用并将值写入新节点，其实还有另一种语法。与其写成类似 `(*temp).data` 这样的形式，我们也可以写成 `temp->data`，我们需要用两个字符来组成这个箭头：一个连字符和一个右尖括号（右尖括号）。所以我们可以这样写。

下面同样地，我们可以这样写来在 C++ 中创建一个内存块。我们可以使用 `malloc`，也可以使用 `new` 操作符。所以在 C++ 中，这变得非常简单，我们可以简单地写成 `Node* temp = new Node()`，就像这样。我们指的是同一件事。这样更清晰明了。而且，new 操作符总是比 malloc 更受推荐。

因此，如果你在使用 C++，建议使用 `new`。到目前为止，在我们的程序中，我们通过创建这个指向头节点的指针并初始赋值为 NULL 来创建一个空列表，然后我们创建了一个节点并将这个第一个节点添加到这个链表中。当列表为空时，我们想要插入一个节点，逻辑相当简单。

当列表不为空时，我们可能想在列表的开头插入一个节点。或者我们可能想在列表的末尾插入一个节点。甚至可能想在列表的中间某个位置插入一个节点。在某个特定位置，我们将为这些不同类型的插入操作编写独立的函数和例程。我们将在编译器中看到运行代码。

在接下来的课程中，我们只讨论这里的逻辑。在我现在这段毫无结构的代码中。所以我想写一段代码，每次在链表末尾插入两个节点，实际上我们想创建一个包含三个节点的链表，节点的值分别为二、四和六。这是我们一开始的初始示例。

好的，那么让我们在链表中再添加两个值为四和六的节点。在代码的这个阶段，我们已经有一个变量 temp 指向这个特定的节点，我们将创建一个新节点并使用相同的变量名来收集这个新节点的地址。所以我们会写这样的语句。因此，创建了一个新节点，temp 现在存储了这个新节点的地址，该地址位于 100。在这里，我们可以再次设置数据。然后，因为这将作为最后一个节点，我们需要将链接设置为 NULL。

现在我们需要做的就是构建最后一个节点的链接。为此，我们必须遍历链表并到达链表的末尾。为此，我们可以这样写代码：创建一个新的变量 temp1，它将被指向节点，最初我们可以让它指向头节点，通过这样的语句将这个变量指向头节点，我们可以这样写一个循环。

```c++
Node* temp1 = A;
while(temp1->link != NULL){
	temp1 = temp1->link;
	Print "temp->data";     // 伪代码
}
temp1->link = temp;
```

现在这是到达链表末尾的通用逻辑。如果我们仅以这个例子中的单个节点来看这个逻辑，可能不太清晰，让我们画一个包含多个节点的链表。因此，我们将临时指针 temp1 指向这里的第一个节点。如果该节点的链接部分为空，则表示我们已到达最后一个节点；否则，我们可以移动到下一个节点。因此，`temp1 = temp1->link` 将带我们到下一个节点。我们将继续这个过程，直到到达最后一个节点。

对于最后一个节点，这个特定条件 `temp->link != NULL` 将为假，因为链接部分将为 NULL，我们将退出这个 while 循环。这就是我们遍历链表直到末尾的代码逻辑。如果我们想打印链表中的元素，我们会这样写，并在这个 while 循环内部写入 `print(temp->data)`。

但现在我们想在链表的末尾插入数据，而我们只是遍历链表以到达最后一个节点。还有一点我想指出的是，我们使用的是这个临时变量 `temp1`，最初将地址存储在 A 中，而不是像 `A=A->link` 这样操作，并使用变量 A 本身来遍历链表。因为如果我们修改 A，就会丢失头节点的地址。

因此，变量 A（存储头节点地址的变量）永远不会被修改，只有这些临时变量会被修改以遍历链表。最终，在所有操作完成后，我们会写一个类似 `temp1->link = temp` 的语句——此时 `temp1` 正指向此处。于是现在这个地址部分就被更新了。

这个链接已经建立。所以现在列表中有两个节点。再次强调，当我们要在列表中插入数字为 6 的节点时，必须按照这个逻辑创建一个新节点，然后按照这个逻辑遍历列表。因此，我们首先将临时指针 `temp1` 指向这里，然后循环会将 `temp1` 移动到末尾。假设这个新块的地址是 300。那么最后这一行代码将调整地址 100 处节点的链接。

要在链表末尾插入一个节点，这里有两种逻辑：前四行处理链表为空的情况，其余行处理链表非空的情况。理想情况下，我们会将这些逻辑封装成一个函数——这将在后续课程中实现。我们将分别实现以下方法：打印链表中所有节点的方法、在链表末尾插入节点的方法、在链表开头插入节点的方法，以及在指定位置插入节点的方法。本节课内容就到这里，感谢观看。


In our previous lesson, we saw how we can map the logical view of a linked list into a C or C plus plus program, we saw how we can implement two basic operations, one traversal of the linked list and another inserting a node at the end of the linked list.

In this lesson, we will see a running code that will insert a node at the beginning of the linked list. So let's get started. I will write a C program here.

The first thing that we want to do in our program is that we want to define a node, a node will be a structure in C, it will have two fields, one to store the data, let's say we want to create a linked list of integers. So our data type will be integer. If we wanted to create a linked list of characters, then our type would be character here.So we will have another field that will be pointed to node that will store the address of the next node, we can name this variable link, or some people, some people also like to name this variable next, because it sounds more intuitive, this variable will store the address of the next node in the linked list. In C, whenever we have to declare node or pointer to node, we will have to write struct node or struct node star. In c++, we will have to write only node star.

And that's one difference. Okay, so this is the definition of our node. Now to create a linked list, we will have to create a variable that will be pointed to node and that will store the address of the first node in the linked list, what we also call the head node.

So I will create a pointer to node here, struct node star, we can name this variable whatever often for the sake of understanding we name this variable head. Now I have declared this variable as a global variable, I have not declared this variable inside any function. And I'll come back to why I'm doing so.

Now I'll write the main method, this is the entry point to my program. The first thing that I want to do is I want to say head is equal to null, which will mean that this pointer variable points nowhere. So right now, the list is empty.

So far, what we have done here in our code is that we have created a global variable named head, which is of type pointer to node, and the value in this pointer variable is null. So so far, the list is empty. Now what I want to do in my program is that I want to ask the user to input some numbers.

And I want to insert all these numbers into the linked list. So I'll print something like how many numbers, let's say the user wants to input n numbers. So I'll collect this number in this variable n. And then I'll define another variable i to run the loop.

And so I'm running a loop here. If it was c++, I could declare this integer variable right here inside the loop. Now I'll write a print statement like this.

And I'll define another variable x and each time I'll take this variable x as input from the user. And now I will insert this particular number x, this particular integer x into the linked list by making a call to the method insert. And then each time we insert we will print all the nodes in the linked list the value of all the nodes in the list.

By calling a function named print, there will be no argument to this function print. Of course, we need to implement these two functions insert and print. Let me first write down the definition of these two functions.

So let us implement these two functions insert and print. Let us first implement the function insert that will insert a node at the beginning of the linked list. Now in the insert function, what we need to do is we first need to create a node in C, we can create a node using malloc function, we have talked about this earlier, malloc returns a pointer to the starting address of the memory block, we are having to type cast here because malloc returns a wide pointer and we need a pointer to node a variable that is pointed to node.

And then only if we dereference we dereference using an asterisk sign, then we will be able to access the fields of the node. So the data part will be x and we have an alternate syntax for this particular syntax, we could simply write something like temp and this arrow and it will mean the same thing. And this is more common with these two lines in the insert function, all we are doing is we are creating a node, let's say we get this node and let's assume that the address that we get for this node is 100.

Now there is a variable temp where we are storing the address, we can do one thing whenever we create a node, we can set data to whatever we want to set and we can set the link field initially to null. And if needed, we can modify the link field. So I'll write one more statement temp dot next is equal to null.

Remember, temp is a pointer variable here and we are dereferencing the pointer variable to modify the value at this particular node. temp will also take some space in the memory. That's why I have shown this rectangular block for both the pointer variables head and temp.

And node has two parts, one for the pointer variables and one for the data. So this part, the link part is now we can either write null here or we can write it like this, it's the same thing. Logically, it means the same.

Now if we want to insert this node in the beginning of the list, there can be two scenarios. One when the list is empty, like in this case, so the only thing that we need to do is we need to point head to this particular node instead of pointing to null. So I will write a statement like head is equal to temp.

And the value in head now will be address 100. And that's what we mean when we say a pointer variable points to a particular node, we store the address of that node. So this is our linked list.

After we insert the first node, let us now see what we can do to insert a node at the beginning if the list is not empty, like what we have right now. Once again, we can create a node, fill in the value x here that is passed as argument. Initially, we may set the link field as null.

And let's say this node gets address 150 in the memory. And we have this variable temp through which we are referencing this particular memory block. Now unlike the previous case, if we just set head is equal to temp, this is not good enough, because we also need to build this link, we need to set the next or the link of the newly created node to whatever the previous head was.

So what we can do is we can write something like if head is not equal to null, or if the list is not empty, first set temp dot next equal head. So we first build this link, the address here will be 100. And then we say head equal temp.

So we cut this link and point head to this newly created node. And this is our modified linked list after insertion of this second node at the beginning of the list. Now one final thing here, this particular line, the third line temp dot next equal null, this is getting used only when the list is empty.

If you see when the list is empty head is already null. So we can avoid writing two statements, we can simply write this one statement m dot next equal head. And this will also cover the scenario when the list is empty.

Now the only thing remaining in this program to get this running is the implementation of this print function. So let us implement this print function. Now what I will do here is I'll create a local variable which is pointer to node named temp.

And I need to write struct node here, I keep missing this in C unit, you need to write it like this. And I want to set this as address of the head node. So this global variable has the address of the head node.

Now I want to traverse the linked list. So I will write a loop like this. While temp is not equal to null, I'll keep going to the next node using this statement temp is equal to temp dot next.

And at each stage, I'll print the value in that node as temp dot data. Now I'll write two more print one outside this while loop and one outside after this while loop to print an end of line. Now why did we use a temporary variable because we do not want to modify head because we will lose the reference of the first node.

So first we collect the address in head in another temporary variable. And we are modifying the addresses in this temporary variable using temp is equal to temp dot next to traverse the list. Let us now run this program and see what happens.

So this is asking how many numbers you want to insert in the list. Let's say we want to insert five numbers. Initially, the list is empty.

Let's say the first number that we want to insert is two. At each stage, we are printing the list. So the list is now to the first element and the last element is two, we will insert another number, the list is now five to five is inserted at the beginning of the list.

Again, we inserted eight and eight is also inserted at the beginning of the list. Okay, let's insert number one, the list is now 1852. And finally, I inserted number 10.

So the final list is 101852. This seems to be working. Now, if we were writing this code in c++, we could have done a couple of things, we could have written a class and organize the code in an object oriented manner, we could also have used a new operator in place of the malloc function.

And now coming back to the that we have declared this head as global variable, what if this was not a global variable, this was declared inside this main function as a local variable. So I'll remove this global declaration. Now this head will not be accessible in other functions.

So we need to pass address of the first node as argument to other functions to both these functions print and insert. So to this print method, we will pass, let's say we name this argument as head, we can name this argument argument as head or a or temp or whatever. If we name this argument as head, this head in print will be a local variable of print and will not be this head in main, these two heads will be different, these two variables will be different.

When the main function calls print passing its head, then the value in this particular head in the main function is copied to this another head in the print function. So now in the print function, we may not use this temp variable, what we can do is we can use this variable head itself to traverse the list. And this should be good, we are not modifying this head here in the main.

Similarly to the insert function, we will have to pass the address of the first node. And this head again is just a copy, this is again a local variable. So after we modify the linked list, the head in main method should also be modified.

There are two ways to do it one we can pass the pointer to node as return from this method. So in the main method, insert function will take another argument head and we will have to collect the return into head again so that it is modified. Now this code will work fine.

Oops, I forgot to write a return here return head. And we can run this program like before, we can give all the values and see that the list is building up correctly. There was another way of doing this instead of asking this insert function to return the address of head, we could have passed this particular variable head by reference.So we could have passed insert ampersand head head is already a pointer to node. So in the insert function, we will have to receive pointer to pointer node star star. And to avoid confusion, let's name this variable something else this time, let's name this pointer to head.

So to get head, we will have to write something like we will have to dereference this particular variable and write asterisk pointer to head everywhere. And the return type will be void. Sometimes we want to vary a name this variable as head this local variable as head doesn't matter, but we will have to take care of using it properly.

Now this code will also work. As you can see here, we can insert nodes. And this seems to be going well.

If you do not understand these concepts of scope, you can refer to the description of this video for additional resources. So this was inserting a node at the beginning of the linked list. Thanks for watching.In our previous lesson, we had written code to insert a node at the beginning of the linked list. Now in this lesson, we will write program to insert a node at any given position in the linked list. So let me first explain the problem in a logical view.

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