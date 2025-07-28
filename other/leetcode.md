

## 一、数组

其他数据结构相比，它主要具有以下优势。  
* 随机访问：由于我们拥有基地址，并且每个项目或引用的大小相同，因此可以在 O(1) 时间内访问第 $i$ 个项目。  
* 缓存友好性：由于项目/引用存储在连续的位置，因此我们获得了引用局部性的优势。
- 在进行诸如在中间插入、从中间删除和在未排序的数据中搜索等操作的地方，它没有优势。
- 它是一种基本的线性数据结构，我们可以使用它构建其他数据结构，如堆栈队列、双端队列、图、哈希表等。

### 1.1 数组的基本操作

在 Python 中，通常使用 list 作为数组演示，但需要注意的是 list 实际上并不是 array，它存储的也不是变量的值，而是变量的指针（引用）。

基本操作：

* 遍历：正序遍历、逆序遍历
* 增加或插入，
	* 确定位置，在哪插
	* 移动元素，从最后位置开始依次向后移动一个位置
	* 插入新元素
* 删除：
	* 确定位置
	* 移动元素，从该位置开始，分别用后面的值覆盖掉当前值
* 改、查

通过其索引来标识。数组索引从 0 开始。
- 数组元素：元素是存储在数组中的项目，可以通过其索引访问。
- 数组长度：数组的长度由它可以包含的元素数量决定。

**数组的内存表示**

在数组中，所有元素都存储在连续的内存位置中。因此，当我们初始化一个数组时，元素会在内存中按顺序分配。这使得对元素的访问和操作非常高效。


### 1.2 数组基本问题

* 数组中的领导者（详见 leaders_in_an_array.py）
* 删除排序后的重复项 （详见 26.py） -- 快慢指针
* 生成所有子数组（详见 78.py）-- 回溯法等
* 反转数组（详见 344.py）-- 对撞指针
- [旋转数组](https://www.geeksforgeeks.org/dsa/complete-guide-on-array-rotations/)
- [以零结尾](https://www.geeksforgeeks.org/dsa/move-zeroes-end-array/)
- [最小增量以使其相等](https://www.geeksforgeeks.org/dsa/minimum-increment-k-operations-make-elements-equal/)[](https://www.geeksforgeeks.org/dsa/equilibrium-index-of-an-array/)
- [尺寸 1 的最低生产成本](https://www.geeksforgeeks.org/dsa/minimum-cost-make-array-size-1-removing-larger-pairs/)

## 简单问题

- [在 K 距离内复制](https://www.geeksforgeeks.org/dsa/check-given-array-contains-duplicate-elements-within-k-distance/)
- [使位置更大](https://www.geeksforgeeks.org/dsa/rearrange-array-such-that-even-positioned-are-greater-than-odd/)
- [所有子数组的总和](https://www.geeksforgeeks.org/dsa/sum-of-all-subarrays/)
- [股票买卖 – 多笔交易](https://www.geeksforgeeks.org/dsa/stock-buy-sell/)
- [双打中的单打](https://www.geeksforgeeks.org/dsa/find-element-appears-array-every-element-appears-twice/)
- [缺失号码](https://www.geeksforgeeks.org/dsa/find-the-missing-number/)
- [缺失和重复](https://www.geeksforgeeks.org/dsa/find-a-repeating-and-a-missing-number/)
- [仅从 1 重复到 n-1](https://www.geeksforgeeks.org/dsa/find-repetitive-element-1-n-1/)
- [大小为 3 的排序子序列](https://www.geeksforgeeks.org/dsa/find-a-sorted-subsequence-of-size-3-in-linear-time/)
- [最大子数组和](https://www.geeksforgeeks.org/dsa/largest-sum-contiguous-subarray/)
- [平衡指数](https://www.geeksforgeeks.org/dsa/equilibrium-index-of-an-array/)
- [两数之和 - 判断是否存在一对](https://www.geeksforgeeks.org/dsa/check-if-pair-with-given-sum-exists-in-array/)
- [两数之和 - 最接近的一对](https://www.geeksforgeeks.org/dsa/two-elements-whose-sum-is-closest-to-zero/)[中等难度部分有更多关于两数之和的题目]
- [将数组拆分为三个相等的数组](https://www.geeksforgeeks.org/dsa/split-array-three-equal-sum-subarrays/)
- [翻转 K 次，最大连续 1](https://www.geeksforgeeks.org/dsa/find-zeroes-to-be-flipped-so-that-number-of-consecutive-1s-is-maximized/)

## 解决剩余问题的前提条件

1. [二分查找](https://www.geeksforgeeks.org/dsa/binary-search/)
2. [选择排序](https://www.geeksforgeeks.org/dsa/selection-sort-algorithm-2/)、[插入排序](https://www.geeksforgeeks.org/dsa/insertion-sort-algorithm/)、[二分查找](https://www.geeksforgeeks.org/dsa/binary-search/)、[快速排序](https://www.geeksforgeeks.org/dsa/quick-sort-algorithm/)、[归并排序](https://www.geeksforgeeks.org/dsa/merge-sort/)、[循环排序](https://www.geeksforgeeks.org/dsa/cycle-sort/)和[堆排序](https://www.geeksforgeeks.org/dsa/heap-sort/)
3. [C++ 排序](https://www.geeksforgeeks.org/cpp/sort-c-stl/) / [Java 排序](https://www.geeksforgeeks.org/java/arrays-sort-in-java/)/ [Python 排序](https://www.geeksforgeeks.org/python/sort-a-list-in-python/)/ [JavaScript 排序](https://www.geeksforgeeks.org/javascript/sort-an-array-in-javascript/)
4. [双指针技术](https://www.geeksforgeeks.org/dsa/two-pointers-technique/)
5. [前缀和技术](https://www.geeksforgeeks.org/dsa/prefix-sum-array-implementation-applications-competitive-programming/)
6. [哈希基础知识](https://www.geeksforgeeks.org/dsa/hashing-data-structure/)
7. [窗口滑动技术](https://www.geeksforgeeks.org/dsa/window-sliding-technique/)

## ****中等****问题

- [使 arr[i] = i](https://www.geeksforgeeks.org/dsa/rearrange-array-arri/)
- [最大循环子数组和](https://www.geeksforgeeks.org/dsa/maximum-contiguous-circular-sum/)
- [根据给定的索引重新排序](https://www.geeksforgeeks.org/dsa/reorder-a-array-according-to-given-indexes/)
- [除自身以外的产品](https://www.geeksforgeeks.org/dsa/a-product-array-puzzle/)
- [第 K 个最大和子数组](https://www.geeksforgeeks.org/dsa/k-th-largest-sum-contiguous-subarray/)
- [最小缺失数](https://www.geeksforgeeks.org/dsa/find-the-first-missing-number/)
- [总和大于 x 的最小子数组](https://www.geeksforgeeks.org/dsa/minimum-length-subarray-sum-greater-given-value/)
- [多数元素](https://www.geeksforgeeks.org/dsa/majority-element/)
- [计算可能的三角形](https://www.geeksforgeeks.org/dsa/find-number-of-triangles-possible/)[](https://www.geeksforgeeks.org/dsa/trapping-rain-water/)
- [给定和的子数组](https://www.geeksforgeeks.org/dsa/find-subarray-with-given-sum/)[](https://www.geeksforgeeks.org/dsa/find-rotation-count-rotated-sorted-array/)
- [0 和 1 相等的最长子数组](https://www.geeksforgeeks.org/dsa/largest-subarray-with-equal-number-of-0s-and-1s/)
- [两个二进制数组的最长公共跨度](https://www.geeksforgeeks.org/dsa/longest-span-sum-two-binary-arrays/)
- [从其对和数组构造一个数组](https://www.geeksforgeeks.org/dsa/construct-array-pair-sum-array/)
- [2 总和 - 所有对子](https://www.geeksforgeeks.org/dsa/2-sum-find-all-pairs-with-given-sum/)
- [2 个不同数对之和](https://www.geeksforgeeks.org/dsa/print-all-pairs-with-given-sum/)
- [3 总和 - 求任意数](https://www.geeksforgeeks.org/dsa/find-a-triplet-that-sum-to-a-given-value/)
- [3 总和 - 最接近的三元组](https://www.geeksforgeeks.org/dsa/find-a-triplet-in-an-array-whose-sum-is-closest-to-a-given-number/)
- [4 总和 - 找出任意数列](https://www.geeksforgeeks.org/dsa/4-sum-find-any-quadruplet-having-given-sum/) [更多 4 总和 的题目请见困难部分]

## ****难题****​

- [超越者计数](https://www.geeksforgeeks.org/dsa/find-surpasser-count-of-each-element-in-array/)
- [收集雨水](https://www.geeksforgeeks.org/dsa/trapping-rain-water/)
- [前 K 个频繁元素](https://www.geeksforgeeks.org/dsa/find-k-numbers-occurrences-given-array/)
- [排序数组中第 K 个缺失的正数](https://www.geeksforgeeks.org/dsa/k-th-missing-element-in-sorted-array/)
- [股票买卖 - 最多 K 笔交易](https://www.geeksforgeeks.org/dsa/maximum-profit-by-buying-and-selling-a-share-at-most-k-times/)
- [股票买卖 - 最多 2 笔交易](https://www.geeksforgeeks.org/dsa/maximum-profit-by-buying-and-selling-a-share-at-most-twice/)
- [溪流中的中线](https://www.geeksforgeeks.org/dsa/median-of-stream-of-integers-running-integers/)
- [3 个数组中最小差异三元组](https://www.geeksforgeeks.org/dsa/smallest-difference-triplet-from-three-arrays/)
- [最大值出现在 n 个范围内](https://www.geeksforgeeks.org/dsa/maximum-occurred-integer-n-ranges/)
- [3 总和 - 不同三元组](https://www.geeksforgeeks.org/dsa/unique-triplets-sum-given-value/)
- [3 总和 - 所有三元组](https://www.geeksforgeeks.org/dsa/find-triplets-array-whose-sum-equal-zero/)
- [4 总和 - 不同四元组](https://www.geeksforgeeks.org/dsa/find-four-elements-that-sum-to-a-given-value-set-2/)
- [4 总和 - 所有四倍数](https://www.geeksforgeeks.org/dsa/4-sum-find-a-quadruplet-with-closest-sum/)
- [4 总和 - 最接近的四元组](https://www.geeksforgeeks.org/dsa/4-sum-find-a-quadruplet-with-closest-sum/)

## ****竞技程序员的专家问题****

- [MO 算法](https://www.geeksforgeeks.org/dsa/mos-algorithm-query-square-root-decomposition-set-1-introduction/)
- [平方根（Sqrt）分解算法](https://www.geeksforgeeks.org/dsa/square-root-sqrt-decomposition-algorithm/)
- [稀疏表](https://www.geeksforgeeks.org/dsa/sparse-table/)
- [使用稀疏表进行范围总和查询](https://www.geeksforgeeks.org/dsa/range-sum-query-using-sparse-table/)
- [范围最小查询（平方根分解和稀疏表）](https://www.geeksforgeeks.org/dsa/range-minimum-query-for-static-array/)
- [范围 LCM 查询](https://www.geeksforgeeks.org/dsa/range-lcm-queries/)
- [范围顺序统计的合并排序树](https://www.geeksforgeeks.org/dsa/merge-sort-tree-for-range-order-statistics/)
- [到达终点所需的最少跳跃次数](https://www.geeksforgeeks.org/dsa/minimum-number-of-jumps-to-reach-end-of-a-given-array/)
- [使用位操作进行空间优化](https://www.geeksforgeeks.org/competitive-programming/space-optimization-using-bit-manipulations/)
- [仅允许旋转的 Sum( i*arr[i]) 的最大值](https://www.geeksforgeeks.org/dsa/find-maximum-value-of-sum-iarri-with-only-rotations-on-given-array-allowed/)

****快速链接：****

- [数组“练习题”](https://www.geeksforgeeks.org/explore?page=1&category=Arrays&sortBy=submissions&itm_source=geeksforgeeks&itm_medium=main_header&itm_campaign=practice_header)
- [热门数组面试问题](https://www.geeksforgeeks.org/dsa/top-50-array-coding-problems-for-interviews/)
- [数组“测验”](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-array-data-structure-with-answers/)

  

什么是数组

评论

更多信息

[广告合作](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[下一篇文章](https://www.geeksforgeeks.org/dsa/string-data-structure) 

[数据结构中的字符串](https://www.geeksforgeeks.org/dsa/string-data-structure)

### 类似读物

[DSA 教程 - 学习数据结构和算法

数据结构和算法 (DSA) 是一门研究如何利用数组、堆栈和树等数据结构高效地组织数据，并结合逐步的程序（或算法）来有效地解决问题的学科。数据结构管理数据的存储和访问方式，而算法则侧重于

阅读需7分钟

](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/)[数组数据结构指南

本文介绍了数组、不同流行语言的实现、其基本操作以及常见的问题/面试题。数组将元素（在 C/C++ 和 Java 原始数组中）或其引用（在 Python、JS 和 Java 非原始数组中）存储在连续的

阅读需3分钟

](https://www.geeksforgeeks.org/dsa/array-data-structure-guide/)[数据结构中的字符串

字符串是字符序列。以下事实使字符串成为一种有趣的数据结构。元素集合较小。与普通数组不同，字符串通常包含较小的元素集合。例如，小写英文字母只有 26 个字符。ASCII 只有 256 个字符。字符串是不可变的。

阅读需2分钟

](https://www.geeksforgeeks.org/dsa/string-data-structure/)[矩阵数据结构

矩阵数据结构是按行和列排列的二维数组。它通常用于表示数学矩阵，是数学、计算机图形学和数据处理等各个领域的基础。矩阵允许在结构中高效地存储和操作数据。

阅读需2分钟

](https://www.geeksforgeeks.org/dsa/matrix/)[搜索算法

搜索算法是计算机科学中用于在数据集合中定位特定项的重要工具。在本教程中，我们主要关注数组中的搜索。当我们在数组中搜索某个项时，根据输入的类型，有两种最常见的算法：

阅读需2分钟

](https://www.geeksforgeeks.org/dsa/searching-algorithms/)[排序算法

排序算法用于按顺序重新排列给定的数组或元素列表。例如，给定数组 [10, 20, 5, 2] 升序排序后变为 [2, 5, 10, 20]，降序排序后变为 [20, 10, 5, 2]。针对不同的情况，存在不同的排序算法。

阅读需3分钟

](https://www.geeksforgeeks.org/dsa/sorting-algorithms/)[数据结构中的哈希

哈希是一种用于数据结构的技术，它能够高效地存储和检索数据，并允许快速访问。哈希算法使用哈希函数将数据映射到哈希表（一个数组）中的特定索引。它能够根据键快速检索信息。

阅读需2分钟

](https://www.geeksforgeeks.org/dsa/hashing-data-structure/)[双指针技术

双指针是一种简单有效的技巧，通常用于排序数组中的两数之和、最近两数之和、三数之和、四数之和、雨水陷阱以及许多其他常见的面试题。给定一个排序数组 arr（按升序排列）和一个目标，查找是否存在一个

阅读需11分钟

](https://www.geeksforgeeks.org/dsa/two-pointers-technique/)[滑动窗口技术

滑动窗口技术是一种用于解决涉及子数组、子字符串或窗口的问题的方法。其主要思想是利用前一个窗口的结果来计算下一个窗口的结果。这种技术通常用于诸如查找具有特定和的子数组、查找

阅读需13分钟

](https://www.geeksforgeeks.org/dsa/window-sliding-technique/)[前缀和数组 - 实现

给定一个数组 arr[]，求该数组的前缀和。前缀和数组是另一个大小相同的数组 prefixSum[]，其中 prefixSum[i] 等于 arr[0] + arr[1] + arr[2] . . . arr[i]。示例：输入：arr[] = [10, 20, 10, 5, 15]输出：[10, 30, 40, 45, 60]解释：对于每个索引 i，将所有

阅读需 4 分钟

](https://www.geeksforgeeks.org/dsa/prefix-sum-array-implementation-applications-competitive-programming/)

[![geeksforgeeks 页脚徽标](https://media.geeksforgeeks.org/auth-dashboard-uploads/gfgFooterLogo.png)](https://www.geeksforgeeks.org/)

公司及通讯地址：

北方邦诺伊达 136 区 Sovereign Corporate Tower 7 楼 A-143 室（201305）

注册地址：

K 061, Tower K, Gulshan Vivante Apartment, Sector 137, Noida, 高塔姆·布德纳加尔, 北方邦, 201305

[

](https://www.facebook.com/geeksforgeeks.org/)[

](https://www.instagram.com/geeks_for_geeks/)[

](https://in.linkedin.com/company/geeksforgeeks)[

](https://twitter.com/geeksforgeeks)[

](https://www.youtube.com/geeksforgeeksvideos)

[![Play 商店中的 GFG 应用](https://media.geeksforgeeks.org/auth-dashboard-uploads/googleplay.png)](https://geeksforgeeksapp.page.link/gfg-app)[![App Store 上的 GFG App](https://media.geeksforgeeks.org/auth-dashboard-uploads/appstore.png)](https://geeksforgeeksapp.page.link/gfg-app)

[广告合作](https://www.geeksforgeeks.org/advertise-with-us/)

- 公司
- [关于我们](https://www.geeksforgeeks.org/about/)
- [合法的](https://www.geeksforgeeks.org/legal/)
- [隐私政策](https://www.geeksforgeeks.org/legal/privacy-policy/)
- [在媒体上](https://www.geeksforgeeks.org/press-release/)
- [联系我们](https://www.geeksforgeeks.org/about/contact-us/)
- [广告合作](https://www.geeksforgeeks.org/advertise-with-us/)
- [GFG企业解决方案](https://www.geeksforgeeks.org/gfg-corporate-solution/)
- [就业培训计划](https://www.geeksforgeeks.org/campus-training-program/)

- [语言](https://www.geeksforgeeks.org/introduction-to-programming-languages/)
- [Python](https://www.geeksforgeeks.org/python-programming-language/)
- [Java](https://www.geeksforgeeks.org/java/)
- [C++](https://www.geeksforgeeks.org/c-plus-plus/)
- [PHP](https://www.geeksforgeeks.org/php-tutorials/)
- [Go语言](https://www.geeksforgeeks.org/golang/)
- [SQL](https://www.geeksforgeeks.org/sql-tutorial/)
- [R 语言](https://www.geeksforgeeks.org/r-tutorial/)
- [Android 教程](https://www.geeksforgeeks.org/android-tutorial/)
- [教程存档](https://www.geeksforgeeks.org/geeksforgeeks-online-tutorials-free/)

- [数字减影血管造影](https://www.geeksforgeeks.org/learn-data-structures-and-algorithms-dsa-tutorial/)
- [DSA 教程](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/)
- [基本DSA问题](https://www.geeksforgeeks.org/basic-coding-problems-in-dsa-for-beginners/)
- [DSA 路线图](https://www.geeksforgeeks.org/complete-roadmap-to-learn-dsa-from-scratch/)
- [DSA 面试常见问题 100 个](https://www.geeksforgeeks.org/top-100-data-structure-and-algorithms-dsa-interview-questions-topic-wise/)
- [Sandeep Jain 的 DSA 路线图](https://www.geeksforgeeks.org/dsa-roadmap-for-beginner-to-advanced-by-sandeep-jain/)
- [所有备忘单](https://www.geeksforgeeks.org/geeksforgeeks-master-sheet-list-of-all-cheat-sheets/)

- [数据科学与机器学习](https://www.geeksforgeeks.org/ai-ml-ds/)
- [使用 Python 进行数据科学](https://www.geeksforgeeks.org/data-science-tutorial/)
- [数据科学初学者](https://www.geeksforgeeks.org/data-science-for-beginners/)
- [机器学习](https://www.geeksforgeeks.org/machine-learning/)
- [机器学习数学](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [数据可视化](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [熊猫](https://www.geeksforgeeks.org/pandas-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [自然语言处理](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [深度学习](https://www.geeksforgeeks.org/deep-learning-tutorial/)

- [Web 技术](https://www.geeksforgeeks.org/web-technology/)
- [HTML](https://www.geeksforgeeks.org/html/)
- [CSS](https://www.geeksforgeeks.org/css/)
- [JavaScript](https://www.geeksforgeeks.org/javascript/)
- [TypeScript](https://www.geeksforgeeks.org/typescript/)
- [ReactJS](https://www.geeksforgeeks.org/learn-reactjs/)
- [NextJS](https://www.geeksforgeeks.org/nextjs/)
- [引导](https://www.geeksforgeeks.org/bootstrap/)
- [网页设计](https://www.geeksforgeeks.org/web-design/)

- [Python 教程](https://www.geeksforgeeks.org/python-programming-language/)
- [Python 编程示例](https://www.geeksforgeeks.org/python-programming-examples/)
- [Python 项目](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/)
- [Python Tkinter](https://www.geeksforgeeks.org/python-tkinter-tutorial/)
- [Python 网页抓取](https://www.geeksforgeeks.org/python-web-scraping-tutorial/)
- [OpenCV 教程](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [Python 面试题](https://www.geeksforgeeks.org/python-interview-questions/)
- [Django](https://www.geeksforgeeks.org/django-tutorial/)

- 计算机科学
- [操作系统](https://www.geeksforgeeks.org/operating-systems/)
- [计算机网络](https://www.geeksforgeeks.org/computer-network-tutorials/)
- [数据库管理系统](https://www.geeksforgeeks.org/dbms/)
- [软件工程](https://www.geeksforgeeks.org/software-engineering/)
- [数字逻辑设计](https://www.geeksforgeeks.org/digital-electronics-logic-design-tutorials/)
- [工程数学](https://www.geeksforgeeks.org/engineering-mathematics-tutorials/)
- [软件开发](https://www.geeksforgeeks.org/software-development/)
- [软件测试](https://www.geeksforgeeks.org/software-testing-tutorial/)

- [DevOps](https://www.geeksforgeeks.org/devops-tutorial/)
- [Git](https://www.geeksforgeeks.org/git-tutorial/)
- [Linux](https://www.geeksforgeeks.org/linux-tutorial/)
- [AWS](https://www.geeksforgeeks.org/aws-tutorial/)
- [Docker](https://www.geeksforgeeks.org/docker-tutorial/)
- [Kubernetes](https://www.geeksforgeeks.org/kubernetes-tutorial/)
- [Azure](https://www.geeksforgeeks.org/microsoft-azure/)
- [地理控制协议](https://www.geeksforgeeks.org/google-cloud-platform-tutorial/)
- [DevOps路线图](https://www.geeksforgeeks.org/devops-roadmap/)

- [系统设计](https://www.geeksforgeeks.org/system-design-tutorial/)
- [高级设计](https://www.geeksforgeeks.org/what-is-high-level-design-learn-system-design/)
- [低级设计](https://www.geeksforgeeks.org/what-is-low-level-design-or-lld-learn-system-design/)
- [UML 图](https://www.geeksforgeeks.org/unified-modeling-language-uml-introduction/)
- [面试指南](https://www.geeksforgeeks.org/system-design-interview-guide/)
- [设计模式](https://www.geeksforgeeks.org/software-design-patterns/)
- [面向对象应用设计](https://www.geeksforgeeks.org/object-oriented-analysis-and-design/)
- [系统设计训练营](https://www.geeksforgeeks.org/system-design-interview-bootcamp-guide/)
- [面试问题](https://www.geeksforgeeks.org/most-commonly-asked-system-design-interview-problems-questions/)

- [面试准备](https://www.geeksforgeeks.org/technical-interview-preparation/)
- [竞技编程](https://www.geeksforgeeks.org/competitive-programming-a-complete-guide/)
- [CP 的顶级 DS 或 Algo](https://www.geeksforgeeks.org/top-algorithms-and-data-structures-for-competitive-programming/)
- [公司招聘流程](https://www.geeksforgeeks.org/company-wise-recruitment-process/)
- [公司层面的准备](https://www.geeksforgeeks.org/company-preparation/)
- [能力准备](https://www.geeksforgeeks.org/aptitude-questions-and-answers/)
- [谜题](https://www.geeksforgeeks.org/puzzles/)

- 学校科目
- [数学](https://www.geeksforgeeks.org/maths/)
- [物理](https://www.geeksforgeeks.org/physics/)
- [化学](https://www.geeksforgeeks.org/chemistry/)
- [生物学](https://www.geeksforgeeks.org/biology/)
- [社会科学](https://www.geeksforgeeks.org/social-science/)
- [英语语法](https://www.geeksforgeeks.org/english-grammar/)
- [商业](https://www.geeksforgeeks.org/commerce/)
- [世界GK](https://www.geeksforgeeks.org/tag/world-general-knowledge/)

- [GeeksforGeeks 视频](https://www.geeksforgeeks.org/videos/)
- [数字减影血管造影](https://www.geeksforgeeks.org/videos/category/sde-sheet/)
- [Python](https://www.geeksforgeeks.org/videos/category/python/)
- [Java](https://www.geeksforgeeks.org/videos/category/java-w6y5f4/)
- [C++](https://www.geeksforgeeks.org/videos/category/c/)
- [Web 开发](https://www.geeksforgeeks.org/videos/category/web-development/)
- [数据科学](https://www.geeksforgeeks.org/videos/category/data-science/)
- [计算机科学科目](https://www.geeksforgeeks.org/videos/category/cs-subjects/)

[@GeeksforGeeks，Sanchhaya Education Private Limited](https://www.geeksforgeeks.org/) ，[保留所有权利](https://www.geeksforgeeks.org/copyright-information/)

我们使用 Cookie 来确保您在网站上获得最佳浏览体验。使用我们的网站即表示您已阅读并理解我们的 [Cookie 政策](https://www.geeksforgeeks.org/cookie-policy/)和 [隐私政策。](https://www.geeksforgeeks.org/privacy-policy/)知道了 ！

![灯箱](https://www.geeksforgeeks.org/dsa/array-data-structure-guide/)