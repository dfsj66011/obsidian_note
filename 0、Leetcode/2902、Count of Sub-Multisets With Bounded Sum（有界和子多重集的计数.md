
2639（Easy，略）


# 2902、Count of Sub-Multisets With Bounded Sum（有界和子多重集的计数）

Tag: Hard、Array、Hash Table、Dynamic Programming、Sliding Window


## 问题描述

计算给定数组 `nums` 中满足元素和在 `[l, r]` 范围内的子多重集的数量。

子多重集是从数组中选取的元素集合，其中：

* 每个元素可以出现 0 次或多次（最多不超过其在原数组中的出现次数）
* 顺序无关紧要（它是无序的）
* 如果两个子多重集包含相同元素且出现频率相同，则认为它们是相同的

例如，如果 `nums = [1,2,2,3]`:

* 你可以使用 0、1 或 2 份元素 2（因为它出现了两次）
* 你可以使用 0 或 1 份元素 1 和 3（因为它们各自出现一次）
* 空集是有效的，其总和为 0

该解决方案采用 *动态规划方法*，并针对处理同一元素的多次出现进行了优化：

1. **初始设置**：创建一个 DP 数组，其中 `dp[i]` 表示和为 `i` 的子多重集的数量。单独处理零，因为它们不影响总和但会乘以方式的数量（可以包含 0、1、2...直到所有零）。
2. **处理每个唯一元素**：对于每个唯一的数字及其出现频率：
	* 创建一个累加和的 `stride` 数组：`stride[i] = dp[i] + dp[i-num] + dp[i-2*num] + ...`
	* 这表示使用任意数量的 `num` 来构成总和 `i` 的所有可能方式。
	* 然后根据频率限制进行调整：如果我们最多只能使用 `freq` 份副本，我们就减去使用超过 `freq` 份副本的贡献。
3. **最终计数**：对区间 `[l, r]` 内所有 `dp[i]` 求和，并乘以 `(zeros + 1)` 以考虑包含零的所有可能方式。

为了防止计数结果过大，采用了模运算 10^9 + 7 来确保结果在合理范围内。


## 直觉

关键洞见在于将其视为经典“子集和”问题的变体，但有一个特殊之处：元素可以根据其在原始数组中的出现频率被多次使用。

让我们思考如何逐步构建子多重集。如果我们已经有一个针对某些元素的动态规划（DP）解决方案，该如何整合一个出现频率为 `freq` 的新元素呢？

最直观的做法是遍历新元素所有可能的出现次数（0 到 `freq`）并更新动态规划数组。对于一个出现 `freq` 次的元素 `num`，我们需要考虑将 `0*num`、`1*num`、`2*num` ...直到 `freq*num` 累加到现有和值中。这种嵌套循环的实现方式效率较低。

聪明的优化来自于识别一种模式。当我们想知道使用最多 `freq` 份 `num` 来构成总和 `i` 有多少种方法时，可以这样思考：

- 不使用 `num`（即 `dp[i]`）形成 `i` 的方法
- 加上形成 `i-num` 的方法（然后添加一个 `num`）
- 加上形成 `i-2*num` 的方法（然后添加两个 `num`）
- 以此类推...

这形成了一个累加和模式！步长数组高效地捕捉了这一点：`stride[i]` 包含了 `dp[i] + dp[i-num] + dp[i-2*num] + ...` 的总和，表示使用任意数量 `num` 来构成和 `i` 的所有方式。

但等等——我们最多只能使用 `freq` 份副本。因此，如果 `stride[i]` 包含了使用超过 `freq` 份副本的贡献，我们需要将其减去。这就是为什么在可能的情况下，我们会计算 `stride[i] - stride[i - num*(freq+1)]` ——这样可以去除使用 `freq+1` 或更多份副本的贡献。

零值需要单独处理，因为它们比较特殊：添加零不会改变总和，但每个零都会使我们的选择翻倍（包含或不包含）。因此，如果我们有 `zeros` 个零，我们需要将最终计数乘以 `(zeros + 1)`，以涵盖所有 `2^zeros` 种包含它们的方式。

这种方法通过巧妙地利用步长技术重复利用计算结果，将指数级问题转化为多项式级问题。

## Solution Implementation


```python
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 1_000_000_007
      
        # Initialize DP array where dp[i] represents the number of submultisets with sum i
        # dp[0] = 1 because there's one way to make sum 0 (empty submultiset)
        dp = [1] + [0] * r
      
        # 计算 nums 中每个数字的频率
        frequency_map = collections.Counter(nums)
      
        # 单独处理 0，因为他们对 sum 没有贡献，但增加有效子多重集的计数
        zero_count = frequency_map.pop(0, 0)
      
        # Process each unique number and its frequency
        for number, frequency in frequency_map.items():
            # Create a prefix sum array to efficiently calculate
            # dp[i] + dp[i - number] + dp[i - 2*number] + ...
            prefix_sum = dp.copy()
          
            # Build prefix sum array by accumulating values
            for sum_value in range(number, r + 1):
                prefix_sum[sum_value] += prefix_sum[sum_value - number]
          
            # Update dp array considering we can use 0 to 'frequency' copies of 'number'
            for sum_value in range(r, 0, -1):
                if sum_value >= number * (frequency + 1):
                    # If sum is large enough, we subtract the contribution of
                    # using more than 'frequency' copies of the current number
                    # This gives us exactly 0 to 'frequency' copies
                    dp[sum_value] = prefix_sum[sum_value] - prefix_sum[sum_value - number * (frequency + 1)]
                else:
                    # If sum is small, we can use all available combinations
                    dp[sum_value] = prefix_sum[sum_value]
      
        # Calculate final result:
        # - Sum dp values from l to r (inclusive)
        # - Multiply by (zero_count + 1) to account for all possible zero selections
        # - Apply modulo to keep result within bounds
        result = (zero_count + 1) * sum(dp[l:r + 1]) % MOD
      
        return result
```

## 解决方案方法

该实现采用动态规划方法，并运用了一种巧妙的优化手段来处理元素的多次出现：

**1. 初始化和设置**

```python
dp = [1] + [0] * r
frequency_map = collections.Counter(nums)
zero_count = frequency_map.pop(0, 0)
```

* 创建一个 `dp` 数组，其中 `dp[i]` 记录总和恰好为 `i` 的子多重集数量
* 初始化 `dp[0] = 1`（构成和为 0 的一种方式：空集）
* 使用 `Counter` 统计每个元素的出现频率
* 单独提取并处理零值元素，因为它们不影响总和

**2. 处理每个独特元素**

对于每个唯一的数字及其频率：

```python
for num, freq in count.items():
    stride = dp.copy()
    for i in range(num, r + 1):
        stride[i] += stride[i - num]
```

* 创建一个<code>stride</code>数组作为当前<code>dp</code>的副本
* 构建累加和：<code>stride[i] = dp[i] + dp[i-num] + dp[i-2*num] + ...</code>
* 这表示使用 0、1、2... 个 <code>num</code> 来构成总和 <code>i</code>

**3. 应用频率约束**

```python
for i in range(r, 0, -1):
    if i >= num * (freq + 1):
        dp[i] = stride[i] - stride[i - num * (freq + 1)]
    else:
        dp[i] = stride[i]
```

* 反向遍历求和以避免干扰
* 如果<code>i >= num * (freq + 1)</code>，我们可以使用差分公式：
	* <code>stride[i]</code> 包含使用不限数量 <code>num</code> 的所有方式
	* <code>stride[i - num * (freq + 1)]</code>表示使用至少 <code>freq + 1</code> 个副本的方式
	* 两者之差即为最多使用 <code>freq</code> 个副本的方式
* 否则，<code>stride[i]</code>已经符合频率限制（无论如何也不能使用超过 <code>i/num</code> 个副本）

**4. 计算最终结果**

```python
return (zeros + 1) * sum(dp[l : r + 1]) % kMod
```

* 对范围 <code>[l, r]</code> 内所有有效的子多重集计数进行求和
* 乘以 <code>(zeros + 1)</code> 以考虑包含/排除零的所有方式
* 应用模运算以保持结果在范围内

**关键洞察：**

* 步幅技术通过预先计算累积和，将每次求和的 <code>O(freq)</code> 操作转化为 <code>O(1)</code>
* 逆序处理可避免过早使用更新后的值
* 分离零值简化了逻辑，因为它们是乘法因子而非加法项
* 总时间复杂度：<code>O(n * r)</code>，其中 <code>n</code> 代表唯一元素数量，<code>r</code> 表示上限值


### 示例演练

让我们通过一个小例子来演示，其中 <code>nums = [1, 2, 2, 3]</code>，<code>l = 1</code>，<code>r = 5</code>。

**步骤 1：初始化**

* 统计频率：<code>{1: 1, 2: 2, 3: 1}</code>（不含零值）
* 创建动态规划数组：<code>dp = [1, 0, 0, 0, 0, 0]</code>（索引代表  0-5的和）
* <code>dp[0] = 1</code> 表示有一种方法可以得到和为 0（空集）


**步骤 2：** 处理元素 1（频率=1）

* 创建步长数组：<code>stride = [1, 0, 0, 0, 0, 0]</code>
* 为 i=1 到 5 构建累加和：
	* <code>stride[1] = stride[1] + stride[0] = 0 + 1 = 1</code>
	* <code>stride[2] = stride[2] + stride[1] = 0 + 1 = 1</code>
	* <code>stride[3] = stride[3] + stride[2] = 0 + 1 = 1</code>
	* <code>stride[4] = stride[4] + stride[3] = 0 + 1 = 1</code>
	* <code>stride[5] = stride[5] + stride[4] = 0 + 1 = 1</code>
* 应用频率约束（频率=1，因此最多 1 份副本）：
	* 对于 i=5：<code>5 >= 1*(1+1) = 2</code>，因此 <code>dp[5] = stride[5] - stride[3] = 1 - 1 = 0</code>
	* 对于 i=4：<code>4 >= 2</code>，因此 <code>dp[4] = stride[4] - stride[2] = 1 - 1 = 0</code>
	* 对于 i=3：<code>3 >= 2</code>，因此 <code>dp[3] = stride[3] - stride[1] = 1 - 1 = 0</code>
	* 对于 i=2：<code>2 >= 2</code>，因此<code>dp[2] = stride[2] - stride[0] = 1 - 1 = 0</code>
	* 对于 i=1：<code>1 < 2</code>，因此 <code>dp[1] = stride[1] = 1</code></li>
* 结果：<code>dp = [1, 1, 0, 0, 0, 0]</code>（可以用 {} 得到和为 0，用 {1} 得到和为 1）


**步骤 3：处理元素 2（频率=2）**

* 创建步幅数组：<code>stride = [1, 1, 0, 0, 0, 0]</code>
* 构建累加和：
	* <code>stride[2] = 0 + stride[0] = 0 + 1 = 1</code>
	* <code>stride[3] = 0 + stride[1] = 0 + 1 = 1</code>
	* <code>stride[4] = 0 + stride[2] = 0 + 1 = 1</code>
	* <code>stride[5] = 0 + stride[3] = 0 + 1 = 1</code></li>

* 应用频率约束（频率=2，最多 2 份副本）：
	* 对于 i=5：`5<=2*(2+1) = 6`，因此 <code>dp[5] = stride[5] = 1</code>
	* 对于 i=4：`4 < 6`，因此<code>dp[4] = stride[4] = 1</code>
	* 对于 i=3：`3 < 6`，因此<code>dp[3] = stride[3] = 1</code>
	* 对于 i=2：`2 < 6`，因此<code>dp[2] = stride[2] = 1</code>
	* 对于 i=1：`1 < 6`，因此<code>dp[1] = stride[1] = 1</code>
`

结果：<code>dp = [1, 1, 1, 1, 1, 1]</code>，可能的集合：{}, {1}, {2}, {1,2}, {2,2}, {1,2,2}


**步骤 4：处理元素 3（频率=1）**

* 创建步幅数组：<code>stride = [1, 1, 1, 1, 1, 1]</code>
* 构建累加和：
	* <code>stride[3] = 1 + stride[0] = 1 + 1 = 2</code>
	* <code>stride[4] = 1 + stride[1] = 1 + 1 = 2</code>
	* <code>stride[5] = 1 + stride[2] = 1 + 1 = 2</code>

* 应用频率约束（freq = 1，最多 1 份副本）：
	* 当 i = 5 时：`5 <3*(1+1) = 6`，因此 <code>dp[5] = stride[5] = 2</code>
	* 当 i = 4 时：`4 < 6`，因此 <code>dp[4] = stride[4] = 2</code>
	* 当 i = 3 时：`3 < 6`，因此 <code>dp[3] = stride[3] = 2</code>
	* 当 i = 2 时：`2 < 6`，因此 <code>dp[2] = stride[2] = 1</code>
	* 当 i = 1 时：`1 < 6`，因此 <code>dp[1] = stride[1] = 1</code>

* 最终结果：<code>dp = [1, 1, 1, 2, 2, 2]</code>


**步骤 5：计算结果**

* 区间 `[1, 5]` 的和：<code>dp[1] + dp[2] + dp[3] + dp[4] + dp[5] = 1 + 1 + 2 + 2 + 2 = 8</code>
* 没有零需要相乘，所以答案为 8

8 个有效子多重集，其和在 `[1, 5]` 范围内为：

* {1} → sum = 1
* {2} → sum = 2
* {3} → sum = 3
* {1,2} → sum = 3
* {2,2} → sum = 4
* {1,3} → sum = 4
* {2,3} → sum = 5
* {1,2,2} → sum = 5


## 时空复杂度

**时间复杂度：** `O(n * r)`，其中 `n` 是 `nums` 中唯一元素的数量，`r` 是求和范围的上限。

算法遍历计数器中的每个唯一元素（最多 <code>n</code> 个唯一元素）。对于每个值为 <code>num</code> 的唯一元素：

* 创建 <code>stride</code> 数组需要 <code>O(r)</code> 时间
* 计算 <code>stride</code> 中的前缀和需要 <code>O(r)</code> 时间（从 <code>num</code> 迭代到 <code>r</code>）
* 更新 <code>dp</code> 数组需要 <code>O(r)</code> 时间（从 <code>r</code> 向下迭代到 1）

因此，对于每个唯一元素，我们执行 <code>O(r)</code> 次操作，最终总时间复杂度为 <code>O(n * r)</code>。

空间复杂度： <code>O(r)</code>

空间复杂度主要由以下因素决定：

* 大小为 <code>r + 1</code> 的 <code>dp</code> 数组：<code>O(r)</code>
* 大小为 <code>r + 1</code> 的 <code>stride</code> 数组：<code>O(r)</code>
* 最多存储 <code>n</code> 个唯一元素的 <code>count</code> 字典：<code>O(n)</code>

因为在涉及求和范围的问题中，通常预期 <code>r</code> 大于 <code>n</code>，而我们为数组使用了 <code>O(r)</code> 的空间，所以整体空间复杂度为 <code>O(r)</code>。



--------

# 2767、Partition String Into Minimum Beautiful Substrings（将字符串分割为最小数量的美丽子串）

Tag：Medium、Hash Table、String、Dynamic Programming、Backtracking


## 问题描述

给定一个仅由 '0' 和 '1' 组成的二进制字符串 `s`。你的任务是将这个字符串分割成一个或多个子字符串，其中每个子字符串都必须是美丽的。

如果一个子字符串满足以下两个条件，则被认为是美丽的：

1. 它不包含前导零（即不能以 “0” 开头，除非它只是 “1”）
2. 当被视为二进制数时，它表示 5 的幂（如 1、5、25、125 等）。

例如：

* "1" 很美，因为它的二进制是 1，即 5 的 0 次方
* "101" 很美，因为它的二进制是 5，即 5 的 1 次方
* "11001" 很美，因为它的二进制是 25，即 5 的 2 次方
* "01" 不美，因为它有前导零
* "10" 不美，因为 2 不是 5 的幂次方

你需要找到可以将字符串分割成的 *最少* 数量的美丽子串。如果无法将字符串分割成仅由美丽子串组成，则返回 -1。

例如，如果字符串 `s = "1011"`，你可以将其分割为 "1" + "011"，但 "011" 以零开头，因此不美观。你也可以尝试 "101" + "1"，其中 "101" 是二进制的 5（美），"1" 是二进制的 1（美），这样我们总共得到 2 个美的子串。

该解决方案采用 *带记忆化的动态规划方法*。它预先处理所有 5 的幂次方直至合理上限，并将它们存储在集合中。然后使用递归函数 `dfs(i)` 尝试从索引 `i` 开始的所有可能字符串分割方式，检查每个子串是否表示 5 的幂次方，并返回所需的最小分割次数。

## 流程图演练

首先，让我们用[流程图](https://algo.monster/flowchart)来确定算法。以下是逐步的讲解：

**这是图吗？** 否：该问题涉及字符串的分割，而非处理图结构中的节点和边。

**需要求解第 $k$ 个最小/最大值吗？**  不：我们寻找的是最小分区数，而不是第 $k$ 小/大的元素。

**涉及链表吗？** 不：这个问题涉及的是字符串操作，而不是链表操作。

**这个问题有小的限制吗？**

是的：从问题来看，字符串的长度通常较小（在这类问题中通常 ≤15-20 个字符），我们需要探索所有可能的分割方式。虽然可能的分割方式数量呈指数级增长，但由于约束条件较小，这是可以处理的。

**暴力破解/回溯法？**

是的：我们需要尝试所有可能的方式将字符串分割为优美的子串。在每个位置，我们都要做出选择：当前子串在哪里结束。然后我们递归地解决剩余的字符串。如果某个选择导致无效的分割（如子串以零开头或不表示 5 的幂次），我们会回溯并尝试不同的分割点。

**结论：** 流程图正确地将其识别为一个回溯问题。该解决方案采用了带有记忆化（动态编程优化）的回溯方法，其中：

1. 我们尝试当前子串的所有可能结束位置
2. 检查当前子串是否为"优美"的（无前导零且是 5 的幂次）
3. 递归求解剩余字符串
4. 若当前路径无法得到有效解则回溯
5. 使用记忆化存储避免重复计算相同子问题

## 直觉

当我们需要将一个字符串分割成具有特定属性的有效子串时，在每个位置上我们都面临一个决策：当前子串应该在哪里结束？这自然引出了一个递归方法，我们会尝试所有可能性。

可以这样理解：就像剪绳子一样，在每个位置，我们都可以选择剪一刀或者继续延伸。对于这个问题，我们从字符串的开头出发，不断自问：“如果我将第一个字符、前两个字符、前三个字符……作为一个子串截取出来，哪种选择能让我得到最少的分割次数？”

关键点在于我们需要探索所有有效的分割点。从位置 `i` 开始，我们可以形成子字符串 `s[i:i+1]`、`s[i:i+2]`、……、`s[i:n]`。对于每个子字符串，我们需要检查：

1. 它是否有前导零？如果有，跳过它。
2. 在二进制中它是否是 5 的幂？如果不是，跳过它。
3. 如果有效，我们就找到了一个美丽的子串，现在我们需要为剩余的字符串解决同样的问题。

由于 5 的幂次是固定值（1、5、25、125、625……），我们可以预先计算它们。在二进制中，这些值分别为：`1`、`101`、`11001`、`1111101` 等。我们将它们存储在一个集合中，以实现 `O(1)` 时间复杂度的查找。

递归的性质变得清晰：`minimum partitions from position i = 1 + minimum partitions from position j+1`，其中 `j` 是我们决定结束当前子串的位置。我们尝试所有有效的 `j` 值，并取最小值。

我们注意到，可能会多次解决相同的子问题（例如，在多个递归路径中可能需要从位置 5 开始寻找最小分割）。这种子问题的重叠表明可以使用记忆化来缓存结果，从而将我们的回溯方法转化为动态规划。

如果在任何时候我们无法形成有效的分区（例如遇到必须以子串开头的 '0'），我们将返回无穷大来表示不可能。如果最小值仍然是无穷大，则最终答案为 -1，否则就是找到的最小分区数。


## 解决方案

```python
from functools import cache
from math import inf
from typing import Set

class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        @cache
        def find_min_partitions(start_index: int) -> int:
            """
            Find minimum number of partitions from start_index to end of string.
            Each partition must be a binary representation of a power of 5.
          
            Args:
                start_index: Current position in the string
              
            Returns:
                Minimum number of partitions needed, or inf if impossible
            """
            # Base case: reached end of string
            if start_index >= string_length:
                return 0
          
            # Cannot partition if current substring starts with '0'
            # (no power of 5 has leading zeros in binary)
            if s[start_index] == "0":
                return inf
          
            current_number = 0
            min_partitions = inf
          
            # Try all possible substrings starting from current position
            for end_index in range(start_index, string_length):
                # Build the number by shifting left and adding the current bit
                current_number = (current_number << 1) | int(s[end_index])
              
                # If current number is a power of 5, try partitioning here
                if current_number in powers_of_five:
                    remaining_partitions = find_min_partitions(end_index + 1)
                    min_partitions = min(min_partitions, 1 + remaining_partitions)
          
            return min_partitions
      
        # Initialize variables
        string_length = len(s)
      
        # Generate all powers of 5 that could appear in the string
        # Maximum possible value is 2^n - 1 where n is string length
        powers_of_five: Set[int] = set()
        power_of_five = 1
        powers_of_five.add(power_of_five)
      
        # Generate powers of 5 up to maximum possible value
        for _ in range(string_length):
            power_of_five *= 5
            powers_of_five.add(power_of_five)
      
        # Find minimum partitions starting from index 0
        result = find_min_partitions(0)
      
        # Return -1 if no valid partition exists, otherwise return the result
        return -1 if result == inf else result
```


## 解决方案方法

该实现采用记忆化搜索（递归动态规划）来寻找美丽子串的最小数量。

**步骤 1：预处理 5 的幂**

首先，我们生成所有可能出现在字符串中的 5 的幂。由于字符串长度为 $n$，最大十进制值为 $2^n - 1$。我们创建一个包含 $5$ 的幂的集合 `ss`：

```python
x = 1
ss = {x}     # Start with 5^0 = 1

for i in range(n):
    x *= 5
    ss.add(x)  # Add 5^1, 5^2, 5^3, ...
```

**第二步：定义递归函数**

我们定义 <code>dfs(i)</code>，它返回从索引 <code>i</code> 到字符串 <code>s</code> 末尾所需的最小分割数。该函数的工作原理如下：

* 基本情况：如果 `i >= n`，表示我们已经成功处理了整个字符串，返回 0。
* 无效情况：如果 `s[i] == "0"`，当前位置以 '0' 开头，使得任何从此处开始的子字符串都无效（前导零）。返回 `inf` 表示不可能。
* 递归情况：尝试从 `i` 到 `n-1` 的所有可能的结束位置 `j`：

```python
x = 0
ans = inf
for j in range(i, n):
    x = x << 1 | int(s[j])  # Build decimal value using bit operations
    if x in ss:  # Check if it's a power of 5
        ans = min(ans, 1 + dfs(j + 1))  # Take minimum
```

**步骤 3：高效构建十进制值**

当我们从位置 <code>i</code> 到 <code>j</code> 扩展子字符串时，我们逐步构建十进制值：

* `x = x << 1` 将现有位左移（乘以 2）
* `| int(s[j])` 添加新位

例如，如果我们正在构建 “101”：

* 开始: <code>x = 0</code>
* 添加 '1': <code>x = 0 <<1 | 1 = 1</code> (二进制: 1)
* 添加 '0': <code>x = 1 <<1 | 0 = 2</code> (二进制: 10)
* 添加 '1': <code>x = 2 <<1 | 1 = 5</code> (二进制: 101)

**步骤 4：记忆**

<code>@cache</code> 装饰器会自动对 <code>dfs</code> 函数进行记忆化处理，存储每个唯一 <code>i</code> 值对应的结果。当相同的子问题出现在不同的递归路径中时，这可以避免冗余计算。

**步骤 5：最终答案**

我们调用<code>dfs(0)</code>来获取从索引 0 开始的最小分割数。如果结果是<code>inf</code>，意味着无法分割，因此返回<code>-1</code>。否则，返回最小分割数。

时间复杂度为<code>O(n²)</code>，其中<code>n</code>代表可能的起始位置数量，每个起始位置最多需要检查<code>n</code>个终止位置。空间复杂度为<code>O(n)</code>，用于递归调用栈和记忆化缓存。

### 示例演练

让我们以 <code>s = "11001"</code> 为例，逐步解析解决方案：

**第 1 步：预处理 5 的幂**

* 生成 5 的幂次序列：<code>ss = {1, 5, 25, 125, ...}</code>
* 二进制表示：<code>{1, 101, 11001, 1111101, ...}</code>

**步骤 2：从索引 0 开始深度优先搜索**

调用 <code>dfs(0)</code>:

* <code>s[0] = '1'</code>（不是 '0'，所以我们可以继续）
* 尝试所有从索引 0 开始的可能子串：

**选项 1：** 取子字符串 <code>s[0:1] = "1"</code>

* 构建十进制数：<code>x = 1</code>（二进制 "1"=十进制 1）
* 检查：1 是否在 <code>ss</code> 中？在！（1 = 5^0）
* 递归求解：<code>1 + dfs(1)</code>
	* 调用 <code>dfs(1)</code>：<code>s[1] = '1'</code>
	* 尝试子串 <code>s[1:2] = "1"</code>：x=1，在 `ss` 中✓，得到 <code>1 + dfs(2)</code>
	* 尝试子串 <code>s[1:3] = "10"</code>：x=2，不在 `ss` 中 ✗
	* 尝试子串 <code>s[1:4] = "100"</code>：x=4，不在 `ss` 中 ✗
	* 尝试子串 <code>s[1:5] = "1001"</code>：x=9，不在 `ss` 中 ✗
	* <code>dfs(1)</code> 返回<code>inf</code>（未找到有效分割）
* 选项 1 总计：<code>1 + inf = inf</code>

**选项 2：** 取子字符串 <code>s[0:2] = "11"</code>

* 构建十进制数：<code>x = 3</code>（二进制 "11" = 十进制 3）
* 检查：3 是否在 <code>ss</code>中？不在！
* 跳过此选项

**选项 3：** 取子字符串 <code>s[0:3] = "110"</code>

* 构建十进制：<code>x = 6</code>（二进制 "110" = 十进制 6）
* 检查：6 是否在 <code>ss</code> 中？不在！
* 跳过此选项

**选项 4：** 取子字符串 <code>s[0:4] = "1100"</code>

* 构建十进制：<code>x = 12</code>（二进制 "1100" = 十进制 12）
* 检查：12 是否在<code>ss</code>中？不在！
* 跳过此选项

**选项 5：** 取子字符串 <code>s[0:5] = "11001"</code>

* 构建十进制数：<code>x = 25</code>（二进制 "11001" = 十进制 25）
* 检查：25 是否在<code>ss</code>中？是！（25 = 5^2）
* 递归求解：<code>1 + dfs(5)</code>
	* 调用 <code>dfs(5)</code>：索引 5 ≥ 长度 5
	* 基础情况：返回 0
* 选项 5 总计：<code>1 + 0 = 1</code>


**最终结果：**

* <code>dfs(0)</code> 返回 <code>min(inf, inf, inf, inf, 1) = 1</code>
* 字符串 "11001" 可以被分割成 1 个美丽子串（其本身表示 25 = 5^2）

答案是 1.


## Time and Space Complexity

时间复杂度： <code>O(n²)</code>

该算法采用了带有记忆化的动态规划方法。<code>dfs</code>函数最多被调用<code>n</code>次（对每个从0到n-1的起始位置各调用一次），每次调用都会被缓存。对于每次调用<code>dfs(i)</code>，该函数会在for循环中遍历从<code>i</code>到<code>n-1</code>的所有位置，执行<code>O(n-i)</code>次操作。在最坏情况下，这将导致：

<ul>
<li>Position 0: iterates through n positions</li>
<li>Position 1: iterates through n-1 positions</li>
<li>Position 2: iterates through n-2 positions</li>
<li>...and so on</li>
</ul>
<p>The total work done is <code>n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = O(n²)</code>.</p>
<p><strong>Space Complexity:</strong> <code>O(n)</code></p>
<p>The space complexity consists of:</p>
<ol>
<li>The recursion call stack depth, which is at most <code>O(n)</code> when the string is partitioned into n single-character substrings</li>
<li>The memoization cache stores at most <code>n</code> different states (one for each starting position)</li>
<li>The set <code>ss</code> stores powers of 5, and since we only generate up to <code>n</code> powers of 5, it uses <code>O(n)</code> space</li>
<li>Other variables use <code>O(1)</code> space</li>
</ol>
<p>Therefore, the overall space complexity is <code>O(n)</code>.</p>



--------

# 2454. Next Greater Element IV（下一个更大的元素 IV）

Tag：Hard、Stack、Array、Binary Search、Sorting、Monotonic Stack、Heap (Priority Queue)

## 问题描述

给定一个从 0 开始索引的非负整数数组 <code>nums</code>。你的任务是为数组中的每个元素找到<strong>第二大的</strong>整数。

对于元素 <code>nums[i]</code>，它的第二大整数被定义为满足以下条件的 <code>nums[j]</code>：

* <code>j >i</code>（该元素必须在数组中的<code>nums[i]</code>之后出现）
* <code>nums[j] >nums[i]</code>（该元素必须大于<code>nums[i]</code>）
* 存在<strong>唯一一个</strong>索引<code>k</code>，满足<code>i &lt; k &lt; j</code> 且 <code>nums[k] >nums[i]</code>（两者之间必须恰好有一个元素也大于 <code>nums[i]</code>）

换句话说，<code>nums[j]</code> 是位于 <code>nums[i]</code> 右侧第二个大于<code>nums[i]</code>的元素。

如果对于 <code>nums[i]</code> 不存在这样的第二大元素，则结果应为 <code>-1</code>。

**例如：** 对于数组 <code>[1, 2, 4, 3]</code>:

* 对于 <code>1</code>（索引0）：第一个更大的元素是索引 1 处的 <code>2</code>，第二个更大的元素是索引 2 处的 <code>4</code>。结果：<code>4</code>
* 对于 <code>2</code>（索引 1）：第一个更大的元素是索引 2 处的 <code>4</code>，第二个更大的元素是索引 3 处的 <code>3</code>。结果：<code>3</code>
* 对于 <code>4</code>（索引 2）：其右侧没有更大的元素。结果：<code>-1</code>
* 对于 <code>3</code>（索引 3）：其右侧没有元素。结果：<code>-1</code>


该函数应返回一个整数数组 <code>answer</code>，其中 <code>answer[i]</code> 表示<code>nums[i]</code>的第二大整数。

## 直觉

关键思路是按从大到小的顺序处理元素。为什么？因为当我们寻找任何值的"第二大"元素时，我们只关心比它大的元素。通过从大到小处理，我们确保在到达某个元素时，所有比它大的元素都已经被处理过了。

这样想：如果我们处于元素 <code>x</code>，我们知道所有大于 <code>x</code> 的元素都已经被看到了。现在我们需要找出这些较大的元素中，哪个会在原始数组中 <code>x</code> 的位置之后作为第二个出现。

有序集合在这里变得至关重要。当我们处理每个元素时，我们会维护一个已遍历元素索引的有序集合（这些元素都大于或等于当前元素）。对于当前位于索引<code>i</code>处的元素，我们需要在集合中查找大于<code>i</code>的索引（即出现在位置<code>i</code>之后的元素）。

通过使用二分查找（<code>bisect_right(i)</code>），我们可以快速找到有序集合中应插入<code>i</code>的位置。这给出了大于<code>i</code>的索引数量。如果至少存在两个这样的索引（即 <code>j + 1 &lt; len(sl)</code>），那么第二个索引处的元素（<code>sl[j + 1]</code>）就是我们的答案——它是位置<code>i</code>之后第二个值大于 <code>nums[i]</code> 的元素。

这种方法的美妙之处在于，通过按元素值降序处理，我们自然过滤掉了较小的元素。当我们在寻找 `nums[i]` 的第二大元素时，有序集合中仅包含值 `≥ nums[i]` 的元素索引，而我们只需找到原始数组顺序中位于 `i` 之后的第二个索引即可。

## Solution Implementation

```python
from sortedcontainers import SortedList
from typing import List

class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        # Create pairs of (value, index) for sorting
        value_index_pairs = [(value, index) for index, value in enumerate(nums)]

        # Sort pairs by value in descending order (largest first)
        value_index_pairs.sort(key=lambda pair: -pair[0])

        # SortedList to maintain indices of processed elements in sorted order
        processed_indices = SortedList()

        # Initialize result array with -1 (default when no second greater element exists)
        n = len(nums)
        result = [-1] * n

        # Process elements from largest to smallest value
        for value, current_index in value_index_pairs:
            # Find position where current_index would be inserted (indices greater than current_index)
            position = processed_indices.bisect_right(current_index)

            # Check if there are at least 2 elements to the right of current_index
            if position + 1 < len(processed_indices):
                # The element at position+1 is the second element to the right
                second_greater_index = processed_indices[position + 1]
                result[current_index] = nums[second_greater_index]

            # Add current index to the sorted list for future iterations
            processed_indices.add(current_index)

        return result
```


## Solution Approach

该实现采用了一种基于 *排序* 的方法，通过有序集合高效地找到每个位置的第二大元素。

**第一步：** 创建并排序配对

```python
arr = [(x, i) for i, x in enumerate(nums)]
arr.sort(key=lambda x: -x[0])
```

我们为每个元素创建 <code>(value, index)</code> 对，并按值降序排序。确保我们优先处理较大的元素。

**步骤 2：初始化数据结构**

```python
sl = SortedList()
ans = [-1] * n
```

* <code>SortedList</code> 在处理元素时保持索引的排序顺序
* <code>ans</code> 数组初始化为 <code>-1</code>（当不存在第二大的元素时的默认值）

**步骤 3：从大到小处理元素**

```python
for _, i in arr:
    j = sl.bisect_right(i)
    if j + 1 < len(sl):
        ans[i] = nums[sl[j + 1]]
    sl.add(i)
```

对于原始索引 <code>i</code> 处的每个元素：

* **在已排序列表中查找位置**：<code>sl.bisect_right(i)</code> 返回将 <code>i</code> 插入以保持排序顺序的位置。这给出了集合中已存在且大于 <code>i</code> 的索引数量。
* **检查是否存在第二个更大的元素**：如果<code>j + 1 &lt; len(sl)</code>，意味着集合中至少有两个索引大于 <code>i</code>。<code>sl[j + 1]</code> 处的元素是原始数组中 <code>i</code> 之后的第二个索引。
* **更新答案**：设置 <code>ans[i] = nums[sl[j + 1]]</code> 以记录第二个更大的元素。
* **添加当前索引：** 将 <code>i</code> 添加到已排序列表中，以便处理未来的元素。


**原理说明：** 当处理索引为 i、值为 <code>x</code> 的元素时，已排序列表中已包含所有值 <code>≥ x</code> 的元素的索引（因为我们按降序处理）。在这些索引中，我们需要精确找到原始数组中位于位置 <code>i</code> 之后出现的第二个索引。<code>bisect_right</code> 操作能高效定位 i 在已排序索引列表中的插入位置，而 <code>sl[j + 1]</code> 则给出大于 <code>i</code> 的第二个索引。

**时间复杂度：** 排序为 O(n log n)，在排序列表中进行 n 次插入和二分查找也为 <code>O(n log n)</code>。<strong>空间复杂度：</strong> 额外数据结构占用 <code>O(n)</code>。


### Example Walkthrough

让我们一步步遍历数组 <code>[2, 4, 0, 9, 6]</code>。

**初始设置**

* 创建配对：<code>[(2,0), (4,1), (0,2), (9,3), (6,4)]</code>
* 按值排序（降序）：<code>[(9,3), (6,4), (4,1), (2,0), (0,2)]</code>
* 初始化：<code>sl = []</code>, <code>ans = [-1, -1, -1, -1, -1]</code>

**处理每个元素（从大到小）：**

**第一步：** 处理 <code>(9, 3)</code> - 索引 3 处的值为 9

* <code>sl.bisect_right(3)</code> = 0（`sl` 中尚无索引 > 3）
* 由于 0 + 1 = 1 不小于 `sl` 长度 0，不存在第二个更大的元素
* 将 3 添加到 `sl`：<code>sl = [3]</code>
* <code>ans = [-1, -1, -1, -1, -1]</code>

**第二步：** 处理 <code>(6, 4)</code> - 索引 4 处的值为 6

* <code>sl.bisect_right(4)</code> = 1（`sl` 中所有索引都 ≤ 4）
* 由于 1 + 1 = 2 不小于 `sl` 长度 1，因此没有第二个更大的元素
* 将 4 添加到 `sl`：<code>sl = [3, 4]</code>
* <code>ans = [-1, -1, -1, -1, -1]</code>

**步骤 3：** 处理 <code>(4, 1)</code> - 索引 1 处的值 4 

* <code>sl.bisect_right(1)</code> = 0（索引 3 和 4 都大于 1）
* 由于 0 + 1 = 1 <  `sl` 长度 2，我们找到了第二个更大的元素！
* <code>ans[1] = nums[sl[1]] = nums[4] = 6</code>
* 将 1 添加到 `sl`：<code>sl = [1, 3, 4]</code>
* <code>ans = [-1, 6, -1, -1, -1]</code>

**步骤 4：** 处理 <code>(2, 0)</code> - 索引 0 处的值 2

* <code>sl.bisect_right(0)</code> = 0（索引 1、3、4 处的值均大于 0）
* 由于 0 + 1 = 1 < `sl` 长度 3，说明存在第二个更大元素！
* <code>ans[0] = nums[sl[1]] = nums[3] = 9</code>
* 将 0 添加到 `sl` 中：<code>sl = [0, 1, 3, 4]</code>
* <code>ans = [9, 6, -1, -1, -1]</code>

**步骤 5：** 处理 <code>(0, 2)</code> - 索引 2 处的值为 0

* <code>sl.bisect_right(2)</code> = 2（索引 3 和 4 对应的值大于 2）
* 由于 2 + 1 = 3 <  `sl` 长度 4，说明存在第二个更大的元素！
* <code>ans[2] = nums[sl[3]] = nums[4] = 6</code>
* 将 2 添加到 `sl` 中：<code>sl = [0, 1, 2, 3, 4]</code>
* <code>ans = [9, 6, 6, -1, -1]</code>

**最终结果：** <code>[9, 6, 6, -1, -1]</code>


## 时间与空间复杂度


<p><strong>Time Complexity:</strong> <code>O(n × log n)</code></p>
<p>The time complexity is dominated by three main operations:</p>
<ol>
<li>Creating the array of tuples <code>arr</code>: <code>O(n)</code></li>
<li>Sorting <code>arr</code> by value in descending order: <code>O(n × log n)</code></li>
<li>Iterating through <code>arr</code> and performing operations with <code>SortedList</code>: <code>O(n × log n)</code>
<ul>
<li>Each <code>bisect_right()</code> operation takes <code>O(log n)</code> time</li>
<li>Each <code>add()</code> operation takes <code>O(log n)</code> time to maintain sorted order</li>
<li>We perform both operations <code>n</code> times</li>
</ul>
</li>
</ol>
<p>Since these operations are sequential, the overall time complexity is <code>O(n) + O(n × log n) + O(n × log n) = O(n × log n)</code>.</p>
<p><strong>Space Complexity:</strong> <code>O(n)</code></p>
<p>The space complexity comes from:</p>
<ol>
<li>The <code>arr</code> list containing <code>n</code> tuples: <code>O(n)</code></li>
<li>The <code>SortedList</code> which can contain up to <code>n</code> elements: <code>O(n)</code></li>
<li>The <code>ans</code> list of size <code>n</code>: <code>O(n)</code></li>
</ol>
<p>All these data structures use linear space, resulting in a total space complexity of <code>O(n)</code>.</p>
<div class="MarkdownRenderer_markdown__OXPld "><h2>Common Pitfalls</h2>
<h3>Pitfall: Misunderstanding the Index Relationship in SortedList</h3>
<p><strong>The Problem:</strong>
A common mistake is misinterpreting what <code>bisect_right(current_index)</code> returns and how it relates to finding the "second greater" element. Developers often confuse:</p>
<ol>
<li>The position in the SortedList (what bisect_right returns)</li>
<li>The actual indices stored in the SortedList</li>
<li>Which element represents the "second greater"</li>
</ol>
<p><strong>Example of the Mistake:</strong></p>
<pre><pre class="CodeBlock_codeBlock__96Xj8" style="display: block; background: transparent; padding: 0px; color: rgb(51, 51, 51); overflow-x: auto;"><code class="language-python" style="white-space: pre;"><span style="color: rgb(150, 152, 150);"># INCORRECT interpretation</span><span>
</span>position = processed_indices.bisect_right(current_index)
<span></span><span style="color: rgb(150, 152, 150);"># Wrong: Thinking position itself is an index in nums</span><span>
</span><span>result[current_index] = nums[position + </span><span style="color: rgb(0, 92, 197);">1</span><span>]  </span><span style="color: rgb(150, 152, 150);"># This would cause IndexError or wrong result</span></code></pre></pre>
<p><strong>Why This Happens:</strong>
When <code>bisect_right(current_index)</code> returns a value like 2, it means there are 2 indices in the SortedList that are greater than <code>current_index</code>. However, these indices might be something like [5, 7] - not [0, 1]. The confusion arises because:</p>
<ul>
<li><code>position</code> is just a count/position in the SortedList</li>
<li><code>processed_indices[position]</code> gives you the actual array index</li>
<li>You need <code>processed_indices[position + 1]</code> for the second greater element's index</li>
</ul>
<p><strong>The Correct Solution:</strong></p>
<pre><pre class="CodeBlock_codeBlock__96Xj8" style="display: block; background: transparent; padding: 0px; color: rgb(51, 51, 51); overflow-x: auto;"><code class="language-python" style="white-space: pre;"><span style="color: rgb(150, 152, 150);"># CORRECT implementation</span><span>
</span>position = processed_indices.bisect_right(current_index)
<span></span><span style="color: rgb(215, 58, 73);">if</span><span> position + </span><span style="color: rgb(0, 92, 197);">1</span><span> &lt; </span><span class="hljs-built_in">len</span><span>(processed_indices):
</span><span>    </span><span style="color: rgb(150, 152, 150);"># Get the actual index from the SortedList at position+1</span><span>
</span><span>    second_greater_index = processed_indices[position + </span><span style="color: rgb(0, 92, 197);">1</span><span>]
</span><span>    </span><span style="color: rgb(150, 152, 150);"># Use that index to get the value from nums</span><span>
</span>    result[current_index] = nums[second_greater_index]</code></pre></pre>
<p><strong>Visual Example:</strong>
For <code>nums = [2, 4, 0, 9, 6]</code>, when processing element <code>0</code> at index 2:</p>
<ul>
<li><code>processed_indices</code> might be <code>[1, 3, 4]</code> (indices of elements 4, 9, and 6)</li>
<li><code>bisect_right(2)</code> returns <code>1</code> (index 2 would go after position 0 but before position 1)</li>
<li><code>processed_indices[1]</code> = 3 (first greater element's index)</li>
<li><code>processed_indices[2]</code> = 4 (second greater element's index)</li>
<li>So <code>result[2] = nums[4] = 6</code></li>
</ul>
<p><strong>Key Takeaway:</strong>
Always remember that SortedList operations return positions within the list, not the actual values stored in the list. You must use these positions to access the actual indices, then use those indices to access values in the original array.</p></div></div><div class="LockContainer_actionForm__4UVeZ"><div class="LockContainer_signInPrompt__Kh1G7"><div>Unlock Full Access</div><div>Free plan can only access <b>2</b> Editorials per month<br>You have used <b>2</b> free accesses</div><div><ul><li>In-Depth Strategy Explanations</li><li>Step-by-Step Example Walkthroughs</li><li>Time &amp; Space Complexity Analysis</li><li>Common Pitfalls &amp; Edge Cases</li></ul></div></div><button type="button" class="LockContainer_actionButton__SYVTK btn btn-primary">Upgrade to premium</button></div></div><hr class="tw-mt-2">


# 2628、JSON Deep Equal（JSON 深度相等，可删）

Tag：Medium

## Problem Description

这个问题要求你实现一个函数，用于检查两个值是否深度相等。深度相等不仅意味着比较表面层的值，还包括递归比较所有嵌套结构。

该函数接收两个参数 `o1` 和 `o2`，并返回一个布尔值，表示根据以下规则它们是否深度相等：

1. 原始值（数字、字符串、布尔值、null、undefined）：如果两个原始值通过严格相等检查 `===`，则它们是深度相等的。例如，`5 === 5` 返回 `true`，但 `5 === "5"` 返回 `false`。

2. 数组：两个数组在以下情况下深度相等：
    
    - 它们具有相同的长度
    - 相同索引处的每个元素都深度相等
    - 顺序很重要 - `[1, 2]` 与 `[2, 1]` 不深度相等

3. 对象：两个对象在以下情况下被认为是深度相等的：
    
    - 它们具有完全相同的键集合
    - 一个对象中每个键对应的值与另一个对象中相同键的值深度相等
    - 额外的键或缺失的键会导致对象不深度相等

该问题保证两个输入值都是有效的 JSON（JSON.parse 的输出），因此您无需处理特殊的 JavaScript 对象，如函数、未定义值作为对象属性或循环引用。

示例：

- `areDeeplyEqual(1, 1)` returns `true` (primitive equality)
- `areDeeplyEqual([1, [2, 3]], [1, [2, 3]])` returns `true` (nested arrays with same structure)
- `areDeeplyEqual({a: 1, b: {c: 2}}, {a: 1, b: {c: 2}})` returns `true` (nested objects with same structure)
- `areDeeplyEqual([1, 2], [1, 3])` returns `false` (different array elements)
- `areDeeplyEqual({a: 1}, {a: 1, b: 2})` returns `false` (different keys)

## Solution Implementation

```python
def areDeeplyEqual(o1, o2):
    """
    Deeply compares two values for equality, including nested objects and arrays

    Args:
        o1: First value to compare
        o2: Second value to compare

    Returns:
        bool: True if values are deeply equal, False otherwise
    """
    # Handle primitives and None values
    if o1 is None or not isinstance(o1, (dict, list)):
        return o1 == o2

    # Check if types are different
    if type(o1) != type(o2):
        return False

    # Check if one is list and the other is not
    if isinstance(o1, list) != isinstance(o2, list):
        return False

    # Handle list comparison
    if isinstance(o1, list):
        # Check list lengths
        if len(o1) != len(o2):
            return False

        # Compare each list element recursively
        for i in range(len(o1)):
            if not areDeeplyEqual(o1[i], o2[i]):
                return False

        return True
    else:
        # Handle dictionary comparison
        keys1 = list(o1.keys())
        keys2 = list(o2.keys())

        # Check if dictionaries have different number of keys
        if len(keys1) != len(keys2):
            return False

        # Compare each property recursively
        for key in keys1:
            # Check if key exists in second dictionary
            if key not in o2:
                return False
            if not areDeeplyEqual(o1[key], o2[key]):
                return False

        return True
```


# 2798. Number of Employees Who Met the Target(达成目标的员工人数，过于简单，可删)



