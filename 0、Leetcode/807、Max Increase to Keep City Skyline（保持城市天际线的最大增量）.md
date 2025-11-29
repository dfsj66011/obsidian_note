
Tag：Medium、Greedy、Array、Matrix


## 问题描述

你有一个由 `n x n` 网格表示的城市，其中每个单元格 `grid[r][c]` 表示位于第 `r` 行第 `c` 列的建筑高度。

天际线是指从远处各个方向（东、南、西、北）眺望城市时所看到的景象。当从以下角度观看时：

- 南北方向：每列中的最高建筑高度可见
- 东西方向：每行中的最高建筑高度可见

你的任务是通过增加建筑物的高度来最大化总增加量，但有一个限制条件：从四个方向看的天际线必须保持不变。

例如，如果位于位置 `(i, j)` 的建筑高度为 `grid[i][j]`，你可以将其增加到 `min(rowMax[i], colMax[j])`，其中：

* `rowMax[i]` 表示第 `i` 行的最大高度（从东/西方向看的天际线）
* `colMax[j]` 表示第 `j` 列的最大高度（从南/北方向看的天际线）

这确保了：

- 该建筑不会比其所在行的最高建筑更高（保留东西向天际线）。
- 该建筑不会超过其所在列的最高建筑（保留南北天际线）

在不改变任何天际线视图的情况下，返回可以增加所有建筑物高度的最大总和。


## 直觉

关键是要理解什么决定了天际线的形状。无论从哪个方向看城市，我们只能看到每行或每列中最高的建筑——这些构成了天际线的轮廓。

考虑位于位置 `(i, j)` 的一栋建筑。如果我们增加它的高度：

- 不能超过当前第 `i` 行的最大值，否则会改变东西方向的天际线
- 它不能超过列 `j` 的当前最大值，否则我们将改变南北天际线。

因此，任何建筑物的最大安全高度是这两个约束条件中的最小值：`min(rowMax[i], colMax[j])`。

这是如何实现的？让我们思考一下，当我们将建筑物的高度精确增加到 `min(rowMax[i], colMax[j])` 时会发生什么：

* 如果 `rowMax[i] < colMax[j]`，建筑物达到了行的最大值但低于列的最大值。行的天际线保持不变（最大值相同），列的天际线不受影响。
* 如果 `colMax[j] < rowMax[i]`，则建筑物达到了列的最大值但仍低于行的最大值。同理可证。
* 如果它们相等，建筑物会同时达到两个极限，但不会超过任何一个。

这种 *贪心算法* 是最优的，因为我们在各自限制条件下独立最大化每栋建筑的高度。由于每栋建筑对总高度的贡献是独立的，逐个最大化就能得到总和的最大值。

解决方案自然如下：首先扫描网格以找出所有行和列的最大值，然后对于每栋建筑，计算我们可以增加多少：`min(rowMax[i], colMax[j]) - grid[i][j]`。


## 解决方案

该实现采用了一种简单的贪心算法，通过两次遍历网格来完成：

**步骤 1：计算行和列的最大值**

首先，我们需要找出每行和每列的最大高度：

- `row_max = [max(row) for row in grid]` - 此操作遍历每一行并找出其最大值
- `col_max = [max(col) for col in zip(*grid)]` - 这里使用 `zip(*grid)` 来转置矩阵（将行转换为列），然后找到每一列的最大值。

`zip(*grid)` 操作是 Python 中的一个惯用法，它能有效地转置矩阵。例如，如果 `grid = [[1,2], [3,4]]`，那么 `zip(*grid)` 会得到 `[(1,3), (2,4)]`，即矩阵的列。

**步骤 2：计算总增长**

接下来，我们计算所有可能增加的总和：

```python
sum(
    min(row_max[i], col_max[j]) - x
    for i, row in enumerate(grid)
    for j, x in enumerate(row)
)
```

这个嵌套的理解：

1. 遍历网格中每个值为 `x` 的单元格 `(i, j)` 
2. 计算最大安全高度：`min(row_max[i], col_max[j])`
3. 减去当前高度 `x` 得到增量
4. 对所有增量进行求和

**时间复杂度**： `O(n²)` where `n` is the grid dimension - we traverse the grid twice (once for finding maximums, once for calculating increases)

**空间复杂度**： `O(n)` for storing the `row_max` and `col_max` arrays

这个解决方案的美妙之处在于其简洁性——通过预先计算约束条件（行和列的最大值），我们可以通过一次计算确定每栋建筑的最佳高度。


### 示例演练

Let's walk through a small example to illustrate the solution approach.

Consider this 3×3 grid:

```
grid = [[3, 0, 8],
        [2, 4, 5],
        [9, 2, 6]]
```

**Step 1: Calculate Row and Column Maximums**

First, find the maximum height in each row:

- Row 0: max(3, 0, 8) = 8
- Row 1: max(2, 4, 5) = 5
- Row 2: max(9, 2, 6) = 9
- `row_max = [8, 5, 9]`

Next, find the maximum height in each column:

- Column 0: max(3, 2, 9) = 9
- Column 1: max(0, 4, 2) = 4
- Column 2: max(8, 5, 6) = 8
- `col_max = [9, 4, 8]`

**Step 2: Calculate Increases for Each Building**

Now examine each building and determine how much we can increase it:

Position (0,0): Current height = 3

- Row max = 8, Column max = 9
- Can increase to min(8, 9) = 8
- Increase = 8 - 3 = **5**

Position (0,1): Current height = 0

- Row max = 8, Column max = 4
- Can increase to min(8, 4) = 4
- Increase = 4 - 0 = **4**

Position (0,2): Current height = 8

- Row max = 8, Column max = 8
- Can increase to min(8, 8) = 8
- Increase = 8 - 8 = **0** (already at maximum)

Position (1,0): Current height = 2

- Row max = 5, Column max = 9
- Can increase to min(5, 9) = 5
- Increase = 5 - 2 = **3**

Position (1,1): Current height = 4

- Row max = 5, Column max = 4
- Can increase to min(5, 4) = 4
- Increase = 4 - 4 = **0** (already at maximum)

Position (1,2): Current height = 5

- Row max = 5, Column max = 8
- Can increase to min(5, 8) = 5
- Increase = 5 - 5 = **0** (already at maximum)

Position (2,0): Current height = 9

- Row max = 9, Column max = 9
- Can increase to min(9, 9) = 9
- Increase = 9 - 9 = **0** (already at maximum)

Position (2,1): Current height = 2

- Row max = 9, Column max = 4
- Can increase to min(9, 4) = 4
- Increase = 4 - 2 = **2**

Position (2,2): Current height = 6

- Row max = 9, Column max = 8
- Can increase to min(9, 8) = 8
- Increase = 8 - 6 = **2**

**Total sum of increases = 5 + 4 + 0 + 3 + 0 + 0 + 0 + 2 + 2 = 16**

The final grid after all increases would be:

```
[[8, 4, 8],
 [5, 4, 5],
 [9, 4, 8]]
```

Notice how the skylines remain unchanged:

- From North/South (column maxes): [9, 4, 8] ✓
- From East/West (row maxes): [8, 5, 9] ✓

## 完整代码


```python
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        # Find the maximum height in each row (skyline from the side)
        row_maximums = [max(row) for row in grid]
      
        # Find the maximum height in each column (skyline from the front/back)
        # zip(*grid) transposes the grid to access columns as rows
        column_maximums = [max(column) for column in zip(*grid)]
      
        # Calculate the total increase possible
        total_increase = 0
      
        # Iterate through each cell in the grid
        for row_index, row in enumerate(grid):
            for column_index, current_height in enumerate(row):
                # The maximum height for this cell is limited by both skylines
                # We take the minimum of the row and column maximum heights
                max_allowed_height = min(row_maximums[row_index], column_maximums[column_index])
              
                # Add the difference between max allowed and current height
                total_increase += max_allowed_height - current_height
      
        return total_increase
```

## 时空复杂度

The time complexity is `O(n²)`, where `n` is the side length of the matrix `grid`.

- Computing `row_max`: Iterating through each row and finding the maximum takes `O(n)` per row, with `n` rows total, resulting in `O(n²)`.
- Computing `col_max`: The `zip(*grid)` operation transposes the matrix in `O(n²)` time, and finding the maximum for each column takes `O(n)` per column, with `n` columns total, resulting in `O(n²)`.
- The final sum operation uses nested iteration through all `n²` elements of the grid, where each operation inside (accessing `row_max[i]`, `col_max[j]`, and computing the difference) takes `O(1)` time, resulting in `O(n²)`.

Overall time complexity: `O(n²) + O(n²) + O(n²) = O(n²)`.

The space complexity is `O(n)`, where `n` is the side length of the matrix `grid`.

- `row_max` stores `n` values (one maximum per row): `O(n)`.
- `col_max` stores `n` values (one maximum per column): `O(n)`.
- The generator expression in the sum doesn't create additional storage beyond temporary variables: `O(1)`.

Overall space complexity: `O(n) + O(n) = O(n)`.

**Learn more about [how to find time and space complexity quickly](https://algo.monster/problems/runtime_summary).**

## 常见陷阱

### 1. **Modifying the Original Grid**

A common mistake is attempting to modify the grid in-place while calculating the increases, which can lead to incorrect results if you need to reference the original values later.

**Incorrect approach:**

```python
def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    row_max = [max(row) for row in grid]
    col_max = [max(col) for col in zip(*grid)]
    total = 0
  
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            new_height = min(row_max[i], col_max[j])
            total += new_height - grid[i][j]
            grid[i][j] = new_height  # DON'T DO THIS - modifies input
  
    return total
```

**Solution:** Keep the grid unchanged and only calculate the differences.

### 2. **Assuming Square Grid**

The code assumes an `n x n` square grid, but the problem might have rectangular grids (`m x n`). Using incorrect indices can cause index out of bounds errors.

**Incorrect approach:**

```python
def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    n = len(grid)  # Assumes square grid
    row_max = [max(grid[i]) for i in range(n)]
    col_max = [max(grid[i][j] for i in range(n)) for j in range(n)]  # Wrong for non-square
```

**Solution:** Handle rectangular grids properly:

```python
def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    row_max = [max(row) for row in grid]
    col_max = [max(grid[i][j] for i in range(rows)) for j in range(cols)]
```

### 3. **Misunderstanding the Constraint**

Some might think they need to increase every building to exactly `min(row_max[i], col_max[j])`, but the problem asks for the maximum _total_ increase, which means each building should be increased as much as possible (up to the constraint).

**Incorrect interpretation:** Thinking you need to choose which buildings to increase. **Correct interpretation:** Every building should be increased to its maximum allowed height.

### 4. **Empty Grid Edge Case**

Not handling empty grids or grids with empty rows can cause the code to crash.

**Solution:** Add validation:

```python
def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    if not grid or not grid[0]:
        return 0
  
    # Rest of the implementation...
```

### 5. **Using Wrong Transpose Method**

Attempting to transpose manually with incorrect indexing is error-prone.

**Incorrect approach:**

```python
col_max = []
for j in range(len(grid)):
    col_max.append(max(grid[i][j] for i in range(len(grid))))  # Might fail if not square
```

**Solution:** Use `zip(*grid)` which handles the transpose elegantly and works for any rectangular grid



--------


# 32. Longest Valid Parentheses（最长有效括号）

Tag：Hard、Stack、String、Dynamic Programming

## Problem Description

给定一个仅包含字符 '(' 和 ')' 的字符串，你需要找出最长有效（格式正确）括号子串的长度。

一个有效的括号子串意味着：

- 每个左括号 '(' 都有一个对应的右括号 ')'
- 括号正确嵌套且平衡
- 它在原始字符串中形成一个连续的子串

例如：

- 在字符串 "(()" 中，最长有效括号子串是 "()"，长度为 2
- 在字符串 ")()())" 中，最长有效括号子串是 "()()"，长度为 4
- 在字符串 "(())" 中，整个字符串都是有效的，长度为 4

目标是返回给定字符串中所有有效括号子串的最大长度。如果不存在有效的括号子串，则返回 0。


## Intuition

要找到最长的有效括号子串，我们需要思考什么使括号有效以及如何高效地跟踪它们的长度。

关键洞察在于，有效的括号具有一种构建模式——它们要么从先前的有效序列延伸而来，要么形成新的有效配对。当我们遇到一个闭合括号“)”时，它可能通过以下两种方式之一完成一个有效序列：

1. 它直接与前一个左括号 '(' 配对，形成一个简单的括号对 "()"。在这种情况下，我们得到了一个有效长度为 2 的序列，但还需要检查这个括号对之前是否存在一个有效的序列，以便我们可以将其连接起来。
2. 它与前面已经验证过的序列之前的开括号配对。例如，在"((...))"中，当我们处理最后一个')'时，它会与第一个 '(' 配对，包裹着一个内部有效的序列。

这表明可以使用动态规划，其中 `f[i]` 表示以位置 `i-1` 结尾的最长有效括号的长度。

为什么要在特定位置结束？因为当我们扫描字符串时，需要知道是否可以扩展或连接之前有效的序列。通过跟踪每个位置结束的有效括号的长度，我们可以：

- 回溯查找匹配的左括号
- 连接相邻的有效序列
- 基于先前计算结果进行构建

关键观察在于，当我们找到一个有效配对时，我们需要检查：

- 这对序列所完成的有效序列的长度（如果有的话）
- 在新形成的序列之前出现的任何有效序列的长度

这样一来，我们就可以将原本分开的有效序列在形成新对后拼接起来，比如 "()" 和 "()" 可以识别为连续的 "()()"，其长度为 4。


## Solution Approach

我们采用动态规划的方法实现该解决方案，使用一维数组 `f`，其中 `f[i]` 表示字符串中以下标 `i-1` 结尾的最长有效括号子串的长度。

**初始化：**

* 创建一个大小为 `n+1` 的数组 `f`，初始化为零，其中 `n` 是字符串的长度
* 我们使用基于 `1` 的索引以便更轻松地处理边界条件

**处理每个字符：** 我们从索引 `i=1` 开始遍历字符串。对于原字符串中位置 `i-1` 的每个字符：

**情况 1：当前字符是 `'('`**

* 有效的括号不能以开括号结尾
* `f[i]` 保持为 0（已初始化）

**情况 2：当前字符是 `')'`，** 我们需要考虑以下两种子情况：

**子情况 2.1：** 前一个字符 `s[i-2]` 是 `'('`

* 我们有一个直接的对 `“()”`
* 长度为 `f[i-2] + 2`
* `f[i-2]` 表示在此对之前结束的任何有效序列

**子情况 2.2：** 前一个字符 `s[i-2]` 是 `')'`

* 我们可能会有一个像 `“((...))”` 这样的模式
* 首先，找到可能包含匹配的 `'('` 的位置 `j`
* 计算 `j = i - f[i-1] - 1`，其中 `f[i-1]` 表示以 `i-2` 结尾的有效括号的长度。
* 如果 `j > 0` 且 `s[j-1] = '('`，我们找到了一个匹配对
* 总长度为：`f[i-1] + 2 + f[j-1]`
	* `f[i-1]`：内部有效序列的长度
	* `2`：新匹配的对
	* `f[j-1]`：在位置 `j-1` 之前结束的任何有效序列的长度

**寻找答案：** ​ 在处理完所有字符后，数组 `f` 中的最大值即为最长有效括号子串的长度。

**时间复杂度**：`O(n)` —— 只需遍历字符串一次
**空间复杂度：** `O(n)` —— 用于存储动态规划数组

这种方法的优点在于它系统地建立在先前计算结果的基础上，确保我们捕捉到所有可能的有效序列及其关联。


### Example Walkthrough

让我们用字符串 `“(())”` 来逐步解析这个解决方案：

**设置：**

* 字符串：`"(())"`
* 长度 `n = 4`
* 创建大小为 5 的 dp 数组 `f`：`[0, 0, 0, 0, 0]`
* 使用基于 1 的索引，其中 `f[i]` 表示字符串中结束于位置 `i-1` 的最长有效括号

**逐步处理：**

**i = 1, 字符 = '(' 在位置 0:**

* 开括号不能结束有效序列
* `f[1] = 0`
* 数组：`[0, 0, 0, 0, 0]`

**i = 2, 字符 = '(' 在位置 1:**

* 另一个左括号
* `f[2] = 0`
* 数组：`[0, 0, 0, 0, 0]`

**i = 3, 字符 = ')' 在位置 2:**

- 位置 1 的前一个字符是 '('
- 这形成了一个直接的配对 "()"
- `f[3] = f[1] + 2 = 0 + 2 = 2`
- 数组：`[0, 0, 0, 2, 0]`

**i = 4, 字符 = ')' 在位置 3:**

* 前一个字符在位置 2 的是 ')'
- 需要为当前的 ')' 找到匹配的 '('
- 计算位置：`j = i - f[i-1] - 1 = 4 - 2 - 1 = 1`
- 检查 `j > 0`（是的，1 > 0）且 `s[j-1] = '('`（是的，`s[0] = '('`）
- 我们找到了匹配！第一个 '(' 与最后一个 ')' 匹配
- `f[4] = f[3] + 2 + f[0] = 2 + 2 + 0 = 4`
    - `f[3] = 2`：内部有效序列 "()" 的长度
    - `2`：用于外部匹配对
    - `f[0] = 0`：位置 0 之前没有有效序列
- 数组：`[0, 0, 0, 2, 4]`

**结果：** 数组中的最大值为 4，这正确地表明整个字符串 "(())" 是有效的。

这个例子展示了算法如何：

1. 首先识别内部的一对 "()"
2. 然后识别出外层的括号包裹着这个有效序列
3. 将两者结合，得到总长度为 4

## Solution Implementation

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # Get the length of the input string
        n = len(s)
      
        # dp[i] represents the length of the longest valid parentheses substring
        # ending at position i-1 in the original string (1-indexed for easier calculation)
        dp = [0] * (n + 1)
      
        # Iterate through each character with 1-based indexing
        for i, char in enumerate(s, 1):
            # Only closing parentheses can form valid pairs
            if char == ")":
                # Case 1: Current ')' matches with previous '(' to form "()"
                if i > 1 and s[i - 2] == "(":
                    # Add 2 for the new pair and include any valid substring before it
                    dp[i] = dp[i - 2] + 2
              
                # Case 2: Current ')' might match with a '(' before a valid substring
                else:
                    # Find the position before the valid substring ending at i-1
                    j = i - dp[i - 1] - 1
                  
                    # Check if there's a matching '(' at position j-1
                    if j > 0 and s[j - 1] == "(":
                        # Length = previous valid substring + 2 for new pair + 
                        # any valid substring before the matching '('
                        dp[i] = dp[i - 1] + 2 + dp[j - 1]
      
        # Return the maximum length found
        return max(dp)
```

## Time and Space Complexity

时间复杂度为 `O(n)`，其中 `n` 是字符串的长度。该算法通过一个单层 for 循环遍历字符串一次，每个字符仅被处理一次。在循环内部，所有操作（数组访问、比较和算术运算）都在常数时间 `O(1)` 内完成。

空间复杂度为 `O(n)`，其中 `n` 是字符串的长度。该算法创建了一个大小为 `n+1` 的辅助数组f来存储动态规划的状态。这个数组存储了以每个位置结尾的最长有效括号子串的长度，所需空间与输入字符串长度成线性关系。


## Common Pitfalls

### 1. **Off-by-One Errors with Index Mapping**

The most common pitfall in this solution is confusion between the 1-based indexing of the DP array and 0-based indexing of the string. This leads to incorrect index calculations when looking for matching parentheses.

**Problem Example:**

```python
# Incorrect: Mixing up indices
j = i - dp[i-1] - 1  # This is correct for 1-based dp
if j > 0 and s[j] == "(":  # Wrong! Should be s[j-1]
    dp[i] = dp[i-1] + 2 + dp[j]  # Wrong! Should be dp[j-1]
```

**Solution:** Always remember that `dp[i]` corresponds to `s[i-1]`. When calculating position `j` in the dp array, access the string at `s[j-1]`.

### 2. **Missing Boundary Checks**

Failing to verify that indices are within valid bounds before accessing array elements can cause runtime errors.

**Problem Example:**

```python
# Without proper boundary checking
if s[i-2] == "(":  # Crashes when i = 1
    dp[i] = dp[i-2] + 2
```

**Solution:** Always add boundary checks:

```python
if i > 1 and s[i-2] == "(":
    dp[i] = dp[i-2] + 2
```

### 3. **Forgetting to Accumulate Previous Valid Sequences**

A critical mistake is only counting the current matching pair without including adjacent valid sequences.

**Problem Example:** For string `"()()"`:

```python
# Incorrect: Only counting the current pair
if s[i-2] == "(":
    dp[i] = 2  # Wrong! Misses previous valid sequences
```

This would give `dp = [0, 0, 2, 0, 2]` instead of `[0, 0, 2, 0, 4]`.

**Solution:** Always add the length of any valid sequence that comes before:

```python
dp[i] = dp[i-2] + 2  # Correctly accumulates previous sequences
```

### 4. **Incorrect Calculation of Matching Position**

When looking for the matching opening parenthesis for a closing one, using the wrong formula to calculate position `j`.

**Problem Example:**

```python
# Incorrect ways to find the matching position
j = i - dp[i-1]      # Missing the -1 for the current ')'
j = i - dp[i] - 1    # Using dp[i] which hasn't been calculated yet
j = dp[i-1] - 1      # Completely wrong logic
```

**Solution:** The correct formula accounts for:

- Current position `i`
- Length of valid substring ending at `i-1`: `dp[i-1]`
- The current `)` itself: `-1`

Correct calculation: `j = i - dp[i-1] - 1`

### 5. **Not Handling Empty String or Edge Cases**

The code might fail on edge cases like empty strings or strings with only opening/closing parentheses.

**Solution:** The current implementation handles these well:

- Empty string: `max([0])` returns 0
- All `(`: All dp values remain 0
- All `)`: No valid pairs formed, all dp values remain 0

### 6. **Using 0-based DP Array Without Adjustment**

Some might try to use 0-based indexing for the DP array, which requires careful adjustment of all index calculations.

**Problem Example:**

```python
# Using 0-based dp array without proper adjustments
dp = [0] * n
for i in range(n):
    if s[i] == ")":
        if i > 0 and s[i-1] == "(":
            dp[i] = dp[i-2] + 2  # Wrong! i-2 could be negative
```

**Solution:** Either stick with 1-based indexing as shown in the original solution, or carefully adjust all index calculations and add appropriate boundary checks for 0-based indexing.