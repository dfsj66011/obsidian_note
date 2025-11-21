
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