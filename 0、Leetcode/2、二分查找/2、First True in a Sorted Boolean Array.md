
一个布尔值数组被分成两部分：左半部分全为 `false`，右半部分全为 `true`。在已排序的布尔数组中，找到右半部分的第一个 `true`，即第一个 `true` 元素的索引。如果没有 `true` 元素，则返回 `-1`。


Input: `arr = [false, false, true, true, true]`
Output: `2`

```python
def find_boundary(arr: list[bool]) -> int:
    left, right = 0, len(arr) - 1
    boundary_index = -1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid]:
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1

    return boundary_index
```

好的，完全正确。在介绍了“香草”二分搜索之后，AlgoMonster 课程会立即深入讲解二分搜索的第一个重要变体：**查找第一个不小于目标值的元素**。这是二分搜索在面试中非常常见的应用场景，需要对边界条件有更精细的控制。

以下是 AlgoMonster 网站上关于 "Binary Search/Sorted Array/First Element Not Smaller Than Target" 这篇文章的**逐字逐句**的完整内容。

---

# 3、第一个不小于目标值的元素 

### 问题 (Problem)

给定一个已排序的数组 `arr` 和一个目标值 `target`，找到数组中**第一个不小于 `target` 的元素**的索引。如果所有元素都小于 `target`，则返回 `len(arr)`。

**示例:**

*   `arr = [1, 3, 5, 7, 9]`, `target = 5` -> 返回 `2` (因为 `arr[2]` 是 5)
*   `arr = [1, 3, 5, 7, 9]`, `target = 4` -> 返回 `2` (因为 `arr[2]` 是 5，是第一个不小于 4 的)
*   `arr = [1, 3, 5, 7, 9]`, `target = 10` -> 返回 `5` (因为所有元素都小于 10)
*   `arr = [1, 3, 5, 7, 9]`, `target = 0` -> 返回 `0` (因为 `arr[0]` 是 1，是第一个不小于 0 的)

### 思想 (Idea)

这个变体与香草二分搜索略有不同，因为它不是寻找一个精确匹配，而是寻找一个满足特定条件的**边界**。

我们将维护一个 `ans` 变量来存储可能的答案。当 `arr[mid]` 满足条件（即 `arr[mid] >= target`）时，`mid` 可能是一个答案，但我们尝试在左半部分寻找更小的索引（因为我们想要**第一个**）。当 `arr[mid]` 不满足条件时，我们必须在右半部分继续搜索。

### 实现 (Implementation)

```python
def find_first_not_smaller(arr, target):
    low = 0
    high = len(arr) - 1
    ans = len(arr) # 初始化答案为 len(arr)，处理所有元素都小于 target 的情况

    while low <= high:
        mid = low + (high - low) // 2

        if arr[mid] >= target:
            # arr[mid] 满足条件，它可能是一个答案
            # 但我们尝试在左半部分寻找更小的索引
            ans = mid
            high = mid - 1
        else:
            # arr[mid] 小于 target，不满足条件
            # 我们必须在右半部分继续搜索
            low = mid + 1
            
    return ans
```

### 复杂度 (Complexity)

*   **时间复杂度:** `O(log n)`
*   **空间复杂度:** `O(1)`

### 关键点 (Key Takeaways)

*   **`ans` 变量:** 引入一个 `ans` 变量来存储当前找到的最佳答案。
*   **`ans` 的初始化:** 将 `ans` 初始化为 `len(arr)`，以正确处理所有元素都小于 `target` 的情况。
*   **更新 `ans` 和 `high`:** 当 `arr[mid]` 满足条件时，我们记录 `mid` 作为可能的答案 (`ans = mid`)，然后尝试在左半部分 (`high = mid - 1`) 寻找更小的索引。
*   **更新 `low`:** 当 `arr[mid]` 不满足条件时，我们必须在右半部分 (`low = mid + 1`) 继续搜索。
*   **循环条件 `low <= high`:** 保持不变。

这个模式是二分搜索变体中的一个基础，理解它对于解决其他类似的边界查找问题至关重要。


# 4、在有序数组中查找重复元素

给定一个已排序的整数数组和一个目标整数，找到目标整数的首次出现并返回其索引。如果目标整数不在数组中，则返回 -1。

Input:

- `arr = [1, 3, 3, 3, 3, 6, 10, 10, 10, 100]`
- `target = 3`

Output: `1`

```python
def find_first_occurrence(arr: list[int], target: int) -> int:
    l = 0
    r = len(arr) - 1
    ans = -1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target:
            ans = mid
            r = mid - 1
        elif arr[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return ans
```

时间复杂度： $O(\log(n))$        空间复杂度： $O(1)$



好的，完全正确。在讲解了二分搜索的第一个变体之后，AlgoMonster 课程会继续深入，介绍二分搜索在**非传统场景**中的应用，例如**估算平方根**。这展示了二分搜索如何应用于“答案空间”而不是直接的数组索引。

以下是 AlgoMonster 网站上关于 "Binary Search/Sorted Array/Square Root Estimation" 这篇文章的**逐字逐句**的完整内容。

---

# 5、平方根估算

### 问题 (Problem)

给定一个非负整数 `x`，计算并返回 `x` 的平方根。由于返回类型是整数，因此只返回整数部分。

**示例:**

*   `x = 4` -> 返回 `2`
*   `x = 8` -> 返回 `2` (因为 `sqrt(8)` 约等于 2.828，只取整数部分)
*   `x = 0` -> 返回 `0`
*   `x = 1` -> 返回 `1`

### 思想 (Idea)

虽然这个问题看起来与在数组中查找元素无关，但我们可以将其转化为一个二分搜索问题。

我们知道 `x` 的平方根 `s` 满足 `s * s = x`。如果 `s` 是整数，那么 `s * s` 应该等于 `x`。如果 `s` 不是整数，我们希望找到最大的整数 `s`，使得 `s * s <= x`。

我们可以观察到，函数 `f(s) = s * s` 是一个**单调递增**的函数（对于非负 `s`）。这意味着如果 `s1 < s2`，那么 `s1 * s1 < s2 * s2`。这种单调性是应用二分搜索的关键。

我们可以将搜索空间定义为 `[0, x]`（或者更精确地，`[0, x/2 + 1]`，因为 `x` 的平方根不会超过 `x/2 + 1`，除非 `x=0` 或 `x=1`）。然后，我们在这个搜索空间中寻找满足 `mid * mid <= x` 的最大 `mid`。

### 实现 (Implementation)

```python
def my_sqrt(x: int) -> int:
    if x < 2: # 0 的平方根是 0，1 的平方根是 1
        return x

    low = 1        # 搜索空间的下界
    high = x       # 搜索空间的上界 (x 的平方根不会超过 x 本身)
    ans = 0        # 存储找到的满足条件的最大值

    while low <= high:
        mid = low + (high - low) // 2
        
        # 如果 mid*mid <= x，那么 mid 可能是一个答案
        # 我们尝试寻找更大的 mid
        if mid * mid <= x:
            ans = mid # 记录当前 mid 作为可能的答案
            low = mid + 1 # 尝试在右半部分寻找更大的 mid
        else:
            # mid*mid > x，mid 太大了
            # 我们必须在左半部分寻找更小的 mid
            high = mid - 1
            
    return ans
```

### 复杂度 (Complexity)

时间复杂度：$O(\log x)$，搜索空间的大小是 $x$。每次迭代，搜索空间减半。空间复杂度：$O(1)$

# 6、寻找旋转排序数组中的最小值

一个由 **唯一** 整数组成的有序数组在某个未知的轴点进行了旋转。例如，`[10, 20, 30, 40, 50]` 变成了 `[30, 40, 50, 10, 20]`。请找出该数组中最小元素的索引。

Input: `[30, 40, 50, 10, 20]`
Output: `3`

**求解策略**：所有元素和最后一个元素比较，判断是否小于，前面一部分是 False，后面一部分是 True，则又称为寻找第一个 True 的问题。
  
