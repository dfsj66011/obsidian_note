
Tag：Medium、Array、Binary Search、Sorting

## 问题描述

你有 `n` 个篮子，它们沿着一条直线分布在不同位置，其中第 `i` 个篮子位于 `position[i]`。你需要将 `m` 个球放入这些篮子中。

当两个球被放入篮子时，它们会相互施加磁力。位于位置 `x` 和 `y` 的两个球之间的磁力定义为它们之间的绝对距离：`|x - y|`。

你的目标是将这些球分配到篮子中，使得任意两球之间的最小磁力最大化。换句话说，你希望放置这些球，使得最近的两个球之间的距离尽可能远。

给定：

* 一个整数数组 `position`，表示 `n` 个篮子的位置
* 一个整数 `m`，表示要放置的球的数量

返回任意两球间最小磁力的最大可能值。

例如，如果你的篮子位于位置 `[1, 2, 3, 4, 7]`，并且需要放置 3 个球，一种可能的排列方式是将球放在位置 1、4 和 7。这将产生磁力 `|1-4| = 3`、`|1-7| = 6` 和 `|4-7| = 3`。在这种排列中，最小磁力为 3。


## 直觉

关键洞见在于认识到这个问题具有单调性：随着我们增加球之间的最小要求距离，我们能成功放置的球的数量会减少。如果我们能以最小距离 `d` 放置 `m` 个球，那么我们肯定能以任何小于 `d` 的最小距离放置 `m` 个球（我们只是有更多的放置灵活性）。

这种单调关系表明可以在答案上使用 *二分查找*。我们不是直接尝试寻找最优放置方案，而是可以转换问题：给定一个特定的最小距离 `f`，我们能否放置所有 `m` 个球，使得每对球之间至少相距 `f`？

要判断给定的最小距离 `f` 是否可行，我们可以采用贪心算法：首先对篮子的位置进行排序，然后从左到右依次放置球，始终将下一个球放在距离前一个球至少 `f` 的最左侧可用篮子中。这种贪心策略之所以有效，是因为尽可能靠左放置球能为后续球的摆放提供最大的空间。

我们寻找答案的搜索空间是有限的。最小可能距离为 1（当球位于相邻位置时），最大可能距离不会超过篮子的总跨度（排序后的 `position[-1] - position[0]`）。我们在这个范围内进行二分搜索，以找到仍能放置所有 `m` 个球的最大最小距离。

该算法将优化问题（“最大化最小距离”）转化为决策问题（“我们能否达到至少这个最小距离？”），这样更容易解决，非常适合二分查找。


## 解决方案

该解决方案采用了二分查找与贪心验证函数相结合的方法。具体实现如下：

**1. 对位置数组进行排序**：首先，我们对篮子的位置进行排序，以便采用贪心放置策略。这样在放置球时，我们可以从左到右依次遍历。

**2. 设置二分查找边界：** 我们将搜索范围初始化为 `l = 1`（最小可能距离）和 `r = position[-1]`（最右侧位置，代表最大距离的上界）。

**3. 定义验证函数 `check(f)`：** 该函数判断我们是否可以放置 `m` 个球，且任意两个球之间的最小距离至少为 `f`。实现采用贪心算法：

* 将 `prev` 初始化为负无穷，以确保第一个球始终可以被放置
* 初始化一个计数器 `cnt = 0` 来追踪放置的球的数量
* 遍历每个篮子位置 `curr`
* 如果当前值与前一值的差大于等于 `f`，我们可以在此放置一个球。
	* 更新 `prev = curr`（将此标记为最后放置的位置）
	* 增加 `cnt += 1`
* 返回 `cnt < m`（如果无法放置所有 `m` 个球，则返回 `True`）

**应用二分查找：** 该解法使用 `bisect_left` 来寻找 `check` 函数从 False 转变为 True 的边界。由于 `check(f)` 在我们无法以距离 `f` 放置所有 `m` 个球时返回 `True`，因此我们需要寻找 `check` 返回 `False` 的最大值（这意味着我们可以放置所有球）。

`bisect_left(range(l, r + 1), True, key=check)` 这个调用会找到 `check` 返回 `True` 的最左侧位置，这意味着答案比这个值小一，也就是我们仍能放置所有 `m` 个球的最大距离。

**时间复杂度：** `O(n log n + n log(max_position))` where `n` is the number of baskets. The [sorting](https://algo.monster/problems/sorting_summary) takes `O(n log n)`, and the [binary search](https://algo.monster/problems/binary-search-speedrun) performs `O(log(max_position))` iterations, each requiring `O(n)` time to verify.

**空间复杂度：** `O(1)` excluding the [sorting](https://algo.monster/problems/sorting_summary) space, as we only use a few variables for the [binary search](https://algo.monster/problems/binary-search-speedrun) and verification.


### Example Walkthrough

Let's walk through a concrete example with baskets at positions `[1, 2, 3, 4, 7]` and we need to place `m = 3` balls.

**Step 1: Sort positions** The array is already sorted: `[1, 2, 3, 4, 7]`

**Step 2: Set binary search bounds**

- `l = 1` (minimum possible distance)
- `r = 7` (rightmost position)

**Step 3: Binary search iterations**

We'll binary search for the maximum minimum distance. Let's trace through the key iterations:

**Iteration 1:** Check `mid = 4`

- Can we place 3 balls with minimum distance 4?
- Start with `prev = -inf`, `cnt = 0`
- Position 1: `1 - (-inf) >= 4` ✓, place ball, `prev = 1`, `cnt = 1`
- Position 2: `2 - 1 = 1 < 4` ✗, skip
- Position 3: `3 - 1 = 2 < 4` ✗, skip
- Position 4: `4 - 1 = 3 < 4` ✗, skip
- Position 7: `7 - 1 = 6 >= 4` ✓, place ball, `prev = 7`, `cnt = 2`
- Result: Only placed 2 balls, need 3. Distance 4 is too large.

**Iteration 2:** Check `mid = 2`

- Start with `prev = -inf`, `cnt = 0`
- Position 1: `1 - (-inf) >= 2` ✓, place ball, `prev = 1`, `cnt = 1`
- Position 2: `2 - 1 = 1 < 2` ✗, skip
- Position 3: `3 - 1 = 2 >= 2` ✓, place ball, `prev = 3`, `cnt = 2`
- Position 4: `4 - 3 = 1 < 2` ✗, skip
- Position 7: `7 - 3 = 4 >= 2` ✓, place ball, `prev = 7`, `cnt = 3`
- Result: Successfully placed all 3 balls! Distance 2 is achievable.

**Iteration 3:** Check `mid = 3`

- Start with `prev = -inf`, `cnt = 0`
- Position 1: `1 - (-inf) >= 3` ✓, place ball, `prev = 1`, `cnt = 1`
- Position 2: `2 - 1 = 1 < 3` ✗, skip
- Position 3: `3 - 1 = 2 < 3` ✗, skip
- Position 4: `4 - 1 = 3 >= 3` ✓, place ball, `prev = 4`, `cnt = 2`
- Position 7: `7 - 4 = 3 >= 3` ✓, place ball, `prev = 7`, `cnt = 3`
- Result: Successfully placed all 3 balls! Distance 3 is achievable.

**Step 4: Binary search converges** The binary search continues narrowing the range. Since both distance 2 and 3 work, but distance 4 doesn't, the maximum achievable minimum distance is 3.

**Final Answer:** 3

The optimal placement is at positions 1, 4, and 7, giving distances:

- Between balls at 1 and 4: `|4 - 1| = 3`
- Between balls at 4 and 7: `|7 - 4| = 3`
- Between balls at 1 and 7: `|7 - 1| = 6`

The minimum distance is 3, which is the maximum possible for this configuration.

## 完整代码


```python
from typing import List
from math import inf
from bisect import bisect_left

class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        """
        Find the maximum minimum distance between m balls placed in given positions.
      
        Args:
            position: List of available positions for placing balls
            m: Number of balls to place
          
        Returns:
            Maximum possible minimum distance between any two balls
        """
      
        def can_place_balls(min_distance: int) -> bool:
            """
            Check if we can place m balls with at least min_distance apart.
          
            Args:
                min_distance: Minimum required distance between consecutive balls
              
            Returns:
                True if we CANNOT place m balls with given min_distance
                (returns True when placement fails for bisect_left to work correctly)
            """
            previous_position = -inf  # Initialize to negative infinity to place first ball
            balls_placed = 0
          
            # Greedily place balls from left to right
            for current_position in position:
                if current_position - previous_position >= min_distance:
                    # Can place a ball at current position
                    previous_position = current_position
                    balls_placed += 1
          
            # Return True if we cannot place all m balls (for bisect_left logic)
            return balls_placed < m
      
        # Sort positions to enable greedy placement
        position.sort()
      
        # Binary search bounds: minimum distance is 1, maximum is the span of positions
        left_bound = 1
        right_bound = position[-1] - position[0]
      
        # Find the first distance where we CANNOT place m balls
        # Then subtract 1 to get the maximum valid distance
        result = bisect_left(range(left_bound, right_bound + 1), True, key=can_place_balls)
      
        # Adjust result since we're looking for the last valid distance
        return left_bound + result - 1 if result > 0 else left_bound
```

## Time and Space Complexity

**Time Complexity:** `O(n × log n + n × log M)`

The time complexity consists of two main parts:

- `O(n × log n)` for sorting the `position` array, where `n` is the length of the position array
- `O(n × log M)` for the binary search operation, where:
    - The binary search runs `O(log M)` iterations, with `M = position[-1] - 1` being the search range (from 1 to the maximum position value)
    - Each iteration calls the `check` function, which iterates through all `n` positions in `O(n)` time
    - Combined: `O(n × log M)`

**Space Complexity:** `O(log n)`

The space complexity comes from:

- The sorting algorithm (typically Timsort in Python) uses `O(log n)` space for its recursion stack
- The `check` function uses `O(1)` additional space with only a few variables (`prev`, `cnt`, `curr`)
- The binary search itself uses `O(1)` space for variables `l` and `r`
- The dominant factor is the sorting's recursion stack: `O(log n)`

**Learn more about [how to find time and space complexity quickly](https://algo.monster/problems/runtime_summary).**

## Common Pitfalls

### 1. **Incorrect Binary Search Bounds**

A common mistake is setting the right bound as `position[-1]` (the maximum position value) instead of `position[-1] - position[0]` (the maximum possible distance between two balls). This leads to unnecessary iterations and potential incorrect results.

**Wrong:**

```python
right_bound = position[-1]  # This is a position, not a distance
```

**Correct:**

```python
right_bound = position[-1] - position[0]  # Maximum possible distance
```

### 2. **Off-by-One Error in Binary Search Result**

The trickiest part is understanding what `bisect_left` returns and how to interpret it. Since the `check` function returns `True` when we **cannot** place all balls, `bisect_left` finds the first distance where placement fails. The maximum valid distance is one less than this value.

**Wrong interpretation:**

```python
# Directly returning the bisect_left result
return bisect_left(range(left_bound, right_bound + 1), True, key=can_place_balls)
```

**Correct interpretation:**

```python
result = bisect_left(range(left_bound, right_bound + 1), True, key=can_place_balls)
return left_bound + result - 1  # Subtract 1 to get the last valid distance
```

### 3. **Greedy Placement Logic Error**

When checking if balls can be placed, always place the first ball at the first available position. A common mistake is trying to optimize the first ball placement, which breaks the greedy approach.

**Wrong:**

```python
def can_place_balls(min_distance):
    balls_placed = 1  # Assuming first ball is placed
    previous_position = position[0]
  
    for i in range(1, len(position)):  # Starting from index 1
        if position[i] - previous_position >= min_distance:
            previous_position = position[i]
            balls_placed += 1
  
    return balls_placed < m
```

**Correct:**

```python
def can_place_balls(min_distance):
    previous_position = -inf  # Allow first ball to be placed anywhere
    balls_placed = 0
  
    for current_position in position:  # Check all positions including first
        if current_position - previous_position >= min_distance:
            previous_position = current_position
            balls_placed += 1
  
    return balls_placed < m
```

### 4. **Confusing Return Logic in Check Function**

The check function's return value can be counterintuitive. It returns `True` when we **cannot** place all balls (for `bisect_left` to find the transition point), not when we can.

**Common confusion:**

```python
# Returning True when placement succeeds (wrong for bisect_left)
return balls_placed >= m  # This inverts the logic needed
```

**Correct:**

```python
# Return True when placement fails (correct for bisect_left)
return balls_placed < m
```

### Alternative Solution Using Standard Binary Search

To avoid the confusion with `bisect_left` and inverted logic, here's a clearer implementation using standard binary search:

```python
def maxDistance(self, position: List[int], m: int) -> int:
    def can_place_all_balls(min_distance: int) -> bool:
        """Returns True if we CAN place all m balls with min_distance apart."""
        balls_placed = 1
        last_position = position[0]
      
        for i in range(1, len(position)):
            if position[i] - last_position >= min_distance:
                balls_placed += 1
                last_position = position[i]
                if balls_placed == m:
                    return True
      
        return False
  
    position.sort()
    left, right = 1, position[-1] - position[0]
    result = 0
  
    while left <= right:
        mid = (left + right) // 2
        if can_place_all_balls(mid):
            result = mid  # Update result when valid
            left = mid + 1  # Try for larger distance
        else:
            right = mid - 1  # Distance too large, try smaller
  
    return result
```

This approach is more intuitive as the check function returns `True` for valid placements and we explicitly track the best result found.