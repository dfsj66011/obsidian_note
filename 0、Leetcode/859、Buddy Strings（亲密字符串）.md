
Tag: Easy、Hash Table、String

## 问题描述

给定两个字符串 `s` 和 `goal`。你需要判断是否可以通过在 `s` 中交换恰好两个字母来使 `s` 等于 `goal`。

交换操作是指在字符串 `s` 中选择两个不同的位置 `i` 和 `j`（其中 `i != j`），并交换这两个位置上的字符。

如果恰好一次交换可以将字符串 `s` 转换为目标字符串 `goal`，则该函数应返回 `true`，否则返回 `false`。

**要点：**

- 你必须执行一次交换（交换两个字符）
- 被交换的两个位置必须不同（`i != j`）
- 如果 `s` 已经等于 `goal`，你仍然需要执行一次交换 - 这种情况只有在存在可以交换而不改变字符串的重复字符时才有效

**示例：**

- 如果 `s = "ab"` 且 `goal = "ba"`，交换位置 0 和 1 可以得到 `"ba"`，因此返回 `true`。
- 如果 `s = "ab"` 且 `goal = "ab"`，它们已经相等但没有重复字符可供交换，因此返回 `false`。
- 如果 `s = "aa"` 且 `goal = "aa"`，它们相等且可以交换两个 'a' 而不改变字符串，因此返回 `true`。 
- 如果 `s = "abcd"` 且 `goal = "cbad"`，交换位置 0 和 2 可以将 `"abcd"` 转换为 `"cbad"`，因此返回 `true`。


## 直觉

要解决这个问题，让我们思考一下，必须满足哪些条件才能使恰好一次交换将字符串 `s` 转换为目标字符串 `goal`。

首先，两个字符串的长度必须相同——如果不同，任何单次交换都无法使它们相等。

其次，两个字符串必须包含相同频率的相同字符。交换只能重新排列 `s` 中的字符，不会添加或删除任何字符。因此，如果 `s` 的字符计数与 `goal` 不同，它们永远无法通过交换变得相等。

现在，假设字符串长度相同且字符频率一致，我们需要考虑两种情况：

**场景 1：字符串恰好有 2 个位置不同**   如果字符串 `s` 和目标字符串 `goal` 恰好有 2 个位置不同（假设为位置 `i` 和 `j`），那么交换 `s[i]` 和 `s[j]` 应该能使它们相等。为此，需要满足 `s[i] = goal[j]` 且 `s[j] = goal[i]` 。我们之前执行的字符频率检查已经确保了这一条件。

**情景 2：字符串已经相同**   如果 `s` 和 `goal` 已经相同，我们仍然需要执行一次交换。唯一可行的方法是 `s` 中至少有一个字符出现多次。我们可以交换同一个字符的两个出现位置，这样不会改变字符串，但满足了必须执行一次交换的要求。

如果字符串在 0 个位置上有差异（完全相同）但没有重复字符，或者它们在 1、3 或更多位置上有差异，那么通过恰好一次交换使它们相等是不可能的。

这促使我们计算不同位置的数量，并在必要时检查重复字符。

## 解决方案方法

让我们一步步来实施：

**步骤 1：检查长度是否相等**

```python
m, n = len(s), len(goal)
if m != n:
    return False
```

首先，我们检查两个字符串的长度是否相同。如果不同，那么通过一次交换不可能使它们相等。

**步骤 2：检查字符频率**

```python
cnt1, cnt2 = Counter(s), Counter(goal)
if cnt1 != cnt2:
    return False
```

我们使用 Python collections 模块中的 `Counter` 来统计两个字符串中每个字符的出现频率。如果字符频率不匹配，那么在字符串 `s` 内部无论如何交换字符都无法使其与 `goal` 相等。`Counter` 会创建一个类似字典的对象，其中键是字符，值是对应的出现次数。

**步骤 3：计算差异**

```python
diff = sum(s[i] != goal[i] for i in range(n))
```

我们计算字符串 `s` 和 `goal` 之间有多少个位置的字符不同。这里使用了带有 `sum()` 的生成器表达式来统计 `True` 值的数量（即字符不同的位置）。

**步骤 4：确定是否存在有效交换**

```python
return diff == 2 or (diff == 0 and any(v > 1 for v in cnt1.values()))
```

最终检查处理两种有效情况：

- `diff == 2`: 恰好有两个位置不同。由于我们已经验证过字符频率匹配，交换这两个位置将使字符串相等。
- `diff == 0 and any(v > 1 for v in cnt1.values())`: 字符串已经完全相同，但只要至少有一个字符出现超过一次，我们仍然可以进行有效的交换。我们通过查看频率计数器的值来检查是否有字符的计数大于 1。

该算法的时间复杂度为 `O(n)`，其中 `n` 为字符串长度，因为我们遍历字符串的次数是恒定的。空间复杂度为 `O(k)`，其中 `k` 为字符串中唯一字符的数量（用于计数器对象）。



### 示例演练

Let's walk through the solution with `s = "abcd"` and `goal = "cbad"`:

**Step 1: Check Length Equality**

- `len(s) = 4`, `len(goal) = 4` ✓
- Lengths match, continue to next step

**Step 2: Check Character Frequency**

- `Counter(s) = {'a': 1, 'b': 1, 'c': 1, 'd': 1}`
- `Counter(goal) = {'c': 1, 'b': 1, 'a': 1, 'd': 1}`
- Both counters are equal (same characters with same frequencies) ✓

**Step 3: Count Differences**

- Compare position by position:
    - Position 0: `s[0] = 'a'`, `goal[0] = 'c'` → Different
    - Position 1: `s[1] = 'b'`, `goal[1] = 'b'` → Same
    - Position 2: `s[2] = 'c'`, `goal[2] = 'a'` → Different
    - Position 3: `s[3] = 'd'`, `goal[3] = 'd'` → Same
- Total differences: `diff = 2`

**Step 4: Determine if Valid Swap Exists**

- Since `diff == 2`, we check if swapping positions 0 and 2 works:
    - `s[0] = 'a'` should become `goal[0] = 'c'`
    - `s[2] = 'c'` should become `goal[2] = 'a'`
    - After swapping positions 0 and 2: `"abcd"` → `"cbad"` ✓
- Return `true`

Let's also consider the edge case where `s = "aa"` and `goal = "aa"`:

**Steps 1-2:** Length and frequency checks pass

**Step 3:** Count differences = 0 (strings are identical)

**Step 4:** Check if we can still perform a valid swap:

- `diff == 0`, so check if any character appears more than once
- `Counter(s) = {'a': 2}`, and 2 > 1 ✓
- We can swap the two 'a's at positions 0 and 1, which keeps the string unchanged
- Return `true`

## 完整解决方案

```python
from collections import Counter
from typing import List

class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        """
        Check if we can swap exactly two characters in string s to make it equal to goal.
      
        Args:
            s: Source string
            goal: Target string to match after one swap
          
        Returns:
            True if exactly one swap can make s equal to goal, False otherwise
        """
        # Check if strings have the same length
        string_length = len(s)
        goal_length = len(goal)
      
        if string_length != goal_length:
            return False
      
        # Count character frequencies in both strings
        s_char_count = Counter(s)
        goal_char_count = Counter(goal)
      
        # If character frequencies don't match, strings can't be made equal
        if s_char_count != goal_char_count:
            return False
      
        # Count the number of positions where characters differ
        difference_count = sum(s[i] != goal[i] for i in range(string_length))
      
        # Two valid cases:
        # 1. Exactly 2 differences (swap those two positions)
        # 2. No differences but at least one duplicate character (swap two identical chars)
        return (difference_count == 2 or 
                (difference_count == 0 and any(count > 1 for count in s_char_count.values())))
```

## 时空复杂度

**时间复杂度: `O(n)`**

该算法执行以下操作：

- Creating `Counter(s)`: `O(n)` where n is the length of string s
- Creating `Counter(goal)`: `O(n)` where n is the length of string goal
- Comparing two Counter objects `cnt1 != cnt2`: `O(n)` in worst case (needs to check all unique characters)
- Computing differences using list comprehension: `O(n)` to iterate through all indices
- Checking if any value > 1 in `cnt1.values()`: `O(k)` where k is the number of unique characters, and k ≤ n

Since all operations are linear and performed sequentially, the overall time complexity is `O(n)`.

**空间复杂度: `O(k)`**

The algorithm uses additional space for:

- `Counter(s)`: `O(k₁)` where k₁ is the number of unique characters in s
- `Counter(goal)`: `O(k₂)` where k₂ is the number of unique characters in goal
- The generator expression for computing diff doesn't create additional space beyond `O(1)`

In the worst case where all characters are unique, k can be at most min(n, 26) for lowercase English letters. Since the alphabet size is constant (26 for lowercase letters), the space complexity can also be considered `O(1)` for practical purposes. However, if we consider the general case without alphabet constraints, the space complexity is `O(k)` where k ≤ n.


## 常见陷阱

### 陷阱 1：忽略字符串已经相等的情况

**The Problem:** Many developers initially write code that only checks if there are exactly 2 differences between the strings, forgetting that when `s` and `goal` are already identical, we still need to perform exactly one swap. This leads to incorrect handling of cases like:

- `s = "ab"`, `goal = "ab"` → Should return `False` (no duplicates to swap)
- `s = "aa"`, `goal = "aa"` → Should return `True` (can swap the two 'a's)

**Incorrect Implementation:**

```python
def buddyStrings(self, s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
  
    differences = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            differences.append(i)
  
    # WRONG: Only checking for 2 differences
    return len(differences) == 2
```

**The Solution:** Always handle both cases explicitly:

1. When there are exactly 2 differences
2. When there are 0 differences AND at least one duplicate character exists

```python
# Correct approach
difference_count = sum(s[i] != goal[i] for i in range(len(s)))
return (difference_count == 2 or 
        (difference_count == 0 and any(count > 1 for count in Counter(s).values())))
```

### Pitfall 2: Not Validating That the Two Differences Are Actually Swappable

**The Problem:** When finding exactly 2 differences, some implementations forget to verify that swapping those two positions would actually make the strings equal. Just having 2 differences doesn't guarantee a valid swap.

**Incorrect Implementation:**

```python
def buddyStrings(self, s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
  
    diff_positions = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            diff_positions.append(i)
  
    # WRONG: Not checking if the swap actually works
    if len(diff_positions) == 2:
        return True  # This assumes any 2 differences can be swapped
  
    return False
```

**The Solution:** Either explicitly check that swapping the two different positions produces the correct result, OR use character frequency counting to ensure the strings contain the same characters:

```python
# Solution 1: Explicit swap verification
if len(diff_positions) == 2:
    i, j = diff_positions[0], diff_positions[1]
    return s[i] == goal[j] and s[j] == goal[i]

# Solution 2: Character frequency validation (as in our main solution)
s_char_count = Counter(s)
goal_char_count = Counter(goal)
if s_char_count != goal_char_count:
    return False  # Different character sets means no valid swap exists
```

### Pitfall 3: Using Set to Check for Duplicates When Strings Are Equal

**The Problem:** When checking if identical strings have duplicate characters (to allow a valid swap), using `len(set(s)) < len(s)` seems elegant but can be inefficient for very long strings with many unique characters.

**Less Efficient Implementation:**

```python
if difference_count == 0:
    # Creates a full set of all characters
    return len(set(s)) < len(s)
```

**The Solution:** Use early termination with `any()` to stop as soon as a duplicate is found:

```python
# More efficient - stops as soon as a duplicate is found
if difference_count == 0:
    return any(count > 1 for count in Counter(s).values())
```

This approach is particularly beneficial when dealing with strings that have duplicates early in the character frequency distribution, as it doesn't need to process all unique characters.

---
