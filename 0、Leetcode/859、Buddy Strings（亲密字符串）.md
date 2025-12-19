
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


# 1067. Digit Count in Range

Hard[Math](https://algo.monster/problems/math-basics)[Dynamic Programming](https://algo.monster/problems/dynamic_programming_intro)

[Leetcode Link](https://leetcode.com/problems/digit-count-in-range)

## Problem Description

You are given a single-digit integer `d` (from 0 to 9) and two integers `low` and `high`. Your task is to count how many times the digit `d` appears across all integers in the inclusive range `[low, high]`.

For example, if `d = 1`, `low = 1`, and `high = 13`, you need to count how many times the digit `1` appears in the numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.

- The digit `1` appears once in: 1, 10, 12, 13
- The digit `1` appears twice in: 11
- Total count = 1 + 1 + 2 + 1 + 1 = 6

The solution uses digit [dynamic programming](https://algo.monster/problems/dynamic_programming_intro) (digit DP) to efficiently count occurrences. The main function `digitsCount` calculates the answer by finding the count of digit `d` from 0 to `high`, then subtracting the count from 0 to `low - 1`.

The helper function `f(n, d)` counts occurrences of digit `d` in all numbers from 0 to `n`. It uses a recursive `dfs` function with memoization that:

- `pos`: tracks the current digit position being processed
- `cnt`: counts occurrences of digit `d` found so far
- `lead`: indicates if we're still in leading zeros
- `limit`: indicates if we're bounded by the original number's digits

The algorithm processes each digit position from most significant to least significant, building valid numbers while counting occurrences of the target digit `d`.

Quick Interview Experience

Help others by sharing your interview experience

Have you seen this problem before?

YesNo

## Intuition

The key insight is that counting digit occurrences in a range `[low, high]` can be transformed into a simpler problem: counting occurrences from `[0, high]` minus occurrences from `[0, low-1]`. This is a common pattern in range-based problems.

Why does this work? Because the count in range `[low, high]` equals all occurrences up to `high` minus all occurrences before `low`.

The challenging part is efficiently counting digit occurrences from 0 to a given number `n`. A brute force approach would iterate through every number and count digits, but this becomes inefficient for large ranges.

This is where digit DP comes in. Instead of generating every number, we can think about building numbers digit by digit. When constructing numbers from 0 to `n`, we observe patterns:

1. **Position matters**: We process digits from most significant to least significant position
2. **Bounded choices**: At each position, our digit choices are limited by whether we're still bounded by `n`'s corresponding digit
3. **Leading zeros**: Numbers like 001, 002 are actually just 1, 2 - we need to handle leading zeros specially
4. **State tracking**: As we build the number, we keep track of how many times we've seen our target digit `d`

The recursive approach naturally fits this problem because at each digit position, we make a choice (which digit to place), and this choice affects our future choices. The memoization (`@cache`) ensures we don't recalculate the same states repeatedly.

For example, when counting digit `1` from 0 to 234:

- At the hundreds place, we can choose 0, 1, or 2
- If we choose 2, at the tens place we can only choose 0-3 (bounded by 234)
- If we choose 1, we've found one occurrence and continue building
- If we choose 0, we're still in leading zeros (building numbers less than 100)

This systematic exploration of all valid number constructions while counting target digit occurrences gives us an efficient `O(log n)` solution.

**Learn more about [Math](https://algo.monster/problems/math-basics) and [Dynamic Programming](https://algo.monster/problems/dynamic_programming_intro) patterns.**

## Solution Approach

The solution implements digit [dynamic programming](https://algo.monster/problems/dynamic_programming_intro) with memoization. Let's break down the implementation:

**Main Function**: `digitsCount(d, low, high)`

- Returns `f(high, d) - f(low - 1, d)`
- This uses the prefix sum principle to get the count in range `[low, high]`

**Helper Function**: `f(n, d)` This function counts occurrences of digit `d` from 0 to `n`:

1. **Digit Extraction**: First, we extract digits of `n` into array `a`:
    
    ```python
    a = [0] * 11  # Array to store digits
    l = 0         # Length counter
    while n:
        l += 1
        a[l] = n % 10
        n //= 10
    ```
    
    For `n = 234`, we get `a = [0, 4, 3, 2, ...]` (stored in reverse, 1-indexed)
    
2. **Recursive DFS with Memoization**: The core logic is in the `dfs` function:
    
    ```python
    @cache
    def dfs(pos, cnt, lead, limit)
    ```
    
    **Parameters**:
    
    - `pos`: Current digit position (from most significant to least)
    - `cnt`: Count of target digit `d` found so far
    - `lead`: Boolean indicating if we're still in leading zeros
    - `limit`: Boolean indicating if we're bounded by the original number's digits
3. **Base Case**:
    
    ```python
    if pos <= 0:
        return cnt
    ```
    
    When all positions are processed, return the accumulated count.
    
4. **Upper Bound Determination**:
    
    ```python
    up = a[pos] if limit else 9
    ```
    
    - If `limit` is True, we can only use digits up to `a[pos]`
    - Otherwise, we can use any digit from 0 to 9
5. **Digit Choice Iteration**:
    
    ```python
    for i in range(up + 1):
        if i == 0 and lead:
            ans += dfs(pos - 1, cnt, lead, limit and i == up)
        else:
            ans += dfs(pos - 1, cnt + (i == d), False, limit and i == up)
    ```
    
    For each valid digit choice `i`:
    
    - **Leading Zero Case**: If `i == 0` and we're still in leading zeros (`lead == True`), we continue with leading zeros without counting this 0 as a digit
    - **Normal Case**: Otherwise, we:
        - Increment `cnt` if `i == d` (found our target digit)
        - Set `lead` to False (no longer in leading zeros)
        - Update `limit` based on whether we chose the maximum allowed digit

**Example Walkthrough** for counting digit `1` from 0 to 13:

- Extract digits: `a = [0, 3, 1]`, `l = 2`
- Start with `dfs(2, 0, True, True)`
- At position 2 (tens place):
    - Can choose 0 or 1 (limited by 1 in 13)
    - If choose 1: `dfs(1, 1, False, True)` (found one `1`, still limited)
    - If choose 0: `dfs(1, 0, True, False)` (leading zero, no longer limited)
- Continue recursively until all positions are processed

The `@cache` decorator memoizes results, preventing redundant calculations for repeated states, making the algorithm efficient with time complexity `O(log n × 10 × 2 × 2)` for each call to `f`.

# Ready to land your dream job?

#### Unlock your dream job with a 5-minute evaluator for a personalized learning plan!

### Example Walkthrough

Let's walk through counting digit `2` in the range `[15, 25]`.

**Step 1: Transform the Problem**

- We need `digitsCount(2, 15, 25)`
- This becomes: `f(25, 2) - f(14, 2)`
- Let's calculate each part

**Step 2: Calculate f(25, 2) - Count digit 2 from 0 to 25**

Extract digits of 25: `a = [0, 5, 2]`, length = 2

Start with `dfs(pos=2, cnt=0, lead=True, limit=True)`:

Position 2 (tens place):

- Upper bound is `a[2] = 2` (since limit=True)
    
- Try digit 0: Leading zero → `dfs(1, 0, True, False)`
    
    - This path will count 2's in numbers 0-9
    - At position 1, can use 0-9 (no limit)
    - When we choose 2: adds 1 to count
    - Result: 1 (just the number 2)
- Try digit 1: Not leading zero → `dfs(1, 0, False, False)`
    
    - This path counts 2's in numbers 10-19
    - At position 1, can use 0-9 (no limit)
    - When we choose 2: adds 1 to count
    - Result: 1 (just the number 12)
- Try digit 2: Found our target! → `dfs(1, 1, False, True)`
    
    - This path counts 2's in numbers 20-25
    - Already found one 2 (cnt=1)
    - At position 1, limited by `a[1] = 5`
    - Can choose 0,1,2,3,4,5
    - When we choose 2: adds another 1 (number 22 has two 2's)
    - Result: 1 + 6 = 7 (one 2 in tens place for all 20-25, plus one more in 22)

Total for f(25, 2) = 1 + 1 + 7 = 9

**Step 3: Calculate f(14, 2) - Count digit 2 from 0 to 14**

Extract digits of 14: `a = [0, 4, 1]`, length = 2

Start with `dfs(pos=2, cnt=0, lead=True, limit=True)`:

Position 2 (tens place):

- Upper bound is `a[2] = 1` (since limit=True)
    
- Try digit 0: Leading zero → `dfs(1, 0, True, False)`
    
    - Counts 2's in numbers 0-9
    - Result: 1 (just the number 2)
- Try digit 1: Not leading zero → `dfs(1, 0, False, True)`
    
    - Counts 2's in numbers 10-14
    - At position 1, limited by `a[1] = 4`
    - Can choose 0,1,2,3,4
    - When we choose 2: adds 1 to count
    - Result: 1 (just the number 12)

Total for f(14, 2) = 1 + 1 = 2

**Step 4: Final Answer**

- `digitsCount(2, 15, 25) = f(25, 2) - f(14, 2) = 9 - 2 = 7`

The digit 2 appears 7 times in the range [15, 25]:

- Once each in: 20, 21, 23, 24, 25
- Twice in: 22

This demonstrates how digit DP efficiently counts occurrences by building numbers digit by digit, tracking whether we're bounded by the original number and handling leading zeros appropriately.

## Solution Implementation

- Python
- Java
- C++
- TypeScript

```python
from functools import cache
from typing import List

class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:
        """
        Count how many times digit 'd' appears in all numbers from 'low' to 'high' (inclusive).
        Uses the principle: count(low, high) = count(0, high) - count(0, low-1)
        """
        return self.count_digit_occurrences(high, d) - self.count_digit_occurrences(low - 1, d)

    def count_digit_occurrences(self, upper_bound: int, target_digit: int) -> int:
        """
        Count occurrences of 'target_digit' in all numbers from 0 to 'upper_bound'.
        Uses digit DP technique to efficiently count without iterating through all numbers.
        """

        @cache
        def digit_dp(position: int, digit_count: int, has_leading_zeros: bool, is_bounded: bool) -> int:
            """
            Dynamic programming function to count digit occurrences.

            Args:
                position: Current digit position (1-indexed from right)
                digit_count: Count of target digit found so far
                has_leading_zeros: True if we haven't placed any non-zero digit yet
                is_bounded: True if we're still bounded by the original number's digits

            Returns:
                Total count of target digit occurrences for all valid numbers from this state
            """
            # Base case: finished processing all digits
            if position <= 0:
                return digit_count

            # Determine the maximum digit we can place at this position
            max_digit = digits_array[position] if is_bounded else 9

            total_count = 0

            # Try each possible digit at current position
            for current_digit in range(max_digit + 1):
                if current_digit == 0 and has_leading_zeros:
                    # Still in leading zeros, don't count this zero
                    total_count += digit_dp(
                        position - 1,
                        digit_count,
                        True,  # Still has leading zeros
                        is_bounded and (current_digit == max_digit)
                    )
                else:
                    # Either non-zero digit or zero after first non-zero digit
                    new_count = digit_count + (1 if current_digit == target_digit else 0)
                    total_count += digit_dp(
                        position - 1,
                        new_count,
                        False,  # No more leading zeros
                        is_bounded and (current_digit == max_digit)
                    )

            return total_count

        # Extract digits from the number (stored in reverse order for easier access)
        digits_array: List[int] = [0] * 11  # Support up to 10-digit numbers
        num_digits = 0
        temp_num = upper_bound

        while temp_num > 0:
            num_digits += 1
            digits_array[num_digits] = temp_num % 10
            temp_num //= 10

        # Handle edge case where upper_bound is 0
        if upper_bound == 0:
            return 1 if target_digit == 0 else 0

        # Start the digit DP from the most significant digit
        return digit_dp(num_digits, 0, True, True)
```

## Time and Space Complexity

**Time Complexity:** `O(log(high) * 10 * log(high))` which simplifies to `O(log²(high))`

The time complexity is determined by the digit DP (dynamic programming) approach:

- The number of digits in the maximum number is `O(log(high))`
- For each position `pos` from 1 to the number of digits, we iterate through at most 10 possible digits (0-9)
- Each state `(pos, cnt, lead, limit)` is computed at most once due to memoization
- The number of unique states is bounded by:
    - `pos`: `O(log(high))` positions
    - `cnt`: `O(log(high))` possible count values (at most the number of digits)
    - `lead`: 2 possible values (True/False)
    - `limit`: 2 possible values (True/False)
- Total states: `O(log(high) * log(high) * 2 * 2)` = `O(log²(high))`
- Each state does `O(10)` work iterating through digits
- Overall: `O(log²(high))` considering the constant factor of 10

**Space Complexity:** `O(log²(high))`

The space complexity consists of:

- Recursion stack depth: `O(log(high))` - maximum depth equals the number of digits
- Memoization cache: `O(log²(high))` - storing all possible states as analyzed above
- Array `a` for storing digits: `O(11)` = `O(1)` constant space
- The dominant factor is the memoization cache, giving us `O(log²(high))` space complexity

**Learn more about [how to find time and space complexity quickly](https://algo.monster/problems/runtime_summary).**

## Common Pitfalls

### 1. **Incorrect Handling of Leading Zeros When Counting Zero**

**The Pitfall**: When counting occurrences of digit `0`, leading zeros should NOT be counted, but zeros that appear after the first non-zero digit SHOULD be counted. Many implementations incorrectly handle this distinction.

**Example Problem**: Count occurrences of `0` from 1 to 105

- Number `105` has one `0` (the middle digit)
- Number `10` has one `0`
- Numbers like `001`, `002` don't exist (leading zeros don't form valid numbers)

**Incorrect Implementation**:

```python
# WRONG: This might count leading zeros
if current_digit == 0:
    total_count += digit_dp(position - 1, digit_count + 1, ...)  # Always counting zeros
```

**Correct Implementation**:

```python
if current_digit == 0 and has_leading_zeros:
    # Leading zero - don't count it
    total_count += digit_dp(position - 1, digit_count, True, ...)
else:
    # Either non-zero OR zero after first non-zero digit - count normally
    new_count = digit_count + (1 if current_digit == target_digit else 0)
    total_count += digit_dp(position - 1, new_count, False, ...)
```

### 2. **Edge Case: When Upper Bound is 0**

**The Pitfall**: The digit extraction loop `while temp_num > 0` won't execute when the upper bound is 0, leaving `num_digits = 0`. This causes the DP to return 0 even when counting occurrences of digit `0` from 0 to 0 (which should return 1).

**Incorrect Handling**:

```python
def count_digit_occurrences(self, upper_bound: int, target_digit: int) -> int:
    # ... digit extraction ...
    while temp_num > 0:
        num_digits += 1
        digits_array[num_digits] = temp_num % 10
        temp_num //= 10

    # If upper_bound is 0, num_digits stays 0, causing incorrect result
    return digit_dp(num_digits, 0, True, True)
```

**Correct Solution**:

```python
def count_digit_occurrences(self, upper_bound: int, target_digit: int) -> int:
    # ... digit extraction ...

    # Handle edge case where upper_bound is 0
    if upper_bound == 0:
        return 1 if target_digit == 0 else 0

    return digit_dp(num_digits, 0, True, True)
```

### 3. **Off-by-One Error in Range Calculation**

**The Pitfall**: Forgetting to subtract 1 when calculating the lower bound, resulting in counting the range `[0, low]` instead of `[0, low-1]`.

**Incorrect**:

```python
def digitsCount(self, d: int, low: int, high: int) -> int:
    return self.f(high, d) - self.f(low, d)  # WRONG: excludes 'low' from the range
```

**Correct**:

```python
def digitsCount(self, d: int, low: int, high: int) -> int:
    return self.f(high, d) - self.f(low - 1, d)  # Correct: includes 'low' in the range
```

### 4. **Incorrect Limit Flag Propagation**

**The Pitfall**: Not properly updating the `limit` flag when recursing. The limit should only remain `True` if we choose the maximum allowed digit at the current position.

**Incorrect**:

```python
# WRONG: Always passing the same limit flag
total_count += digit_dp(position - 1, new_count, False, is_bounded)
```

**Correct**:

```python
# Correct: Update limit based on whether we chose the maximum allowed digit
total_count += digit_dp(
    position - 1,
    new_count,
    False,
    is_bounded and (current_digit == max_digit)
)
```

### 5. **Array Size and Indexing Issues**

**The Pitfall**: Using 0-based indexing when the algorithm expects 1-based indexing for digit positions, or allocating insufficient array size for large numbers.

**Prevention**:

- Always allocate enough space: `digits_array = [0] * 11` for 10-digit numbers
- Use 1-based indexing consistently: digits are stored from index 1 to `num_digits`
- The digit at position `pos` represents the `pos`-th digit from the right (1-indexed)