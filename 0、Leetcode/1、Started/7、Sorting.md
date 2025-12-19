
### 1、冒泡排序 (Bubble Sort)

**思想：** 重复地遍历列表，比较相邻的两个元素，如果它们的顺序错误，就交换它们。

```
第一轮
x x x x x x x x x x x
  <---- j ---------->            j 从 1 比较到最后

第二轮
x x x x x x x x x x Z
  <---- j -------->              j 从 1 比较到 n-1
  
第二轮
x x x x x x x x x Y Z
  <---- j ------>                j 从 1 比较到 n-2
```


```python
def bubble_sort(array: List[int]) -> List[int]:  
    """冒泡排序"""  
    len_array = len(array)  
    for i in range(len_array-1):  
        for j in range(1, len_array-i):  
            if array[j] < array[j-1]:  
                array[j], array[j-1] = array[j-1], array[j]  
    return array
```

**复杂度**

*   **时间复杂度:** $O(n^2)$
*   **空间复杂度:** $O(1)$
*   稳定 (Stable)、就地（In-place）

---

### 2、选择排序 (Selection Sort)

**思想：** 重复地从列表的未排序部分找到最小的元素，并将其放置在已排序部分的末尾。

```
第一轮：
x x x x x x x x x x x
i
  <-------- j ------>       # i=0, 只要比 arr[i] 小就交换
  
第二轮：
X x x x x x x x x x x
  i
    <-------j ------>       # i=1, 只要比 arr[1] 小就交换

第三轮：
X Y x x x x x x x x x
    i
      <-----j ------>       # i=2, 只要比 arr[2] 小就交换
```

```python
def selection_sort(array: List[int]) -> List[int]:  
    """选择排序"""  
    len_array = len(array)  
    for i in range(len_array-1):  
        for j in range(i, len_array):  
            if array[j] < array[i]:  
                array[j], array[i] = array[i], array[j]  
    return array
```

**复杂度**

*   **时间复杂度:** $O(n^2)$`
*   **空间复杂度:** $O(1)$
*   不稳定 (Not Stable)，就地（In-place）

---

### 3、插入排序 (Insertion Sort)

**思想:** 像打扑克牌一样，一次构建一个排好序的数组。它遍历输入元素，并通过将较大的元素向右移动，将每个元素插入到已排序数组中的正确位置。

```
第一轮：
sorted             unsorted
12                 11   13    5    6
j                  i
key = 11, 比较已排序区最后一个元素和 key 的大小，12 > 11, 所以将 12 后移
12，               12   13    5    6
j 继续左移 j=-1，出界到头了，则 arr[j+1] = key
  
第二轮：
sorted             unsorted
11  12             13    5    6
     j             i
key = 13, 比较已排序区最后一个元素和 key 的大小，12 < 13
则直接把 13 纳入已排序区即可，arr[j+1] = key

第三轮：
sorted             unsorted
11  12   13        5    6
          j        i
key = 5, 比较已排序区最后一个元素和 key 的大小，13 > 5, 所以将 13 后移，j 继续右移
11  12   13        13    6
    j              i
12 > 5, 所以将 12 右移
11  12   12        13    6
    j              i
....
继续右移
11  11   12        13    6
此时 j =-1,  arr[j+1] = key

第四轮

sorted             unsorted
5  11  12   13     6

key=6, 经过一些右移后：
5  11  11   12     13
j                  i
此时 5 < 6, arr[j+1] = key
```

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        # 将比 key 大的元素向右移动一位
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

**复杂度**

*   **时间复杂度:** $O(n^2)$
    *   最佳情况 (Best Case): $O(n)$ (当数组几乎已排序时)
*   **空间复杂度:** $O(1)$
*   **稳定性:** 稳定 (Stable)



### 4、归并排序 (Merge Sort)

归并排序是一种高效、通用、基于比较的排序算法。它是一个**分而治之**算法的例子。

1.  **分解 (Divide):** 将 n 个元素的序列递归地分成两半，直到每个子序列只包含一个元素（一个元素的列表被认为是已排序的）。
2.  **解决 (Conquer):** 递归地排序两个子序列。
3.  **合并 (Combine):** 合并两个已排序的子序列，以产生一个单一的、已排序的序列。

```python
def merge_sort(arr):
    if len(arr) > 1:
        # 找到中间点
        mid = len(arr) // 2
        
        # 分解成两半
        left_half = arr[:mid]
        right_half = arr[mid:]

        # 递归地对两半进行排序
        merge_sort(left_half)
        merge_sort(right_half)

        # 合并已排序的两半
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # 检查是否有任何剩余的元素
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
```

*   **时间复杂度:** $O(n log n)$
    *   **分解:** 将数组分成两半的操作需要 $O(log n)$ 的时间（递归树的高度）。
    *   **合并:** 在递归的每一层，我们都需要对总共 $n$ 个元素执行 $O(n)$ 的合并工作。
    *   因此，总时间复杂度是 $O(n log n)$。

*   **空间复杂度:** $O(n)$
    *   归并排序需要额外的空间来存储用于合并的临时数组。在最坏的情况下，它需要与原始数组大小相等的额外空间。

归并排序是一种**稳定**的排序算法。在合并步骤中，如果两个元素相等，我们会先取来自左边子数组的元素，从而保持了它们的原始相对顺序。

### 5、快速排序 (Quick Sort)

1.  选择基准 (Pick a pivot)：从数组中选择一个元素作为基准 (pivot)。
2.  分区 (Partition)：重新排列数组，使得所有小于基准的元素都移动到基准的左边，所有大于基准的元素都移动到基准的右边。完成分区后，基准元素就处于其最终的排序位置。
3.  递归 (Recurse): 递归地对基准左边和右边的两个子数组应用上述过程。

```
第一轮：
   low                     high
i  j
   10, 80, 30, 90, 40, 50, 70
                           pivot

i 表示的是迄今为止找到的 pivot 正确的位置，初始为 -1;
j 用于遍历元素，范围是 low 到 high
所以要将 j 处元素与 pivot 比较，如果小，则更新 i 的位置（i+1），指示 pivot
此时 ij 位置相同，元素交换，还是 10 和 10 自身交换
   ij
   10, 80, 30, 90, 40, 50, 70
                           pivot

第二轮：j 位于 80 > 70，j 继续后移

第三轮：j 位于 30 < 70，这说明又有一个新元素比 pivot 小了

   i       j
   10, 80, 30, 90, 40, 50, 70
                           pivot
                           
所以 i 指示的位置需要 +1，然后把 j 指示的元素换到这里来。

       i   j
   10, 30, 80, 90, 40, 50, 70
                           pivot
                           
继续这个过程，j 在 90 位置继续后移，j 在 40 位置，遇到小于 70 的，
更新 i，把 40 换到这里

           i       j
   10, 30, 40, 90, 80, 50, 70
                           pivot

再继续 j 在 50，比 70 小，更新 i，然后把 50 换过来               
               i       j
   10, 30, 40, 50, 80, 90, 70
                           pivot
                           
这就结束了，显而易见，最后 pivot 应该和 [i+1] 位置互换，
这样就实现了，pivot 左侧都比 pivot 小，右侧的都比它大。
然后对左右两侧再如此分别排序。

```



```python
def quick_sort(arr, low, high):
    if low < high:
        # pi 是分区索引，arr[pi] 现在在正确的位置
        pi = partition(arr, low, high)

        # 分别对分区之前和之后的元素进行排序
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    # 基准（我们选择最后一个元素）
    pivot = arr[high]
    
    # 较小元素的索引，表示迄今为止找到的 pivot 正确位置
    i = low - 1
    
    for j in range(low, high):
        # 如果当前元素小于或等于基准
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 示例用法:
# arr = [10, 7, 8, 9, 1, 5]
# n = len(arr)
# quick_sort(arr, 0, n - 1)
```


*   **时间复杂度:**
    *   **平均情况 (Average Case):** $O(n log n)$
        *   在平均情况下，分区步骤会将数组分成两个大小大致相等的子数组。
    *   **最坏情况 (Worst Case):** $O(n^2)$
        *   最坏情况发生在基准元素总是未排序数组中最小或最大的元素时。这会导致分区后的一个子数组包含 $n-1$ 个元素，而另一个为空。这种情况常见于数组已经排好序或逆序时。

*   **空间复杂度:**
    *   **平均情况:** $O(\log n)$ (用于递归调用栈)
    *   **最坏情况:** $O(n)$ (当递归树严重不平衡时)

快速排序是一种**不稳定**的排序算法。在分区过程中，具有相等键值的元素的相对顺序可能会改变。


| 特性 | 归并排序 (Merge Sort) | 快速排序 (Quick Sort) |
| :--- | :--- | :--- |
| **时间复杂度** | `O(n log n)` (最坏情况) | `O(n log n)` (平均), `O(n^2)` (最坏) |
| **空间复杂度** | `O(n)` | `O(log n)` (平均) |
| **稳定性** | 稳定 (Stable) | 不稳定 (Not Stable) |
| **原地排序** | 否 (No) | 是 (Yes) |
| **应用** | 适用于链表，外部排序 | 适用于数组，通常在实践中更快 |

-------


### 6、TimSort

TimSort 是一种混合排序算法，结合了 *归并排序* 和 *插入排序* 的思想。它被用作 Python（`sorted()`、`list.sort()`）和 Java（从 Java 7 开始用于对象数组的 `Arrays.sort()`）的默认排序算法。Timsort 的核心思想是识别数组中已排序的小片段（称为 "runs" ），然后高效地将它们合并以形成完全排序的数组。

主要分为三个步骤运行：

* 识别有序段：扫描数组以找出已经有序的小段（称为 "run"）。若某段为降序，则将其反转调整为升序。
* 排序短序列：若某段长度小于固定值（通常为 32），则使用插入排序对其进行排序——这种方法对小规模或接近有序的数据效率极高。
* 合并有序段：最后，Timsort 会按照保持合并平衡高效的规则（类似归并排序但针对现实数据优化）合并这些有序段。

详见 [TimSort](https://www.geeksforgeeks.org/dsa/timsort/)

