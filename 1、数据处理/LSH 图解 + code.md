
> Pinecone 教程： [# Locality Sensitive Hashing (LSH): The Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)，
> [The full code is here.](https://github.com/pinecone-io/examples/blob/master/learn/search/faiss-ebook/locality-sensitive-hashing-traditional/testing_lsh.ipynb)


*局部敏感哈希（LSH）* 是一种广泛应用于 *近似* 最近邻（ANN）搜索的流行技术。LSH 是最初用于实现高质量搜索同时保持闪电般快速搜索速度的技术之一。在本文中，我们将深入探讨该算法背后的理论，并提供一个易于理解的 Python 实现！

## 一、搜索复杂度

想象一个包含数百万甚至数十亿样本的数据集——我们如何高效地比较所有这些样本？

即便使用最先进的硬件，对所有样本进行两两比较也是不现实的。这种方法的最佳时间复杂度也高达 $O(n^2)$。即使仅将单个查询与数十亿样本进行比对，其最佳时间复杂度仍为 $O(n)$。

我们还需考虑单一相似度计算背后的复杂性——每个样本都以向量形式存储，且通常是高维向量——这进一步增加了复杂度。

我们该如何避免这种情况？是否有可能实现低于线性复杂度的搜索？*是的，完全可以*！

解决方案是 *近似* 搜索。相较于逐一比较每个向量（*穷举* 搜索）——我们可以采用近似方法，将搜索范围限制在最相关的向量上。LSH 是一种能够提供亚线性搜索时间的算法。

## 二、LSH

我们需要一种方法来减少比较次数。理想情况下，希望仅对比那些可能匹配的向量——即 *候选对*。

LSH 包含多种不同方法。本文将重点介绍传统实现方案——其核心流程分为三步：构造特征矩阵（shingling）、最小哈希（MinHashing）以及最终的带状 LSH 函数计算。

本质上，最终的 LSH 函数使我们能够对同一样本进行多次分段和哈希处理。当我们发现一对向量 *至少有一次* 被哈希到相同的值时，就将它们标记为 *候选对* ——即 *潜在* 的匹配项。

这一过程与 Python 字典的实现机制非常相似。我们将键值对存入字典时，键会经过字典的哈希函数处理，被映射到特定的存储桶中，随后对应的值就会与该存储桶建立关联。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Ffaaae9e55d33b08629574243d8456446650df096-1400x787.png&w=3840&q=75)

典型的哈希函数旨在将不同的值（无论多么相似）分配到不同的存储桶中。

然而，这类哈希函数与 LSH 中使用的哈希函数存在一个关键区别。在使用字典时，我们的目标是将多个键值对映射到同一桶中的概率降至最低——即 *最小化冲突*。

LSH 则几乎相反。在 LSH 中，我们希望 *最大化碰撞次数*——尽管理想情况下仅针对 *相似* 输入发生。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F862f88182a796eb16942c47d93ee03ba4cdaee4d-1920x1080.png&w=3840&q=75)

LSH 函数的目标是将相似的值放入相同的桶中。

LSH 的哈希方法并不唯一。尽管所有变体都遵循 "*通过哈希函数将相似样本分入同一桶*" 的核心逻辑，但在此基础上的实现方式可能存在显著差异。

还有其他几种技术，例如随机投影（[Random Projection](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/) ），我们将在另一篇文章中详细介绍。

## 三、Shingling, MinHashing, 和 LSH

我们所探索的 LSH 方法包含三个步骤：首先通过 k-shingle（和 one-hot 编码）将文本转化为稀疏向量，接着利用 minhashing 生成"签名"——这些签名会被输入到 LSH 流程中，用于筛选出候选配对。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F89413953597fbfdd36c4fa77ca0eeafaf6cf944a-1280x980.png&w=3840&q=75)


### 3.1 k-Shingling

k-Shingling 过程类似于在文本字符串上滑动一个长度为 $k$ 的窗口，并在每一步截取片段。我们将所有这些片段汇集起来，形成我们的 shingles *集合*。

![[shingle.gif|500]]

```python
a = "flying fish flew by the space station"
b = "we will not allow you to bring your pet armadillo along"
c = "he figured a few sticks of dynamite were easier than a fishing pole to catch fish"

def shingle(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)

a = shingle(a, k)
b = shingle(b, k)
c = shingle(c, k)
print(a)

{'y ', 'pa', 'ng', 'yi', 'st', 'sp', 'ew', 'ce', 'th', 'sh', 'fe', ...}
```

这样，我们得到了 shingles。接下来，我们创建稀疏向量。为此，首先需要将所有集合合并，生成一个包含所有集合中所有 shingles 的大集合——我们称之为词汇表（或 vocab）。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F3b72d736d96c9da94344668c96a6b9b066522bb6-1280x720.png&w=3840&q=75)

我们使用该词汇表为每个集合创建稀疏向量表示。具体做法是：先创建一个与词汇表长度相同、全为零的空向量，然后检查哪些 shingle 出现在当前集合中。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fc259242606006f5c5505a6e677f3d05be75a26da-1280x720.png&w=3840&q=75)

（这其实并不是真正的 one-hot 向量）。

### 3.2 Minhashing

MinHash 能将稀疏向量转换为密集向量。我们拥有稀疏向量，所做的是为签名（即稠密向量）中的每个位置随机生成一个 minhash 函数。

因此，如果我们想生成一个由 20 个数字组成的密集向量/签名，就需要使用 20 个 minhash 函数。

现在，这些 minhash 函数本质上就是一组随机排列的数字序列——从 1 开始计数，直到词汇表的总长度（即 `len(vocab)`）。由于数字顺序经过随机打乱，我们可能会发现数字 1 出现在随机 minhash 函数的第 57 位（举例而言）。

我们的签名值是通过首先生成一个随机排列的计数向量，然后找到与稀疏向量中 1 的位置对齐的最小数字来创建的。

![[vocab_minhash.gif|400]]

重新排列索引顺序后，索引号为 1 的对应值为 0，继续，索引号为 2 处的对应值为 1，因此最小哈希输出结果为 2。

这就是我们生成 minhash 签名中一个值的方式。但我们需要生成 20个（或更多）这样的值。因此，我们为每个签名位置分配不同的 minhash 函数——并重复这一过程。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F866cea917043cfd7eb8221fc1a3b715a61e9d14f-1280x720.png&w=3840&q=75)

最终，我们生成了最小哈希签名——或者说稠密向量。

```python
from random import shuffle

# 1. 生成一个随机的 MinHash 向量。
hash_ex = list(range(1, len(vocab)+1))    # [1, 2, 3, 4, 5 ... 99, 100, 101, 102]
shuffle(hash_ex)                          # [63, 7, 94, 16, 36 ... 6, 55, 80, 56]

```

遍历这个随机化的 minhash 向量（从 1 开始），并将每个值的索引与稀疏向量中的对应值进行匹配。如果找到 1——该索引就是我们的签名值。

```python
for i in range(1, len(vocab)+1):
    idx = hash_ex.index(i)
    signature_val = a_1hot[idx]
    print(f"{i} -> {idx} -> {signature_val}")
    if signature_val == 1:
        print('match!')
        break

1 -> 58 -> 0
2 -> 19 -> 0
3 -> 96 -> 0
4 -> 92 -> 0
5 -> 83 -> 0
6 -> 98 -> 1
match!
```

通过多次迭代上述过程构建签名：

```python
def create_hash_func(size: int):
    # function for creating the hash vector/function
    hash_ex = list(range(1, len(vocab)+1))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size: int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes

# we create 20 minhash vectors
minhash_func = build_minhash_func(len(vocab), 20)


def create_hash(vector: list):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for func in minhash_func:
        for i in range(1, len(vocab)+1):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature

# now create signatures
a_sig = create_hash(a_1hot)
b_sig = create_hash(b_1hot)
c_sig = create_hash(c_1hot)

print(a_sig)
print(b_sig)

[44, 21, 73, 14, 2, 13, 62, 70, 17, 5, 12, 86, 21, 18, 10, 10, 86, 47, 17, 78]
[97, 96, 57, 82, 43, 67, 75, 24, 49, 28, 67, 56, 96, 18, 11, 85, 86, 19, 65, 75]
```

这就是 minhashing 的全部精髓——它的原理不过如此简单。我们将一个稀疏向量压缩成了一个更紧凑的 20 维数字签名。

### 3.3 从稀疏到签名的信息传递

信息是否确实在我们更大的稀疏向量和更小的密集向量之间得以保留？我们很难从这些新的密集向量中直观地识别出模式——但我们可以计算向量之间的相似度。

如果信息确实在我们缩减规模的过程中得以保留——那么向量之间的相似度也理应保持一致吧？

我们可以验证这一点。首先用 Jaccard 相似度计算 shingle 格式下句子的相似性，再对签名格式下的相同向量重复这一计算过程：

```python
def jaccard(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))

jaccard(a, b), jaccard(set(a_sig), set(b_sig))
(0.14814814814814814, 0.10344827586206896)

jaccard(a, c), jaccard(set(a_sig), set(c_sig))
(0.22093023255813954, 0.13793103448275862)

jaccard(b, c), jaccard(set(b_sig), set(c_sig))
(0.45652173913043476, 0.34615384615384615)
```

稀疏向量与签名之间的相似性信息似乎得到了保留！至此，我们已完全准备好进入 LSH 处理阶段。

## 四、Band 和 Hash

识别相似句子的最后一步是 LSH 函数本身。

我们将采用 banding 方法来实现 LSH——这可以说是传统方法。具体而言，该方法会对我们的签名进行哈希处理，对每个签名的分段进行哈希运算，并寻找哈希碰撞。

现在，如果我们要将这些向量整体进行哈希处理，可能难以构建一个能准确识别它们之间相似性的哈希函数——我们并不要求整个向量完全相等，只需部分相似即可。

在大多数情况下，即使两个向量的部分内容完全匹配——只要其余部分不相等，该函数仍可能将它们散列到不同的存储桶中。

我们不需要这样。我们希望具有某些相似性的签名能被哈希到同一个桶中，从而被识别为候选对。

### 4.1 工作原理

分带方法通过将向量分割成称为"带"（band）的子部分来解决这个问题。随后，我们不再将完整向量输入哈希函数处理，而是将每个向量带单独通过哈希函数进行运算。

假设我们将一个 100 维向量划分为 20 个波段，这样就有 20 次机会来识别向量间匹配的子向量。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F1b1fc2e08469ea024573078b275f7228e2e7d824-1920x1080.png&w=3840&q=75)

我们将签名分割为 $b$ 个子向量，每个子向量通过哈希函数（可使用单个哈希函数或 $b$ 个不同的哈希函数）处理，并映射到对应的哈希桶中。

我们现在可以添加一个更灵活的条件——给定任意两个子向量之间的碰撞，我们将相应的完整向量视为候选对。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F00a27d5963a54c82b9f751845218b6beb8c09324-1280x720.png&w=3840&q=75)

我们将签名分割为多个子向量。所有签名中对应的子向量必须通过相同的哈希函数进行处理。不过，并不需要为每个子向量使用不同的哈希函数（可以统一使用同一个哈希函数处理所有子向量）。

现在，我们只需两个向量的部分匹配即可将其视为候选。当然，这也会增加误报的数量（即被标记为候选匹配但实际并不相似的样本）。尽管如此，我们仍会尽可能减少这类情况的发生。

我们可以实现一个简单的版本。首先，我们将签名向量 a、b 和 c 进行分割：

```python
def split_vector(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature) / b)
    # code splitting signature in b parts
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i : i+r])
    return subvecs

band_a = split_vector(a_sig, 10)
band_b = split_vector(b_sig, 10)
band_b
[[42, 43],
 [69, 55],
 [29, 96],
 [86, 46],
 [92, 5],
 [72, 65],
 [29, 5],
 [53, 33],
 [40, 94],
 [96, 70]]

band_c = split_vector(c_sig, 10)
band_c
[[90, 43],
 [69, 55],
 [4, 101],
 [35, 15],
 [92, 22],
 [18, 65],
 [40, 18],
 [53, 33],
 [40, 94],
 [80, 14]]
```

然后我们遍历列表以识别子向量之间的任何匹配。如果发现任何匹配——我们将这些向量作为候选对。

```python
for b_rows, c_rows in zip(band_b, band_c):
    if b_rows == c_rows:
        print(f"Candidate pair: {b_rows} == {c_rows}")
        # we only need one band to match
        break
        
Candidate pair: [69, 55] == [69, 55]
```

我们发现，两个更相似的句子 b 和 c 被识别为候选对，而三者中相似度较低的 a 则未被识别为候选。这一结果令人满意，但若想真正测试局部敏感哈希（LSH）的性能，我们还需处理更多数据。


## 五、测试 LSH

迄今为止我们构建的实现效率非常低下——如果你想实现局部敏感哈希（LSH），这绝非正确方式。

但像这样逐步梳理代码——即便没有其他作用——至少能清晰地展示 LSH 的工作原理。不过接下来我们要处理更庞大的数据集，因此将用 Numpy 重写现有代码。

### 5.1 获取数据

首先，我们需要获取数据。这里包含多个专为相似性搜索测试构建的数据集。

```python
import requests
import pandas as pd
import io

url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt"

text = requests.get(url).text

data = pd.read_csv(io.StringIO(text), sep='\t')
data.head()
```

```
   pair_ID                                         sentence_A  \
0        1  A group of kids is playing in a yard and an ol...   
1        2  A group of children is playing in the house an...   
2        3  The young boys are playing outdoors and the ma...   
3        5  The kids are playing outdoors near a man with ...   
4        9  The young boys are playing outdoors and the ma...   

                                          sentence_B  relatedness_score  \
0  A group of boys in a yard is playing and a man...                4.5   
1  A group of kids is playing in a yard and an ol...                3.2   
2  The kids are playing outdoors near a man with ...                4.7   
3  A group of kids is playing in a yard and an ol...                3.4   
4  A group of kids is playing in a yard and an ol...                3.7   

  entailment_judgment  
0             NEUTRAL  
1             NEUTRAL  
2          ENTAILMENT  
3             NEUTRAL  
4             NEUTRAL  
```

```python
sentences = data['sentence_A'].tolist()
sentences[:3]
```

```
['A group of kids is playing in a yard and an old man is standing in the background',
 'A group of children is playing in the house and there is no man standing in the background',
 'The young boys are playing outdoors and the man is smiling nearby']
```

### 5.2 Shingles

```python
k = 8  # shingle size

# build shingles
shingles = []
for sentence in sentences:
    shingles.append(build_shingles(sentence, k))

# build vocab
vocab = build_vocab(shingles)

# one-hot encode our shingles
shingles_1hot = []
for shingle_set in shingles:
    shingles_1hot.append(one_hot(shingle_set, vocab))
# stack into single numpy array
shingles_1hot = np.stack(shingles_1hot)
shingles_1hot.shape          # (4500, 36466)

sum(shingles_1hot[0])        # confirm we have 1s， 73.0
```

### 5.3 MinHashing

```python
arr = minhash_arr(vocab, 100)

signatures = []

for vector in shingles_1hot:
    signatures.append(get_signature(arr, vector))

# merge signatures into single array
signatures = np.stack(signatures)
signatures.shape          # (4500, 100)
signatures[0]

array([  65,  438,  534, 1661, 1116,  200, 1206,  583,  141,  766,   92,
         52,    7,  287,  587,   65,  135,  581,  136,  838, 1293,  706,
         31,  414,  374,  837,   72, 1271,  872, 1136,  201, 1109,  409,
        384,  405,  293,  279,  901,   11,  904, 1480,  763, 1610,  518,
        184,  398,  128,   49,  910,  902,  263,   80,  608,   69,  185,
       1148, 1004,   90,  547, 1527,  139,  279, 1063,  646,  156,  357,
        165,    6,   63,  269,  103,   52,   55,  908,  572,  613,  213,
        932,  244,   64,  178,  372,  115,  427,  244,  263,  944,  148,
         55,   63,  232, 1266,  371,  289,  107,  413,  563,  613,   65,
        188])
```

我们将稀疏向量的长度从 36466 压缩至 100 维的特征签名。虽然维度差异显著，但正如前文所证实的，这种压缩技术能很好地保留相似性信息。

### 5.4 LSH

最后是 LSH 部分。我们将在此使用 Python 字典来哈希并存储候选对。

```python
b = 20

lsh = LSH(b)

for signature in signatures:
    lsh.add_hash(signature)
    
lsh.buckets

[{'65,438,534,1661,1116': [0],
  '65,2199,534,806,1481': [1],
  '312,331,534,1714,575': [2, 4],
  '941,331,534,466,75': [3],
  ...
  '5342,1310,335,566,211': [1443, 1444],
  '1365,722,3656,1857,1023': [1445],
  '393,858,2770,1799,772': [1446],
  ...}]
```

需要指出的是，我们的 lsh.buckets 变量实际上为每个波段都包含了一个独立的字典——不同波段的桶并不会混合在一起。

我们在桶中看到的是向量 ID（行号），因此要提取候选对，只需遍历所有桶并提取其中的配对即可。

```python
candidate_pairs = lsh.check_candidates()
len(candidate_pairs)      # 7243

list(candidate_pairs)[:5]
[(1646, 1687), (3234, 3247), (1763, 2235), (2470, 2622), (3877, 3878)]
```

在确定候选对后，我们将相似度计算限制在这些配对中——其中部分配对会符合相似度阈值，其余则不会。此处的目标是在保持高准确识别率的同时，缩小范围并降低搜索复杂度。

我们可以通过将候选对分类结果（1 或 0）与实际余弦（或 Jaccard）相似度进行对比，在此直观展示模型性能。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F547308d2f04bb52ab733485a0014696a9c7924d0-1280x720.png&w=3840&q=75)

现在，这看起来可能是一种奇怪的展示我们表现的方式——你说得对，确实如此——但我们这么做是有原因的。

### 5.5 优化 Bands

我们可以通过优化波段值 $b$ 来调整 LSH 函数的相似度阈值。该阈值决定了 LSH 函数何时将数据对从非候选状态切换为候选状态。$$p=1-(1-s^r)^b$$

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fb470799575b8e77911bacb8500977afef06d6c85-1280x720.png&w=3840&q=75)

候选分类（左侧 $y$ 轴）和计算概率 $P$（右侧 $y$ 轴）随相似度（计算所得或归一化余弦相似度）的变化关系。这表明我们计算的概率 $P$ 与相似度 $s$ 值反映了候选/非候选对的总体分布情况。参数 $b$ 与 $r$ 的取值分别为 20 和 5。

尽管对齐并不完美，但我们仍能观察到理论计算概率与实际候选对结果之间的相关性。现在，我们可以通过调整参数 $b$ 来左右移动不同相似度得分下返回候选对的概率。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Faace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png&w=3840&q=75)

这是我们计算得出的概率值。如果我们认为先前 $b=20$ 的结果要求相似度过高、难以将数据对判定为候选对，就会尝试将相似度阈值向左调整。

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F5402cb0b40da6b128a53f69fbcbd36a1ff8bdace-1280x720.png&w=3840&q=75)

当 $b$ 等于 25 时，真实结果与模拟结果分别以蓝色和品红色显示。我们之前的 LSH 结果（水蓝色）也一并展示以供对比。需要注意的是，此举生成了更多候选对。

由于我们现在返回了更多的候选对，这自然会导致更多的误报——即我们会将不相似的向量也作为“候选对”返回。这是调整参数 $b$ 不可避免的后果，我们可以将其可视化如下：

![|500](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd6b9466efa2e6875ff98f4cce94ae1737e36c53b-1280x720.png&w=3840&q=75)

