



Rotary Position Embedding (RoPE)[1] is a widely used positional encoding technique, which is utilized by many large language models such as Llama[2], Llama2[3], PaLM[4], CodeGen[5], and more.

Recently, I have carefully studied the paper[1] on RoPE and derived its formulas. I would like to share them here in the hope of helping readers understand this clever idea.

**This article mainly consists of three parts, including an introduction to the underlying principles, visual illustrations, and an analysis of the RoPE code in the Llama model.**

Florian’s Substack is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe

### Why do we need a positional encoding technique?

The Transformer model owes its remarkable performance to the essential Attention mechanism, which calculates the attention weights between each token in the input sequence.

Let’s assume a sequence has `N` tokens. The embeddings of the `m-th` token is `xm`, and the embeddings of the `n-th` token is `xn`.

Without adding position information to the word embeddings, we can transform them into queries `qm`, keys `kn`, and values `vn` as shown in equation (1):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6fbabdaf-3178-4668-823a-16a32b7540b5_800x207.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6fbabdaf-3178-4668-823a-16a32b7540b5_800x207.png)

The queries and keys are then used to compute the attention weights, while the output is computed as the weighted sum over the values, as shown in equation (2):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc356fac8-3053-4007-939e-27573958516a_800x275.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc356fac8-3053-4007-939e-27573958516a_800x275.png)

We discovered that when positional information is not included, the attention weight `a(m, n)` between tokens `xm` and `xn` remains constant regardless of their positions. In other words, the attention weight `a(m, n)` is position-independent, which goes against our intuition. For instance, the meanings of “dog bites cat” and “cat bites dog” are clearly distinct.

Furthermore, when two tokens are closer in distance, we expect the attention weight between them to be larger. Conversely, when the distance is greater, the attention weight should be smaller.

To resolve this issue, we can introduce positional encoding to the model. This allows each word embedding to incorporate information about its position in the input sequence. We define a function `f` to incorporate positional information `m` into word embedding `xm`, resulting in `qm`. Similarly, we incorporate positional information `n` into word embedding `xn`, resulting in `kn` and `vn`, as shown in equation (3):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb8b9ad-6c9f-4fc2-800e-2b1846f7a804_800x194.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb8b9ad-6c9f-4fc2-800e-2b1846f7a804_800x194.png)

After incorporating the position information, we can substitute equation (3) into equation (2) to introduce the position information in the attention mechanism. This is particularly important for tasks that are sensitive to position, such as NER (Named Entity Recognition).

### Core Idea of Rotary Position Embedding (RoPE)

RoPE aims to incorporate relative position information `(m — n)` into the inner product of `qm` and `kn` in equation(3).

How can we determine if it contains position information? It is sufficient to represent the inner product of `qm` and `kn` as a function `g(xm, xn, m-n)` of `xm`, `xn`, and `m-n`, where `m-n` represents the relative position information between the two vectors. Therefore, our modeling objective becomes finding a function `f` that satisfies the following equation(4):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe9a189bf-d096-42bb-b9e7-5c98316ff854_800x67.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe9a189bf-d096-42bb-b9e7-5c98316ff854_800x67.png)

In the attention mechanism, the interaction between tokens is implied in the dot product operation between query and key. If we define the dot product of `qm` and `kn` as a function of `m-n`, we can assign position information to each token by implementing absolute positional encoding using function `f`.

### How does RoPE find a function f that satisfies the conditions

**Currently, the only known information is equation (3) and (4), and nothing else is known.**

Finding a function f that satisfies a given condition is not an easy task in the vast space of functions. **A common approach when facing a difficult problem is to try to simplify it. First, consider the case of simplicity and clarity, and then generalize it to more complex situations.**

#### Step 1: Simplify the problem by assuming that the embedding dimension is 2.

The embedding dimension of LLMs is certainly much larger than 2, but we can generalize from this simple case.

In the 2D case, `qm` and `kn` are two-dimensional vectors. **For a 2D vector, we can view it as a complex number on the complex plane.** Therefore, `qm` and `kn` can be written in the form of complex numbers with their respective modulus and argument. Similarly, we can also express the inner product function g in the form of complex numbers, where `R` and `Θ` represent the modulus and argument respectively. This yields equation (5):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1109b651-2a40-468b-b705-84f146c0b9e8_800x137.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1109b651-2a40-468b-b705-84f146c0b9e8_800x137.png)

#### Step 2: Substituting equation (5) into equation (4)

We can obtain the following relationship:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28d38d4a-c14a-43fa-a588-40e135c6079e_800x172.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28d38d4a-c14a-43fa-a588-40e135c6079e_800x172.png)

#### Step 3: Calculate the modulus of function f based on equation (6)

For equation (6), let `m = n`, we obtain equation (8):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1120ce50-c583-40d0-9db4-bcc74d117dbb_800x61.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1120ce50-c583-40d0-9db4-bcc74d117dbb_800x61.png)

The reason why the second equal sign in Equation (8) holds is that for Equation (6), we can set `m = n = 0`.

Equation (8)’s final equality holds true due to the initial conditions (`m = 0, n = 0`) of equation (5), as shown in equation (9):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66ee02ad-b965-42cd-b5f3-fb1c79103892_800x114.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66ee02ad-b965-42cd-b5f3-fb1c79103892_800x114.png)

From equation (8), it can be seen that the modulus of function `f` is only related to the modulus of `qm` and `kn`, and is independent of the value of `m`. Therefore, let’s give a solution directly using the simplest relationship:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F80f37c80-9121-4a56-926c-2301fb8a3a12_800x130.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F80f37c80-9121-4a56-926c-2301fb8a3a12_800x130.png)

In this way, the modulus of function `f` is obtained. Next, we need to find the argument of the function `f`.

#### Step 4: Determine the argument of function f based on equation (7)

For equation (7), by setting `m = n`, we obtain equation (11):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F606b1d73-9273-4cef-9e1f-e1d715aa1100_800x92.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F606b1d73-9273-4cef-9e1f-e1d715aa1100_800x92.png)

The reason why the second equal sign in equation (11) holds is because for equation (7), we can set `m = n = 0`.

Equation (11)’s final equality holds true due to equation(9).

Rearrange according to equation (11):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb44d70ca-1550-43f0-a1bd-8e5eff71b06b_800x69.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb44d70ca-1550-43f0-a1bd-8e5eff71b06b_800x69.png)

Observing equation (12), it explains an important problem. **The values on both sides of equation (12) are only related to `m` and are independent of `x`.** Whether `x = xm` or `x = xn`, it remains the same. The left side of equation (12) can be denoted as:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F38506232-2ca0-405e-8191-9f2c2f6af652_800x81.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F38506232-2ca0-405e-8191-9f2c2f6af652_800x81.png)

Observing the relationship between `ϕ(m+1)` and `ϕ(m)`:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d136c9f-ec6d-4dfc-9e59-6bb9dc325691_800x136.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d136c9f-ec6d-4dfc-9e59-6bb9dc325691_800x136.png)

It can be seen that `ϕ(m)` is a function of `m`, while the value of `ϕ(m+1) — ϕ(m)` is independent of `m`. This indicates that `ϕ(m)` should be an arithmetic sequence with respect to `m` :

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb115e1c0-948c-4d99-897e-019332180501_800x97.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb115e1c0-948c-4d99-897e-019332180501_800x97.png)

It can be seen that step 4 is to prove that `{ϕ(`_`m`_`)}` is an arithmetic sequence.

#### Step 5: Finding the function f

Combining equations (10) and (15), we find that the modulus and argument of the function `f` have already been determined, which means we have found the function `f`.

Specifically, substituting equation (15) (for simplicity, setting `γ = 0`) and equation (10), (13) into equation (5):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71141435-432c-4b66-83b5-95b4db06208c_800x206.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71141435-432c-4b66-83b5-95b4db06208c_800x206.png)

#### Step 6: Determine q and the final result

A typical choice[6][7] of equation (3) is:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a73b008-0192-448c-bec4-68392126f74d_800x72.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a73b008-0192-448c-bec4-68392126f74d_800x72.png)

where `pm` is a vector depending of the position of token `xm` .

Recalling the definition of `q` in equation (9), it is defined for the case when `m = 0`. Here, we assume that there is no position information when `m = 0`, and this is also done to be compatible with equation (17). We directly define it as:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3ad99322-cd9a-490f-a72b-a1f10ebda33a_800x79.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3ad99322-cd9a-490f-a72b-a1f10ebda33a_800x79.png)

So the final result is:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5b85198-e55f-45cf-a3a4-b338dadf1a0f_800x118.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5b85198-e55f-45cf-a3a4-b338dadf1a0f_800x118.png)

We can substitute equation (19) into equation (10) to verify that it also holds true. Interested readers can calculate it themselves.

Write equation (19) in the form of a 2D matrix, where `Wq` is a 2x2 matrix, `xm` and `q` are 2D vectors:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa998b60b-8348-49e5-9f50-963eb011c7e6_800x162.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa998b60b-8348-49e5-9f50-963eb011c7e6_800x162.png)

**This is a vector rotation function, which means that by rotating the vector by an angle `mθ`, we can add absolute positional information to the vector. This is the origin of rotational position encoding. It is amazing how beautiful mathematics can be.**

### Visual Representation

To gain a better understanding of positional encoding in RoPE, the following description combines graphics to illustrate how to assign positional encoding to a two-dimensional embedding.

Assuming a 2D embedding `q = (1, 0)`, and the `θ` in equation(20) is a constant, let’s assume `θ = 1` in this case. When the token is located at position `m = [0, 1, 2, 3, 4, 5]`, corresponding positional information can be assigned to it, as shown in Figure 1:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5b191a5-0104-499c-b48f-ea7ce14e5f74_800x598.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5b191a5-0104-499c-b48f-ea7ce14e5f74_800x598.png)

Figure 1: RoPE in 2D. Image by author.

### Promotion to high-dimensional space

The previous content introduced how to assign position information to a two-dimensional vector, which can be achieved by rotating a certain angle. However, in practice, the dimensions of embeddings are usually in the hundreds or even thousands. Now, the question is how to extend the two-dimensional case to multiple dimensions.

The approach presented in the paper is quite straightforward. Typically, embedding dimensions are even numbers. Therefore, we decompose the high-dimensional vectors into pairs and rotate them individually. The rotation of the high-dimensional vectors can be represented as the following equations:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4b932740-3ca3-4c88-9e22-092a44e1a695_800x76.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4b932740-3ca3-4c88-9e22-092a44e1a695_800x76.png)

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff434bac2-66cd-40e5-9242-cabf71840c68_800x270.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff434bac2-66cd-40e5-9242-cabf71840c68_800x270.png)

**Equation (23)**

Here `θ` are all constants, and in the paper, they are directly assigned values, the inspiration may come from sinusoidal position encoding[6]:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7fab1d7-0dc8-4aab-abef-d5ad0726ef26_800x73.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7fab1d7-0dc8-4aab-abef-d5ad0726ef26_800x73.png)

where `d` is the embedding dimension.

Figure 2 shows the approach to dealing with high-dimensional situations:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40b600d4-9286-42ab-9b2a-c7a145809569_800x456.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40b600d4-9286-42ab-9b2a-c7a145809569_800x456.png)

Figure 2: Implementation of RoPE in high-dimensional situation. Source: [1]

### RoPE Implementation in Llama

The following code snippets are all from [the same file](https://github.com/facebookresearch/llama/blob/main/llama/model.py). I have added comments at the key code section.

#### precompute_freqs_cis function

```
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.



    """
    # Each group contains two components of an embedding,
    # calculate the corresponding rotation angle theta_i for each group.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Generate token sequence index m = [0, 1, ..., sequence_length - 1]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # Calculate m * theta_i
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cisp
```

This function is still quite abstract. Let me give an example to illustrate it, when `dim = 4`(the embedding dimension is `4`) and the sequence length is `3`, the generated `freqs_cis` would be:

```
tensor([[ 1.0000+0.0000j,  1.0000+0.0000j],
        [ 0.5403+0.8415j,  0.9999+0.0100j],
        [-0.4161+0.9093j,  0.9998+0.0200j]])
```

You can see in equation(25):

- `freqs_cis` has `3` components, corresponding to a sequence length of `3`.
    
- Each component is composed of two complex numbers.
    

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2d25b5a-b138-4c6f-bcc0-f48e5cea9cf3_800x162.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2d25b5a-b138-4c6f-bcc0-f48e5cea9cf3_800x162.png)

**Why is it necessary to calculate this form in advance**, you will see in the `apply_rotary_emb` function below.

#### apply_rotary_emb function

The function is to apply RoPE to input tensors. It first reshapes `xq` into two components per group, and then converts it into complex form as `xq_`.

**`xq_` is then multiplied with `freqs_cis` using multiplication of complex numbers.**

```
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    # Reshape and convert xq and xk to complex number
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Apply rotation operation, and then convert the result back to real numbers.
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

To explain the reason for using multiplication with complex numbers, let’s looking back at the previous high-dimensional rotation matrix(equation (23)), the rotation matrix is decomposed into `d/2` groups, with each group containing only two components. **Here, let’s take the example of `d = 4`.**

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97de59d2-5a43-4eec-bbdf-dc59465532c7_800x292.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97de59d2-5a43-4eec-bbdf-dc59465532c7_800x292.png)

For the 4-dimensional case, the calculation method of `apply_rotary_emb` is as follows: the multiplication of complex numbers is performed between the red boxes, and the multiplication of complex numbers is also performed between the green boxes. The complex form of the rotation matrix is provided by the pre-calculated `freqs_cis`, and the complex form of `q` is provided by `xq_`.

Why does the multiplication of complex numbers work?

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b5d1d32-1dc5-4c77-8c9d-355ff38b9b60_800x458.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b5d1d32-1dc5-4c77-8c9d-355ff38b9b60_800x458.png)

Figure 3: Implementation of RoPE, 4-dimensional scenario. Image by author.

As shown in Figure 3, this is because the multiplication result between the red boxes is given by(without loss of generality, let’s take the red box as an example here) equation(27):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa91d3dd6-5de6-4497-8f5a-afe0807926d4_800x60.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa91d3dd6-5de6-4497-8f5a-afe0807926d4_800x60.png)

The complex form of equation(27) is obtained by multiplying the following two complex numbers provided respectively by `xq_` and pre-calculated `freqs_cis`:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F024568cd-5e32-4286-ba6d-daf50e8af1bc_800x73.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F024568cd-5e32-4286-ba6d-daf50e8af1bc_800x73.png)

Similarly, the multiplication of complex numbers between the green boxes yields the last two dimensions of the first token’s `qm`. When combined with Equation (27), it forms the query embedding`qm` of the first token, as shown in equation(29):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecb4994b-ea37-47e8-b9ef-a20786187bb7_800x87.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecb4994b-ea37-47e8-b9ef-a20786187bb7_800x87.png)

**Equation (29)**

It can be seen that `precompute_freqs_cis` and `apply_rotary_emb` **cleverly achieve high-dimensional RoPE position encoding by** complex operations and the conversion between complex and real numbers.

#### Attention:: forward

Then, use `apply_rotary_emb` to calculate RoPE in the forward function of the `Attention` class.

```
class Attention(nn.Module):
    """Multi-head attention module."""

    ...
    ...

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Calculate RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        ...
        ...
```

### Conclusion

It is worth mentioning that **RoPE** was proposed in 2021 when its application was not as widespread. For example, Transformer used sinusoidal position encoding[6], and later the representative model BERT used learnable position embedding[7].

When RoPE-based large models like Llama became widely used that **it was discovered that** **RoPE could extrapolate position encoding beyond the pre-training length by using rotation matrices**. This improves the model’s generalization ability and robustness, which is not possible with previous position encoding methods. As a result, RoPE has been widely applied.

Overall, RoPE cleverly applies the idea of rotating vectors to position encoding in large language models, and it is implemented using complex operations. **It is a shining example of mathematical thinking in the field of artificial intelligence.**

Finally, if there are any errors or omissions in this article, please do not hesitate to point them out.

Florian’s Substack is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe

### References

[1]: J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu. [Roformer: Enhanced transformer with rotary position embedding](https://arxiv.org/pdf/2104.09864.pdf) (2021). arXiv preprint arXiv:2104.09864.

[2]: H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. [Llama: Open and efficient foundation language models](https://arxiv.org/pdf/2302.13971.pdf) (2023). arXiv preprint arXiv:2302.13971.

[3]: H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. [Llama 2: Open foundation and fine-tuned chat models](https://arxiv.org/pdf/2307.09288.pdf) (2023). arXiv preprint arXiv:2307.09288.

[4]: A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, H. Chung, C. Sutton, S. Gehrmann, P. Schuh, et al. [PaLM: Scaling language modeling with Pathways](https://arxiv.org/pdf/2204.02311v5.pdf) (2022). arXiv preprint arXiv:2204.02311.

[5]: E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, C. Xiong. [Codegen: An open large language model for code with multi-turn program synthesis](https://arxiv.org/pdf/2203.13474.pdf) (2022). arXiv preprint arXiv:2203.13474.

[6]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, L. Kaiser, I. Polosukhin. [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) (2017). arXiv preprint arXiv:1706.03762.

[7]: J. Devlin, M. Chang, K. Lee, K. Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (2019). arXiv preprint arXiv:1810.04805.

---

#### Subscribe to AI Exploration Journey

By Florian · Launched a year ago

A passionate AI researcher who enjoys delving into underlying principles and writing in-depth articles. With nearly 10K followers on Medium, I'm now gradually moving to Substack.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[

![Aniket Mishra's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F09cc3610-d14e-4716-9961-7fc4606859c0_96x96.png)



](https://substack.com/profile/165406715-aniket-mishra)

[

![Christopher's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fyellow.png)



](https://substack.com/profile/10734631-christopher)

[

![MuchSeeker's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7e697547-ee8e-46c3-843a-3c3dd943bc66_144x144.png)



](https://substack.com/profile/155965609-muchseeker)

[

![Daniel Kleine's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff200d88b-772c-4998-9230-3aa57b79ee37_650x472.png)



](https://substack.com/profile/129677294-daniel-kleine)

[

![Subhash Ramesh's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d6a14d3-3779-4f40-9204-3567f601d7c2_4284x4284.jpeg)



](https://substack.com/profile/39356021-subhash-ramesh)

5 Likes∙

[1 Restack](https://substack.com/note/p-144428516/restacks?utm_source=substack&utm_content=facepile-restacks)

5

- 

[](https://aiexpjourney.substack.com/p/an-in-depth-exploration-of-rotary-position-embedding-rope-ac351a45c794/comments)

1

Share

#### Discussion about this post

CommentsRestacks

![dfsj's avatar](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c03b8d8-032e-4d23-8164-a30abec05eb2_144x144.png)

TopLatestDiscussions

[Building LLM-Based Agents: From Toolbox to Autonomous Architect](https://aiexpjourney.substack.com/p/building-llm-based-agents-from-toolbox)

[Valuable Content and Insights from Anthropic’s Guide on Agents](https://aiexpjourney.substack.com/p/building-llm-based-agents-from-toolbox)

Dec 30, 2024 • 

[Florian](https://substack.com/@aiexpjourney)

10

[](https://aiexpjourney.substack.com/p/building-llm-based-agents-from-toolbox/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb9a20cf0-8c2f-4cbb-8e0a-b7cbb55fb308_2401x1000.webp)

[The Number of Parameters of GPT-4o and Claude 3.5 Sonnet](https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o)

[Recently, I saw a paper from Microsoft that surprisingly revealed the parameter counts for models such as GPT-4o and Claude 3.5 Sonnet.](https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o)

Jan 1 • 

[Florian](https://substack.com/@aiexpjourney)

7

[](https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffc06016f-c837-40fe-a589-44ea9aecb310_1792x1024.webp)

[AI Innovations and Insights 30: Agentic Reasoning and Key Features of Claude 3.7 Sonnet](https://aiexpjourney.substack.com/p/ai-innovations-and-insights-30-agentic)

[This article is the 30th in this deeply interesting series. In this post, we will explore two mind-opening topics:](https://aiexpjourney.substack.com/p/ai-innovations-and-insights-30-agentic)

Feb 28

7

[](https://aiexpjourney.substack.com/p/ai-innovations-and-insights-30-agentic/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6ee33b2a-8cf9-4b39-b608-8385cab518cb_2106x1166.png)

See all

Ready for more?

Subscribe

© 2025Florian June 

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture