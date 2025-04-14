[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • N-Dimensional Tensor Product

- [Overview](https://aman.ai/primers/ai/matmul/#overview)
- [General Procedure](https://aman.ai/primers/ai/matmul/#general-procedure)
- [Implementation](https://aman.ai/primers/ai/matmul/#implementation)
- [References](https://aman.ai/primers/ai/matmul/#references)
- [Citation](https://aman.ai/primers/ai/matmul/#citation)

## Overview

- In this article, let’s go over the rules and procedure for an nn-dimensional tensor product, i.e., say A[a,b,c]×B[i,j,k]A[a,b,c]×B[i,j,k].

## General Procedure

- The general procedure is called [tensor contraction](http://en.wikipedia.org/wiki/Tensor_contraction). Concretely it’s given by summing over various indices. For example, just as ordinary matrix multiplication C=A×BC=A×B is given by,

cij=∑kaikbkjcij=∑kaikbkj

- We can contract by summing across any index. For example,
    
    cijlm=∑kaijkbklmcijlm=∑kaijkbklm
    
    - which gives a 4-tensor (“4-dimensional matrix”) rather than a 3-tensor.
- We can also contract twice, for example
    
    cil=∑j,kaijkbkjlcil=∑j,kaijkbkjl
    
    - which gives a 2-tensor.

## Implementation

- First, let’s look how matrix multiplication works. Say you have A[m,n]A[m,n] and B[n,p]B[n,p]. **A requirement for the multiplication to be valid is that the number of columns of AA must match the number of rows of BB.** Then, all you do is iterate over rows of AA `(i)` and columns of BB `(j)` and the common dimension of both `(k)` (matlab/octave example):

![](https://aman.ai/images/copy.png)

`m = 2 n = 3 p = 4 A = randn(m,n) B = randn(n,p) C = zeros(m,p)  for i = 1:m     for j = 1:p         for k = 1:n              C(i,j) = C(i,j) + A(i,k)*B(k,j)         end     end end  C-A*B %to check the code, should output zeros`

- Note that the common dimension nn got **“contracted”** in the process.
    
- Now, assuming you want something similar to happen in 3D case, i.e., one common dimension to contract, what would you do? Assume you have A[l,m,n]A[l,m,n] and B[n,p,q]B[n,p,q]. **The requirement of the common dimension is still there - the last one of A must equal the first one of B**. Then, nn just cancels in L×M×N×N×P×QL×M×N×N×P×Q and what you get is L×M×P×QL×M×P×Q, which is 4-dimensional. To compute it, just append two more for loops to iterate over new dimensions:
    

![](https://aman.ai/images/copy.png)

`l = 5 m = 2 n = 3 p = 4 q = 6 A = randn(l,m,n) B = randn(n,p,q) C = zeros(l,m,p,q)  for h = 1:l     for i = 1:m         for j = 1:p             for g = 1:q                 for k = 1:n                     C(h,i,j,g) = C(h,i,j,g) + A(h,i,k)*B(k,j,g)                 end             end         end     end end`

- At the heart of it, it is still row-by-column kind of operation (hence only one dimension “contracts”), just over more data.
    
- Now, let’s tackle the multiplication of A[m,n]A[m,n] and B[n,p,q]B[n,p,q], where the tensors have different ranks (i.e., number of dimensions), i.e, 2D×3D2D×3D, but it seems doable nonetheless (after all, matrix times vector is 2D×1D2D×1D). The result is C[m,p,q]C[m,p,q], as follows:
    

![](https://aman.ai/images/copy.png)

`m = 2 n = 3 p = 4 q = 5 A = randn(m,n) B = randn(n,p,q) C = zeros(m,p,q)  Ct=C for i = 1:m     for j = 1:p         for g = 1:q             for k = 1:n                 C(i,j,g) = C(i,j,g) + A(i,k)*B(k,j,g)             end         end     end end`

- which checks out against using the full for-loops:

![](https://aman.ai/images/copy.png)

`for j = 1:p     for g = 1:q         Ct(:,j,g) = A*B(:,j,g); %"true", but still uses for-loops     end end C-Ct`

## References

- [Is there a 3-dimensional “matrix” by “matrix” product?](https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledN-DimTensorProduct,   title   = {N-Dimensional Tensor Product},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)