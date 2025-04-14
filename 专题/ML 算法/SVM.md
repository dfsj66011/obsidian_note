[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • Support Vector Machines (SVM)

- [Overview](https://aman.ai/primers/ai/support-vector-machines/#overview)
- [Key Concepts](https://aman.ai/primers/ai/support-vector-machines/#key-concepts)
    - [Hyperplane](https://aman.ai/primers/ai/support-vector-machines/#hyperplane)
        - [Equation of a Hyperplane](https://aman.ai/primers/ai/support-vector-machines/#equation-of-a-hyperplane)
    - [Margin](https://aman.ai/primers/ai/support-vector-machines/#margin)
        - [Functional Margin](https://aman.ai/primers/ai/support-vector-machines/#functional-margin)
    - [Support Vectors](https://aman.ai/primers/ai/support-vector-machines/#support-vectors)
    - [Optimization Problem](https://aman.ai/primers/ai/support-vector-machines/#optimization-problem)
- [Loss Function: Hinge Loss](https://aman.ai/primers/ai/support-vector-machines/#loss-function-hinge-loss)
- [Kernel Trick](https://aman.ai/primers/ai/support-vector-machines/#kernel-trick)
    - [Why Kernel Trick?](https://aman.ai/primers/ai/support-vector-machines/#why-kernel-trick)
        - [The Key Insight](https://aman.ai/primers/ai/support-vector-machines/#the-key-insight)
    - [Common Kernel Functions](https://aman.ai/primers/ai/support-vector-machines/#common-kernel-functions)
        - [Linear Kernel](https://aman.ai/primers/ai/support-vector-machines/#linear-kernel)
        - [Polynomial Kernel](https://aman.ai/primers/ai/support-vector-machines/#polynomial-kernel)
        - [Radial Basis Function (RBF) Kernel (Gaussian Kernel)](https://aman.ai/primers/ai/support-vector-machines/#radial-basis-function-rbf-kernel-gaussian-kernel)
        - [Sigmoid Kernel](https://aman.ai/primers/ai/support-vector-machines/#sigmoid-kernel)
    - [Advantages of Kernel Trick](https://aman.ai/primers/ai/support-vector-machines/#advantages-of-kernel-trick)
    - [Choosing the Right Kernel](https://aman.ai/primers/ai/support-vector-machines/#choosing-the-right-kernel)
- [Applications](https://aman.ai/primers/ai/support-vector-machines/#applications)
- [Pros and Cons](https://aman.ai/primers/ai/support-vector-machines/#pros-and-cons)
    - [Pros](https://aman.ai/primers/ai/support-vector-machines/#pros)
    - [Cons](https://aman.ai/primers/ai/support-vector-machines/#cons)
- [Example: SVM in Classification](https://aman.ai/primers/ai/support-vector-machines/#example-svm-in-classification)
    - [Given Dataset](https://aman.ai/primers/ai/support-vector-machines/#given-dataset)
    - [Model](https://aman.ai/primers/ai/support-vector-machines/#model)
- [Citation](https://aman.ai/primers/ai/support-vector-machines/#citation)

## Overview

- Support Vector Machines (SVMs) are a powerful supervised learning algorithm used for both classification and regression tasks. They are widely appreciated for their ability to handle high-dimensional data and robust performance, even in complex datasets.
- SVMs were first introduced by Vladimir Vapnik and Alexey Chervonenkis in 1963 as a binary classifier. The goal of SVM is to find the best hyperplane that separates data points of different classes in the feature space. For datasets that are not linearly separable, SVM employs kernel functions to map data into a higher-dimensional space where separation becomes feasible.

## Key Concepts

### Hyperplane

- A hyperplane is a decision boundary that separates data into different classes. In an nn-dimensional space, the hyperplane is an (n−1)(n−1)-dimensional subspace.

#### Equation of a Hyperplane

w⋅x+b=0w⋅x+b=0

- where:
    - ww: Weight vector (normal to the hyperplane)
    - xx: Feature vector
    - bb: Bias term
- The goal is to maximize the margin between the hyperplane and the nearest data points of any class (support vectors).

### Margin

- The margin is the distance between the hyperplane and the closest data points (support vectors) of either class. A larger margin generally leads to better generalization.

#### Functional Margin

yi(w⋅xi+b)≥1yi(w⋅xi+b)≥1

- where yi∈{−1,+1}yi∈{−1,+1}: True label of the ii-th data point.

### Support Vectors

- Support vectors are the data points that lie closest to the hyperplane and influence its orientation and position. These points are critical in defining the margin.

### Optimization Problem

- The optimization problem for a linear SVM is:

minw,b12‖w‖2minw,b12‖w‖2

- Subject to:

yi(w⋅xi+b)≥1∀iyi(w⋅xi+b)≥1∀i

- This is a convex optimization problem, solvable using techniques like Lagrange multipliers and quadratic programming.

## Loss Function: Hinge Loss

- The hinge loss function is used to penalize misclassified points and points within the margin: L=∑Ni=1max(0,1−yi(w⋅xi+b))L=∑i=1Nmax(0,1−yi(w⋅xi+b))
    - where NN is the total number of samples.
- The objective function becomes:
    
    minw,b12‖w‖2+C∑i=1Nmax(0,1−yi(w⋅xi+b))minw,b12‖w‖2+C∑i=1Nmax(0,1−yi(w⋅xi+b))
    
    - where CC is a regularization parameter controlling the trade-off between maximizing the margin and minimizing the classification error.

## Kernel Trick

- The kernel trick is a fundamental concept in SVMs that enables them to classify data that is not linearly separable in the original feature space. It works by implicitly mapping data into a higher-dimensional feature space using kernel functions, without explicitly performing the transformation. This makes computations efficient while still allowing complex decision boundaries.

### Why Kernel Trick?

- For non-linearly separable data, a linear hyperplane cannot effectively classify the data. Instead of manually adding more dimensions, the kernel trick allows SVM to compute the separation in a higher-dimensional space using only inner products.

#### The Key Insight

- If ϕ(x)ϕ(x) is the mapping to a higher-dimensional space, the dot product in the higher dimension can often be computed directly using a kernel function KK in the original space:

K(xi,xj)=ϕ(xi)⋅ϕ(xj)K(xi,xj)=ϕ(xi)⋅ϕ(xj)

- This avoids the explicit computation of ϕ(x)ϕ(x), saving time and resources.

### Common Kernel Functions

#### Linear Kernel

K(xi,xj)=x⊤ixjK(xi,xj)=xi⊤xj

- Suitable for linearly separable data.
- No explicit transformation is performed.

#### Polynomial Kernel

K(xi,xj)=(x⊤ixj+c)dK(xi,xj)=(xi⊤xj+c)d

- cc: A constant that controls the flexibility.
- dd: Degree of the polynomial.
- Captures polynomial relationships between features.

#### Radial Basis Function (RBF) Kernel (Gaussian Kernel)

K(xi,xj)=exp(−γ‖xi−xj‖2)K(xi,xj)=exp⁡(−γ‖xi−xj‖2)

- γγ: Parameter controlling the influence of each data point.
- Maps data into an infinite-dimensional space.
- Effective for highly non-linear relationships.

#### Sigmoid Kernel

K(xi,xj)=tanh(αx⊤ixj+c)K(xi,xj)=tanh⁡(αxi⊤xj+c)

- Inspired by neural networks.
- Parameters αα and cc control the shape of the decision boundary.

### Advantages of Kernel Trick

1. **Efficiency:**
    - Avoids the explicit computation of the high-dimensional feature space, reducing computational cost.
2. **Flexibility:**
    - Provides the ability to apply SVM to a wide range of non-linear problems.
3. **Scalability:**
    - Works for very high-dimensional data without increasing memory requirements significantly.

### Choosing the Right Kernel

- **Linear Kernel:** For linearly separable data or datasets with many features.
- **Polynomial Kernel:** When interactions of features up to a certain degree are relevant.
- **RBF Kernel:** Default choice for most non-linear datasets.
- **Sigmoid Kernel:** Use with caution; can behave unpredictably in certain datasets.

## Applications

1. **Classification Tasks:**
    - Handwriting recognition (e.g., MNIST dataset)
    - Text categorization (e.g., spam filtering)
    - Bioinformatics (e.g., cancer detection)
2. **Regression Tasks:**
    - Predicting real estate prices.
    - Financial forecasting.
3. **Outlier Detection:**
    - Detecting anomalies in time series data.
    - Fraud detection in financial systems.

## Pros and Cons

### Pros

1. **Effective in High Dimensions:** SVM works well in datasets with a large number of features.
    
2. **Robust to Overfitting:** Particularly effective in scenarios with a clear margin of separation.
    
3. **Versatility:** Through the use of kernels, SVM can handle nonlinear relationships.
    
4. **Sparse Solutions:** The model depends only on support vectors, reducing computation.
    

### Cons

1. **Scalability:** SVM is computationally expensive for large datasets due to quadratic programming.
    
2. **Choice of Kernel:** The performance heavily depends on the choice of the kernel function and its parameters.
    
3. **Interpretability:** Nonlinear SVM models are harder to interpret than linear models.
    
4. **Memory-Intensive:** Storing support vectors for large datasets can be memory-intensive.
    

## Example: SVM in Classification

### Given Dataset

- X={x1,x2,...,xN}X={x1,x2,...,xN}
    
- Labels: yi∈{−1,+1}yi∈{−1,+1}

### Model

- Using the RBF kernel: K(xi,xj)=exp(−γ‖xi−xj‖2)K(xi,xj)=exp⁡(−γ‖xi−xj‖2)
    
- Optimization: minw,b12‖w‖2+C∑Ni=1max(0,1−yi(w⋅xi+b))minw,b12‖w‖2+C∑i=1Nmax(0,1−yi(w⋅xi+b))
    
- After training, for a new point xnewxnew:
    
    f(xnew)=sign(∑Ni=1αiyiK(xi,xnew)+b)f(xnew)=sign(∑i=1NαiyiK(xi,xnew)+b)
    
    - where αiαi are the Lagrange multipliers obtained during optimization.

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledSVM,   title   = {Support Vector Machines},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)

