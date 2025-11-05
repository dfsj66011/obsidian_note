
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
- [Kernel / Polynomial Trick](https://aman.ai/primers/ai/support-vector-machines/#kernel--polynomial-trick)
    - [Conceptual Overview](https://aman.ai/primers/ai/support-vector-machines/#conceptual-overview)
    - [Linear vs. Non-Linear Separability](https://aman.ai/primers/ai/support-vector-machines/#linear-vs-non-linear-separability)
- [The Kernel Trick in SVMs](https://aman.ai/primers/ai/support-vector-machines/#the-kernel-trick-in-svms)
    - [Why Use the Kernel Trick?](https://aman.ai/primers/ai/support-vector-machines/#why-use-the-kernel-trick)
    - [The Key Insight](https://aman.ai/primers/ai/support-vector-machines/#the-key-insight)
- [Common Kernel Functions](https://aman.ai/primers/ai/support-vector-machines/#common-kernel-functions)
    - [Linear Kernel](https://aman.ai/primers/ai/support-vector-machines/#linear-kernel)
    - [Polynomial Kernel](https://aman.ai/primers/ai/support-vector-machines/#polynomial-kernel)
    - [Radial Basis Function (RBF) or Gaussian Kernel](https://aman.ai/primers/ai/support-vector-machines/#radial-basis-function-rbf-or-gaussian-kernel)
    - [Sigmoid Kernel](https://aman.ai/primers/ai/support-vector-machines/#sigmoid-kernel)
- [Advantages of the Kernel Trick](https://aman.ai/primers/ai/support-vector-machines/#advantages-of-the-kernel-trick)
- [Choosing the Right Kernel](https://aman.ai/primers/ai/support-vector-machines/#choosing-the-right-kernel)
- [Applications](https://aman.ai/primers/ai/support-vector-machines/#applications)
- [Pros and Cons](https://aman.ai/primers/ai/support-vector-machines/#pros-and-cons)
    - [Pros](https://aman.ai/primers/ai/support-vector-machines/#pros)
    - [Cons](https://aman.ai/primers/ai/support-vector-machines/#cons)
- [Example: SVM in Classification](https://aman.ai/primers/ai/support-vector-machines/#example-svm-in-classification)
    - [Given Dataset](https://aman.ai/primers/ai/support-vector-machines/#given-dataset)
    - [Model](https://aman.ai/primers/ai/support-vector-machines/#model)
- [References](https://aman.ai/primers/ai/support-vector-machines/#references)
- [Citation](https://aman.ai/primers/ai/support-vector-machines/#citation)

## Overview

- Support Vector Machines (SVMs) are a powerful supervised learning algorithm used for both classification and regression tasks. They are widely appreciated for their ability to handle high-dimensional data and robust performance, even in complex datasets.
- SVMs were first introduced by Vladimir Vapnik and Alexey Chervonenkis in 1963 as a binary classifier. The goal of SVM is to find the best hyperplane that separates data points of different classes in the feature space. For datasets that are not linearly separable, SVM employs kernel functions to map data into a higher-dimensional space where separation becomes feasible.

## Key Concepts

### Hyperplane

- A hyperplane is a decision boundary that separates data into different classes. In an n-dimensional space, the hyperplane is an (n−1)-dimensional subspace.

#### Equation of a Hyperplane

w⋅x+b=0

- where:
    - w: Weight vector (normal to the hyperplane)
    - x: Feature vector
    - b: Bias term
- The goal is to maximize the margin between the hyperplane and the nearest data points of any class (support vectors).

### Margin

- The margin is the distance between the hyperplane and the closest data points (support vectors) of either class. A larger margin generally leads to better generalization.

#### Functional Margin

yi(w⋅xi+b)≥1

- where yi∈{−1,+1}: True label of the i-th data point.

### Support Vectors

- Support vectors are the data points that lie closest to the hyperplane and influence its orientation and position. These points are critical in defining the margin.

### Optimization Problem

- The optimization problem for a linear SVM is:

minw,b12‖w‖2

- Subject to:

yi(w⋅xi+b)≥1∀i

- This is a convex optimization problem, solvable using techniques like Lagrange multipliers and quadratic programming.

## Loss Function: Hinge Loss

- The hinge loss function is used to penalize misclassified points and points within the margin: L=∑Ni=1max(0,1−yi(w⋅xi+b))
    - where N is the total number of samples.
- The objective function becomes:
    
    minw,b12‖w‖2+C∑i=1Nmax(0,1−yi(w⋅xi+b))
    
    - where C is a regularization parameter controlling the trade-off between maximizing the margin and minimizing the classification error.

## Kernel / Polynomial Trick

### Conceptual Overview

- In classical machine learning, the goal of a classification task is to learn a function y=f(x;θ), that is, to fit a model (or a curve) to the data in such a way that a decision boundary separates the classes — for instance, deciding whether to approve or reject loan applications.
    
- The figure below, adapted from [Prithvi Da](https://www.linkedin.com/in/prithivirajdamodaran/), visually summarizes different approaches to achieving separability in data.
    

![](https://aman.ai/primers/ai/assets/kernel-trick/kernel-trick.jpeg)

### Linear vs. Non-Linear Separability

- **Fig A:** When data points are linearly separable, the problem is straightforward — algorithms such as logistic regression or a linear Support Vector Machine (SVM) can effectively fit a linear decision boundary.
    
- **Fig B:** However, when the data points are **not linearly separable**, a simple linear boundary cannot adequately classify the samples. While neural networks are theoretically capable of learning any Borel measurable function, they may be excessive for small or tabular datasets.
    
- **Fig C:** Instead, a simpler and elegant solution involves **transforming the input space** using a _non-linear mapping_. This transformation allows us to represent the same data in a higher-dimensional space where it becomes linearly separable. For example, transforming a line y=mx+c (a first-degree polynomial) into a parabola x2=4ay (a second-degree polynomial) can make classification feasible in the higher-dimensional feature space.
    
- This process of implicitly applying a non-linear transformation via a function known as a **kernel** forms the foundation of what is called the **kernel trick**. In essence, **learning a linear model in a higher-dimensional feature space is equivalent to learning a non-linear model in the original space**. In other words, by judiciously transforming the representation of the problem — figuratively “bending” it — one can often achieve a more effective and computationally elegant solution.
    

## The Kernel Trick in SVMs

- The **kernel trick** is a cornerstone concept in SVMs that enables them to classify data that is not linearly separable in the original feature space. It achieves this by implicitly mapping data into a higher-dimensional feature space using kernel functions — without explicitly performing the transformation. This allows complex decision boundaries to be computed efficiently.

### Why Use the Kernel Trick?

- For data that cannot be linearly separated in its original feature space, a simple hyperplane is insufficient. Instead of manually introducing additional dimensions, the kernel trick allows the SVM to **compute the necessary separation in a higher-dimensional space** using only inner products, greatly simplifying computation.

### The Key Insight

- If ϕ(x) represents the mapping from the input space to a higher-dimensional feature space, then the dot product in that space can be represented as a kernel function K:

K(xi,xj)=ϕ(xi)⋅ϕ(xj)

- This approach eliminates the need to explicitly compute ϕ(x), making training and inference computationally efficient.

## Common Kernel Functions

### Linear Kernel

K(xi,xj)=x⊤ixj

- Suitable for linearly separable data.
- No explicit transformation is performed.

### Polynomial Kernel

K(xi,xj)=(x⊤ixj+c)d

- c: A constant controlling flexibility.
- d: The degree of the polynomial.
- Captures polynomial relationships between features and mirrors the principle illustrated in Fig C above, where a polynomial transformation enhances separability.

### Radial Basis Function (RBF) or Gaussian Kernel

K(xi,xj)=exp(−γ|xi−xj|2)

- $$\gammah: Controls the influence of each data point.
- Maps data into an infinite-dimensional space.
- Highly effective for complex, non-linear relationships.

### Sigmoid Kernel

K(xi,xj)=tanh(αx⊤ixj+c)

- Inspired by neural network activation functions.
- Parameters α and c govern the decision boundary’s shape.
- Can approximate neural network behavior under specific parameter settings.

## Advantages of the Kernel Trick

1. **Efficiency:** Avoids explicit computation of the high-dimensional feature space, thereby reducing computational overhead.
    
2. **Flexibility:** Allows SVMs to handle both linear and highly non-linear datasets effectively.
    
3. **Scalability:** Can manage very high-dimensional data without significantly increasing memory requirements.
    

## Choosing the Right Kernel

- **Linear Kernel:** Best for linearly separable data or datasets with many features.
- **Polynomial Kernel:** Suitable when feature interactions up to a specific degree are significant.
- **RBF Kernel:** Often the default choice for non-linear datasets due to its flexibility.
- **Sigmoid Kernel:** Use cautiously; its performance can vary depending on data characteristics.

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

- X={x1,x2,...,xN}
    
- Labels: yi∈{−1,+1}

### Model

- Using the RBF kernel: K(xi,xj)=exp(−γ‖xi−xj‖2)
    
- Optimization: minw,b12‖w‖2+C∑Ni=1max(0,1−yi(w⋅xi+b))
    
- After training, for a new point xnew:
    
    f(\mathbf{x}_{\text{new}}) = \text{sign} \left\sum_{i=1}^N \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}_{\text{new}}) + b \right) $$
    
    - where αi are the Lagrange multipliers obtained during optimization.

## References

- [Prithvi Da on LinkedIn](https://www.linkedin.com/posts/prithivirajdamodaran_explain-polynomial-trick-to-5-year-olds-activity-6901421431121440768-i-Tu)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledSVM,   title   = {Support Vector Machines},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)