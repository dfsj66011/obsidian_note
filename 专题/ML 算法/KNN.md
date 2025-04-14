[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • K-Nearest Neighbors

- [K-Nearest Neighbors (kNN)](https://aman.ai/primers/ai/k-nearest-neighbors/#k-nearest-neighbors-knn)
    - [Overview](https://aman.ai/primers/ai/k-nearest-neighbors/#overview)
    - [Algorithm Description](https://aman.ai/primers/ai/k-nearest-neighbors/#algorithm-description)
        - [Training Phase](https://aman.ai/primers/ai/k-nearest-neighbors/#training-phase)
        - [Prediction Phase](https://aman.ai/primers/ai/k-nearest-neighbors/#prediction-phase)
    - [Distance Metrics](https://aman.ai/primers/ai/k-nearest-neighbors/#distance-metrics)
    - [Parameters](https://aman.ai/primers/ai/k-nearest-neighbors/#parameters)
    - [Stopping Criterion](https://aman.ai/primers/ai/k-nearest-neighbors/#stopping-criterion)
    - [Advantages](https://aman.ai/primers/ai/k-nearest-neighbors/#advantages)
    - [Disadvantages](https://aman.ai/primers/ai/k-nearest-neighbors/#disadvantages)
    - [Example (Classification)](https://aman.ai/primers/ai/k-nearest-neighbors/#example-classification)
        - [Dataset](https://aman.ai/primers/ai/k-nearest-neighbors/#dataset)
        - [Test Point: (5,5)(5,5)](https://aman.ai/primers/ai/k-nearest-neighbors/#test-point-5-5)
    - [FAQ: What are Some Methods That Do Not Require a Predefined kk Value for Nearest Neighbor Selection?](https://aman.ai/primers/ai/k-nearest-neighbors/#faq-what-are-some-methods-that-do-not-require-a-predefined-k-value-for-nearest-neighbor-selection)
        - [Distance-Weighted KNN](https://aman.ai/primers/ai/k-nearest-neighbors/#distance-weighted-knn)
        - [Radius-Based Neighbors](https://aman.ai/primers/ai/k-nearest-neighbors/#radius-based-neighbors)
        - [Adaptive kk-Nearest Neighbors](https://aman.ai/primers/ai/k-nearest-neighbors/#adaptive-k-nearest-neighbors)
        - [Local Outlier Factor (LOF) or Similar Techniques](https://aman.ai/primers/ai/k-nearest-neighbors/#local-outlier-factor-lof-or-similar-techniques)
        - [Ensemble of KNN Models](https://aman.ai/primers/ai/k-nearest-neighbors/#ensemble-of-knn-models)
        - [Dynamic KNN](https://aman.ai/primers/ai/k-nearest-neighbors/#dynamic-knn)
        - [Data-Driven Methods (Validation or Cross-Validation)](https://aman.ai/primers/ai/k-nearest-neighbors/#data-driven-methods-validation-or-cross-validation)
        - [Comparison of Approaches](https://aman.ai/primers/ai/k-nearest-neighbors/#comparison-of-approaches)
- [Comparison: K-Nearest Neighbors vs. K-Means Clustering](https://aman.ai/primers/ai/k-nearest-neighbors/#comparison-k-nearest-neighbors-vs-k-means-clustering)
    - [Key Takeaways](https://aman.ai/primers/ai/k-nearest-neighbors/#key-takeaways)
- [Citation](https://aman.ai/primers/ai/k-nearest-neighbors/#citation)

## K-Nearest Neighbors (kNN)

### Overview

- **Type:** Supervised learning algorithm.
- **Category:** Non-parametric.
    - kNN is classified as a non-parametric model because it does not make assumptions about the underlying data distribution, making it flexible for a wide range of applications.
- **Primary Use Case:**
    - **Classification:** Assigns discrete class labels to a new data point by evaluating the labels of nearby points.
    - **Regression:** Predicts continuous numerical values by considering the values of neighboring points.
- **Key Idea:**
    - The fundamental principle of kNN revolves around the notion that similar data points are located close to each other in the feature space. Thus, the label or value of a new data point can be inferred based on the labels or values of its closest neighbors.

### Algorithm Description

#### Training Phase

- kNN is a **lazy learning algorithm**, meaning it does not construct a specific model during training. Instead:
    - The algorithm simply memorizes the dataset, denoted as: ={(xi,yi)}ni=1D={(xi,yi)}i=1n Here:
        - xixi is the feature vector representing the ithith data point.
        - yiyi is the corresponding label or output (a class label for classification or a numerical value for regression).
    - As there is no explicit model construction, all computation is deferred to the prediction phase.

#### Prediction Phase

- For a given test point xtxt:
    1. **Compute Distances:**
        - Measure the distance between the test point xtxt and each training data point xixi in the dataset using a selected distance metric d(xt,xi)d(xt,xi).
        - Common distance metrics include **Euclidean distance**, **Manhattan distance**, and others (discussed later).
    2. **Sort Neighbors:**
        - Rank the training points based on their computed distances to xtxt in ascending order.
    3. **Select Nearest Neighbors:**
        - Identify the kk-nearest neighbors (i.e., the kk training points with the smallest distances).
    4. **Make Predictions:**
        - For **classification tasks**:
            - Assign the class label that appears most frequently among the kk neighbors. This is expressed mathematically as: ŷ t=mode{yi∣xi∈neighbors of xt}y^t=mode{yi∣xi∈neighbors of xt}
        - For **regression tasks**:
            - Compute the prediction as the mean or weighted mean of the output values of the kk neighbors: ŷ t=1k∑xi∈neighbors of xtyiy^t=1k∑xi∈neighbors of xtyi

### Distance Metrics

- The choice of distance metric has a profound impact on the accuracy and efficiency of kNN. Popular metrics include:
    
    1. **Euclidean Distance:**
        - The most commonly used metric for continuous numerical data. It calculates the straight-line distance between two points in nn-dimensional space: d(x,y)=∑ni=1(xi−yi)2‾‾‾‾‾‾‾‾‾‾‾‾‾√d(x,y)=∑i=1n(xi−yi)2
        - Suitable when all features are on similar scales.
    2. **Manhattan Distance (L1 Norm):**
        - Measures the distance by summing the absolute differences along each dimension: d(x,y)=∑ni=1|xi−yi|d(x,y)=∑i=1n|xi−yi|
        - Often used when capturing the magnitude of differences is more meaningful than the straight-line distance.
    3. **Minkowski Distance:**
        - A generalization of Euclidean and Manhattan distances, controlled by a parameter pp: d(x,y)=(∑ni=1|xi−yi|p)1pd(x,y)=(∑i=1n|xi−yi|p)1p
            - Special cases:
                - p=2p=2: Euclidean distance.
                - p=1p=1: Manhattan distance.
    4. **Hamming Distance:**
        - Used for categorical features, it counts the number of differing feature values: d(x,y)=∑ni=1𝟙(xi≠yi)d(x,y)=∑i=1n1(xi≠yi) where 𝟙(⋅)1(⋅) is an indicator function that outputs 1 if the condition is true, otherwise 0.

### Parameters

1. **Number of Neighbors (kk):**
    - kk specifies how many neighbors influence the prediction.
    - **Small kk:**
        - Sensitive to noise, as predictions depend on fewer data points.
        - Risks overfitting the data.
    - **Large kk:**
        - Incorporates more neighbors, leading to smoother predictions.
        - Risks underfitting, as it may dilute the influence of closer neighbors.
    - **Optimal kk:**
        - Determined using techniques such as cross-validation, which evaluates performance on a validation set for various kk values.
2. **Distance Metric:**
    - The metric must suit the data type. For instance:
        - Use **Euclidean distance** for numerical data.
        - Use **Hamming distance** for categorical data.
3. **Weighted kNN:**
    - In some cases, assigning higher importance to closer neighbors improves predictions. A common weighting scheme is: wi=1d(xt,xi)2wi=1d(xt,xi)2

### Stopping Criterion

- kNN does not involve iterative updates. Predictions conclude once:
    - Distances are calculated.
    - kk neighbors are identified.
    - The output (classification or regression) is determined.

### Advantages

- **Intuitive and Easy to Implement:** kNN is straightforward and does not require extensive mathematical formulations or parameter tuning.
- **Adaptable to Various Data Types:** Works with both numerical and categorical data.
- **No Training Phase:** Reduces computational costs during training since no explicit model is built.

### Disadvantages

- **Computationally Expensive During Prediction:** Requires distance computation for every training data point at prediction time, which can be slow for large datasets.
- **Sensitive to Feature Scaling:** Features on different scales can dominate the distance metric unless standardized.
- **Performance Relies on Parameter Choices:** Poor choices for kk or distance metrics can significantly degrade model performance.
- **Sensitive to Irrelevant Features:** Irrelevant or noisy features can skew the distance calculations, reducing accuracy.

### Example (Classification)

#### Dataset

|**Point**|**Feature 1 (x₁)**|**Feature 2 (x₂)**|**Label**|
|---|---|---|---|
|A|2|4|Red|
|B|4|4|Blue|
|C|4|6|Blue|
|D|6|4|Red|

#### Test Point: (5,5)(5,5)

1. Compute distances from (5,5)(5,5) to all training points.
2. Sort distances and pick the kk-nearest points.
3. Assign the majority class as the label for (5,5)(5,5).

### FAQ: What are Some Methods That Do Not Require a Predefined kk Value for Nearest Neighbor Selection?

- There are several approaches to adapt the kk-Nearest Neighbors (kNN) algorithm to avoid requiring a predefined kk value. These methods focus on dynamically determining the value of kk or using alternative mechanisms to achieve robust and adaptive nearest neighbor selection. Below are some common techniques explained in greater detail:

#### Distance-Weighted KNN

- **Concept**: Instead of relying on a fixed kk, the algorithm assigns weights to each neighbor based on its distance to the query point. This approach emphasizes the contribution of closer neighbors while reducing the influence of more distant ones.
- **Implementation**:
- Use weighting functions such as:
    - **Inverse Distance Weighting**: wi=1diwi=1di, where didi is the distance to the ii-th neighbor.
    - **Gaussian Function**: wi=e−d2iσ2wi=e−di2σ2, where σσ controls the sensitivity to distance.
- **Advantages**:
- Effectively adapts to local density variations without requiring a fixed neighbor count.
- Mitigates issues arising from abrupt cutoff at a predefined kk.
- **Applications**: Commonly used in regression tasks and probabilistic classification.

#### Radius-Based Neighbors

- **Concept**: Select neighbors within a specified radius rr around the query point instead of a fixed kk number of neighbors.
- **Implementation**:
- Define a radius rr and include all data points within this radius as neighbors.
- Optionally use an **adaptive radius** that depends on the local data density or a scaling factor.
- **Advantages**:
- Eliminates the need for a fixed kk, adapting to local variations in data distribution.
- Suitable for highly imbalanced datasets or datasets with varying density clusters.
- **Challenges**:
- Choosing an appropriate radius rr can be complex, especially for high-dimensional data.
- **Extensions**:
- Dynamically adjust rr based on density estimation techniques or cross-validation.

#### Adaptive kk-Nearest Neighbors

- **Concept**: Dynamically adjust kk based on local data characteristics, such as density or classification stability.
- **Implementation**:
- Increase kk incrementally until:
    - Classification or regression results stabilize.
    - A confidence threshold (e.g., prediction probability) is met.
- Use cross-validation to determine kk for each query point or region of the dataset.
- **Advantages**:
- Allows for tailored neighborhood selection, improving accuracy in heterogeneous datasets.
- Works well in scenarios with overlapping class boundaries or noisy data.
- **Applications**:
- Often used in domains with non-uniform data distributions or high variability.

#### Local Outlier Factor (LOF) or Similar Techniques

- **Concept**: Techniques like LOF generalize neighborhood selection by computing local densities rather than relying on a strict count of neighbors.
- **Implementation**:
- Compare the density of the query point’s neighborhood to the densities of neighboring points.
- Use relative density to determine “neighborhood influence” dynamically.
- **Advantages**:
- Focuses on meaningful neighbors by accounting for density-based variation.
- Robust against outliers and sparse regions in the dataset.
- **Applications**:
- Anomaly detection, density-based clustering, and non-parametric density estimation.

#### Ensemble of KNN Models

- **Concept**: Combine predictions from multiple kNN models built with varying kk values to avoid reliance on a specific kk.
- **Implementation**:
- Train several kNN models with different kk values.
- Aggregate predictions using:
    - **Majority Voting** for classification tasks.
    - **Weighted Averaging** for regression tasks.
- **Advantages**:
- Reduces sensitivity to the choice of kk.
- Improves robustness and generalization performance.
- **Challenges**:
- Increased computational complexity due to multiple models.
- **Applications**:
- Used in ensemble learning frameworks to boost predictive accuracy.

#### Dynamic KNN

- **Concept**: Adjust kk dynamically for each query point based on specific criteria or thresholds.
- **Implementation**:
- Stop increasing kk when:
    - The cumulative distance of the kk-nearest neighbors exceeds a predefined threshold.
    - The classification result achieves a minimum level of confidence.
- Use statistical measures (e.g., confidence intervals or error bounds) to guide kk selection.
- **Advantages**:
- Provides flexibility and adapts to query-specific requirements.
- Ensures that neighborhood selection aligns with the task’s performance criteria.
- **Applications**:
- Real-time decision-making systems, active learning, and streaming data analysis.

#### Data-Driven Methods (Validation or Cross-Validation)

- **Concept**: Automatically select kk through data-driven optimization techniques, eliminating the need for user input.
- **Implementation**:
- Use **cross-validation** to evaluate different kk values on a validation set and select the optimal one.
- Apply advanced optimization techniques, such as:
    - **Grid Search**: Explore a predefined range of kk values systematically.
    - **Bayesian Optimization**: Probabilistically search for the best kk based on performance metrics.
- **Advantages**:
- Provides a principled approach to kk selection.
- Optimizes the balance between bias and variance for the specific dataset.
- **Challenges**:
- Computational overhead due to repeated evaluations.
- **Applications**:
- Suitable for model tuning in supervised learning tasks.

#### Comparison of Approaches

|**Method**|**Concept**|**Implementation**|**Advantages**|**Challenges**|**Applications**|
|---|---|---|---|---|---|
|Distance-Weighted kNN|Weights neighbors based on distance to query point instead of fixed kk.|- Use weighting functions like inverse distance weighting or Gaussian functions.  <br>- Weight reduces influence of distant points.|- Adapts to local density variations.  <br>- Avoids abrupt cutoff issues.  <br>- Improves robustness in regression and probabilistic tasks.|- Requires choice of weighting function and parameters (e.g., σσ).  <br>- Computationally expensive for large datasets.|- Regression tasks.  <br>- Probabilistic classification.|
|Radius-Based Neighbors|Selects neighbors within a specified radius rr instead of a fixed count.|- Define a radius rr around query point.  <br>- Optionally, use adaptive radius based on data density.|- Adapts to varying data densities.  <br>- Suitable for imbalanced datasets.|- Choosing rr is complex, especially in high-dimensional spaces.  <br>- Performance depends on radius selection mechanism.|- Highly imbalanced datasets.  <br>- Datasets with varying density clusters.|
|Adaptive<br><br>kk<br><br>-NN|Dynamically adjusts kk based on local data characteristics.|- Incrementally increase kk until stability or a confidence threshold is achieved.  <br>- Use cross-validation to tailor kk per query or region.|- Improves accuracy in heterogeneous datasets.  <br>- Handles overlapping class boundaries.  <br>- Robust to noisy data.|- Computationally intensive.  <br>- May require domain-specific thresholds for stability or confidence.|- Non-uniform data distributions.  <br>- Noisy or complex datasets.|
|Local Outlier Factor|Selects neighbors dynamically based on relative density rather than a strict count.|- Compare local density of query point with densities of neighbors.  <br>- Use relative density to determine influence dynamically.|- Adapts to density variations.  <br>- Robust to outliers and sparse regions.|- Parameter sensitivity (e.g., scale of density).  <br>- High-dimensional data can complicate density estimation.|- Anomaly detection.  <br>- Density-based clustering.  <br>- Non-parametric density estimation.|
|Ensemble of kNN Models|Combines predictions from multiple kNN models with varying kk values.|- Train several kNN models using different kk values.  <br>- Aggregate predictions via majority voting (classification) or weighted averaging (regression).|- Reduces sensitivity to<br><br>kk<br><br>choice.  <br>- Improves robustness and generalization.|- Increased computational complexity.  <br>- Requires careful aggregation strategy.|- Ensemble learning frameworks.  <br>- Tasks requiring robust predictive accuracy.|
|Dynamic kNN|Adjusts kk dynamically for each query based on criteria like confidence or distance thresholds.|- Increase kk until:  <br>  - Cumulative distance of neighbors exceeds a threshold.  <br>  - Classification result achieves minimum confidence.  <br>- Use statistical measures to guide kk selection.|- Query-specific adaptability.  <br>- Ensures alignment with task performance needs.  <br>- Flexible for diverse scenarios.|- Requires well-defined criteria or thresholds for kk selection.  <br>- High computational demand for real-time applications.|- Real-time decision-making.  <br>- Active learning.  <br>- Streaming data analysis.|
|Data-Driven Methods|Selects kk through validation or cross-validation on data, avoiding user input.|- Use cross-validation to evaluate and select kk.  <br>- Employ optimization techniques like grid search or Bayesian optimization to find the best kk.|- Provides principled<br><br>kk<br><br>selection.  <br>- Balances bias and variance for specific datasets.|- High computational overhead due to repeated evaluations.  <br>- Optimization algorithms may require significant tuning.|- Model tuning in supervised learning.  <br>- Optimizing<br><br>kk<br><br>for varied datasets and performance metrics.|

## Comparison: K-Nearest Neighbors vs. K-Means Clustering

- Here is a detailed comparative analysis of k-Nearest Neighbors and k-Means Clustering:

|**Aspect**|**k-Nearest Neighbors (kNN)**|**k-Means Clustering**|
|---|---|---|
|Type of Algorithm|Supervised learning algorithm (classification or regression).|Unsupervised learning algorithm (clustering).|
|Purpose|Predicts the label or value for a given data point based on its neighbors.|Groups data into kk clusters based on similarity.|
|Input Requirements|Labeled data for training (requires both features and target values).|Unlabeled data (only features are required).|
|Working Mechanism|Finds the kk-nearest points in the training dataset to a given query point and uses their labels to make predictions.|Iteratively partitions data into kk clusters by minimizing intra-cluster variance.|
|Distance Metric|Typically uses Euclidean distance, but other metrics like Manhattan or Minkowski can also be used.|Uses Euclidean distance (or other metrics) to compute cluster centroids.|
|Output|Predicted label (classification) or value (regression).|Cluster assignments for each data point.|
|Training Phase|No explicit training; kNN is a lazy learner and computes distances during prediction.|Training involves multiple iterations to adjust centroids and assign clusters.|
|Prediction Phase|Involves computing distances from the query point to all training points.|Assigns new data points to the nearest cluster based on the trained centroids.|
|Scalability|Not scalable; high computational cost for large datasets due to distance calculations during prediction.|Scalable with optimizations; faster for large datasets after training.|
|Parameters|Number of neighbors (kk) and distance metric.|Number of clusters (kk) and initialization of centroids.|
|Sensitivity to Parameters|Sensitive to the choice of kk; inappropriate kk can lead to overfitting or underfitting.|Sensitive to the choice of kk and initialization of centroids; poor initialization can lead to suboptimal clustering.|
|Interpretability|Intuitive; directly uses neighboring points for prediction.|Less intuitive; requires interpreting clusters.|
|Handling of Outliers|Outliers can strongly influence predictions by affecting the nearest neighbors.|Outliers can distort cluster centroids and lead to poor clustering.|
|Applications|Classification (e.g., image recognition, fraud detection) and regression.|Clustering (e.g., customer segmentation, document classification).|
|Strengths|Simple, effective, and easy to implement; no need for explicit training.|Works well for clustering large datasets; discovers inherent structure.|
|Weaknesses|Computationally expensive during prediction; performance decreases with irrelevant features.|Sensitive to the choice of kk; may converge to local minima.|

### Key Takeaways

- **kNN:** A supervised algorithm used for prediction tasks, often with low computational overhead in the training phase but expensive at prediction time.
- **k-Means:** An unsupervised algorithm used to discover groupings in data, efficient for partitioning compact clusters but sensitive to initialization.
- Both algorithms rely heavily on distance metrics, making feature scaling (e.g., normalization or standardization) critical for their performance.

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledKNN,   title   = {k-Nearest Neighbors},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)