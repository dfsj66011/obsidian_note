[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • MLOps Tooling

- [Overview](https://aman.ai/primers/ai/mlops-tooling/#overview)
- [Tooling](https://aman.ai/primers/ai/mlops-tooling/#tooling)
    - [CI/CD for Machine Learning](https://aman.ai/primers/ai/mlops-tooling/#cicd-for-machine-learning)
    - [Cron Job Monitoring](https://aman.ai/primers/ai/mlops-tooling/#cron-job-monitoring)
    - [Data Catalog](https://aman.ai/primers/ai/mlops-tooling/#data-catalog)
    - [Data Enrichment](https://aman.ai/primers/ai/mlops-tooling/#data-enrichment)
    - [Exploratory Data Analysis](https://aman.ai/primers/ai/mlops-tooling/#exploratory-data-analysis)
    - [Data Management](https://aman.ai/primers/ai/mlops-tooling/#data-management)
    - [Vector Databases](https://aman.ai/primers/ai/mlops-tooling/#vector-databases)
    - [Data Processing](https://aman.ai/primers/ai/mlops-tooling/#data-processing)
    - [Data Validation](https://aman.ai/primers/ai/mlops-tooling/#data-validation)
    - [Data Visualization](https://aman.ai/primers/ai/mlops-tooling/#data-visualization)
    - [Drift Detection](https://aman.ai/primers/ai/mlops-tooling/#drift-detection)
    - [Feature Engineering](https://aman.ai/primers/ai/mlops-tooling/#feature-engineering)
    - [Feature Store](https://aman.ai/primers/ai/mlops-tooling/#feature-store)
    - [Hyperparameter Tuning](https://aman.ai/primers/ai/mlops-tooling/#hyperparameter-tuning)
    - [Model Lifecycle](https://aman.ai/primers/ai/mlops-tooling/#model-lifecycle)
    - [Model Serving](https://aman.ai/primers/ai/mlops-tooling/#model-serving)
    - [Model Testing & Validation](https://aman.ai/primers/ai/mlops-tooling/#model-testing--validation)
    - [Simplification Tools](https://aman.ai/primers/ai/mlops-tooling/#simplification-tools)
- [Further Reading](https://aman.ai/primers/ai/mlops-tooling/#further-reading)
    - [Awesome MLOps: References and Articles](https://aman.ai/primers/ai/mlops-tooling/#awesome-mlops-references-and-articles)
    - [Awesome MLOps: Tools](https://aman.ai/primers/ai/mlops-tooling/#awesome-mlops-tools)
- [Citation](https://aman.ai/primers/ai/mlops-tooling/#citation)

## Overview

- The figure below summarizes a typical MLOps flow:

![](https://aman.ai/primers/ai/assets/mlops/mlops.jpg)

## Tooling

### CI/CD for Machine Learning

- Tools for performing CI/CD for Machine Learning.
    - [Github Actions](https://docs.github.com/en/actions): Automate, customize, and execute your software development workflows right in your repository with GitHub Actions.
    - [ClearML](https://github.com/allegroai/clearml): Auto-Magical CI/CD to streamline your ML workflow.

### Cron Job Monitoring

- Tools for monitoring cron jobs (recurring jobs).
    - [Cronitor](https://cronitor.io/cron-job-monitoring): Monitor any cron job or scheduled task.

### Data Catalog

- Tools for data cataloging.
    - [Apache Atlas](https://atlas.apache.org/): Provides open metadata management and governance capabilities to build a data catalog.

### Data Enrichment

- Tools and libraries for data enrichment.
    - [Snorkel](https://github.com/snorkel-team/snorkel): A system for quickly generating training data with weak supervision.

### Exploratory Data Analysis

- Tools for performing data exploration.
    - [Google Colab](https://colab.research.google.com/): Hosted Jupyter notebook service that requires no setup to use.
    - [Jupyter Notebook](https://jupyter.org/): Web-based notebook environment for interactive computing.]

### Data Management

- Tools for performing data management.
    - [DVC](https://dvc.org/): Management and versioning of datasets and machine learning models.
    - [Git LFS](https://git-lfs.github.com/): An open source Git extension for versioning large files.

### Vector Databases

- Tools for VectorDB storage.
    - [Milvus](https://github.com/milvus-io/milvus/): An open source embedding vector similarity search engine powered by Faiss, NMSLIB and Annoy.
    - [Pinecone](https://www.pinecone.io/): Managed and distributed vector similarity search used with a lightweight SDK.
    - [Qdrant](https://github.com/qdrant/qdrant): An open source vector similarity search engine with extended filtering support.

### Data Processing

- Tools related to data processing and data pipelines.
    - [Airflow](https://airflow.apache.org/): Platform to programmatically author, schedule, and monitor workflows.
    - [Hadoop](https://hadoop.apache.org/): Framework that allows for the distributed processing of large data sets across clusters.
    - [Spark](https://spark.apache.org/): Unified analytics engine for large-scale data processing.

### Data Validation

- Tools related to data validation.
    - [Cerberus](https://github.com/pyeve/cerberus): Lightweight, extensible data validation library for Python.
    - [Cleanlab](https://github.com/cleanlab/cleanlab): Python library for data-centric AI and machine learning with messy, real-world data and labels.
    - [Great Expectations](https://greatexpectations.io/): A Python data validation framework that allows to test your data against datasets.

### Data Visualization

- Tools for data visualization, reports and dashboards.
    - [Tableau](https://www.tableau.com/): Powerful and fastest growing data visualization tool used in the business intelligence industry.

### Drift Detection

- Tools and libraries related to drift detection.
    - [TorchDrift](https://github.com/torchdrift/torchdrift/): A data and concept drift library for PyTorch.

### Feature Engineering

- Tools and libraries related to feature engineering.
    - [Featuretools](https://github.com/alteryx/featuretools): Python library for automated feature engineering.

### Feature Store

- Feature store tools for data serving.
    - [Feast](https://feast.dev/): End-to-end open source feature store for machine learning.

### Hyperparameter Tuning

- Tools and libraries to perform hyperparameter tuning.
    - [Hyperopt](https://github.com/hyperopt/hyperopt): Distributed Asynchronous Hyperparameter Optimization in Python.
    - [Optuna](https://optuna.org/): Open source hyperparameter optimization framework to automate hyperparameter search.

### Model Lifecycle

- Tools for managing model lifecycle (tracking experiments, parameters and metrics).
    - [Aim](https://github.com/aimhubio/aim): A super-easy way to record, search and compare 1000s of ML training runs.
    - [Mlflow](https://mlflow.org/): Open source platform for the machine learning lifecycle.
    - [Neptune AI](https://neptune.ai/): The most lightweight experiment management tool that fits any workflow.
    - [Weights and Biases](https://github.com/wandb/client): A tool for visualizing and tracking your machine learning experiments.

### Model Serving

- Tools for serving models in production.
    - [BentoML](https://github.com/bentoml/BentoML): Open-source platform for high-performance ML model serving.
    - [Cortex](https://www.cortex.dev/): Machine learning model serving infrastructure.
    - [KFServing](https://github.com/kubeflow/kfserving): Kubernetes custom resource definition for serving ML models on arbitrary frameworks.

### Model Testing & Validation

- Tools for testing and validating models.
    - [Deepchecks](https://github.com/deepchecks/deepchecks): Open-source package for validating ML models & data, with various checks and suites.

### Simplification Tools

- Tools related to machine learning simplification and standardization.
    - [Hydra](https://github.com/facebookresearch/hydra): A framework for elegantly configuring complex applications.
    - [Ludwig](https://github.com/uber/ludwig): Allows users to train and test deep learning models without the need to write code.

## Further Reading

### [Awesome MLOps: References and Articles](https://github.com/visenger/awesome-mlops)

[![](https://aman.ai/images/read/MLOps-references.jpg)](https://github.com/visenger/awesome-mlops)

### [Awesome MLOps: Tools](https://github.com/kelvins/awesome-mlops)

- A list of tools for machine learning operations (MLOps).

[![](https://aman.ai/images/read/MLOps-tools.jpg)](https://github.com/kelvins/awesome-mlops)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledMLOpsTooling, title   = {MLOps Tooling}, author  = {Chadha, Aman}, journal = {Distilled AI}, year    = {2020}, note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)