[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • MLOps Testing

- [Overview](https://aman.ai/primers/ai/mlops-testing/#overview)
- [Test Workflows](https://aman.ai/primers/ai/mlops-testing/#test-workflows)
- [Test Types](https://aman.ai/primers/ai/mlops-testing/#test-types)
- [References](https://aman.ai/primers/ai/mlops-testing/#references)
- [Citation](https://aman.ai/primers/ai/mlops-testing/#citation)

## Overview

- Test development is key for most software projects. This article details how we approach MLOps testing. While there are four major types of tests: unit tests, smoke tests, integration tests, and utility tests, we will delve into a bunch more.
- In simple words, unit tests make sure that each class or function behaves as it should, smoke tests make sure that the system works, integration tests make sure that the program results are acceptable, and utility tests give an example on how to use a class or function.

## Test Workflows

- All tests in the categories below fall under two test workflows:
    - **PR gate:** Triggered every time there is a pull request.The objective is to validate that the code is not breaking anything before merging it. The PR gate tests should should be quick and ideally shouldn’t not surpass 20-30 minutes.
    - **Nightly builds:** Tests executed asynchronously and can take hours. Some tests take so long that they cannot be executed in a PR gate, therefore they are executed asynchronously in the nightly builds.
- Notice that the errors in the nightly builds are detected after the code has been merged. This is the reason why, with nightly builds, it is interesting to have a two-level branching strategy. In the standard one-level branching strategy, all pull requests go to the main branch. If a nightly build fails, then the main branch has broken code. In the two-level branching strategy, a pre-production or staging branch is where developers send pull requests to. The main branch is only updated from the staging branch after the nightly builds are successful. This way, the main branch always has working code.

## Test Types

- From Microsoft’s Recommenders [Test workflows](https://github.com/microsoft/recommenders/tree/main/tests), the following categories of tests are essential for MLOps testing:
    - **Data validation tests:** In the data validation tests, we ensure that the schema for input and output data for each function in the pipeline matches the desired prespecified schema, that the data is available and has the correct size. They should be fast and can be added to the PR gate.
    - Unit tests: In the unit tests we just make sure the python utilities and notebooks run correctly. Unit tests are fast, ideally less than $5 \mathrm{~min}$ and are run in every pull request. They belong to the PR gate. For this type of tests, synthetic data can be used.
    - **Functional tests:** These tests make sure that the components of the project not just run but their function is correct. For example, we want to test that an ML model evaluation of RMSE gives a positive number. These tests can be run asynchronously in the nightly builds and can take hours. In these tests, we want to use real data.
    - **Integration tests:** We want to make sure that the interaction between different components is correct. For example, the interaction between data ingestion pipelines and the compute where the model is trained, or between the compute and a database. These tests can be of variable length, if they are fast, we could add them to the PR gate, otherwise, we will add them to the nightly builds. For this type of tests, synthetic and real data can be used.
    - **Smoke tests:** The smoke tests are gates to the slow tests in the nightly builds to detect quick errors. If we are running a test with a large dataset that takes 4 hours, we want to create a faster version of the large test (maybe with a small percentage of the dataset or with 1 epoch) to ensure that it runs end-to-end without obvious failures. Smoke tests can run sequentially with functional or integration tests in the nightly builds, and should be fast, ideally less than 20 mins. They use the same type of data as their longer counterparts.
    - **Performance test:** The performance tests are tests that measure the computation time or memory footprint of a piece of code and make sure that this is bounded between some limits. Another kind of performance testing can be a load test to measure an API response time, this can be specially useful when working with large deep learning models. For this type of tests, synthetic data can be used.
    - **Responsible Al tests:** Responsible Al tests are test that enforce fairness, transparency, explainability, human-centeredness, and privacy.
    - Security tests: Security tests are tests that make sure that the code is not vulnerable to attacks. These can detect potential security issues either in python packages or the underlying OS, in addition to scheduled scans in the production pipelines.
    - **Regression tests:** In some situations, we are migrating from a deprecated version to a new version of the code, or maybe we are maintaining two versions of the same library (i.e. Tensorflow v1 and v2). Regression tests make sure that the code works in both versions of the code. These types of tests sometimes are done locally, before upgrading to the new version, or they can be included in the tests pipelines if we want to execute them recurrently.

## References

- [A Beginner’s Guide To Python Testing](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledMLOpsTesting, title   = {MLOps Testing}, author  = {Chadha, Aman}, journal = {Distilled AI}, year    = {2020}, note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)