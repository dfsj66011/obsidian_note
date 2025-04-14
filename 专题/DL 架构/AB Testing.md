[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Models • A/B Testing

- [Overview](https://aman.ai/primers/ai/ab-testing/#overview)
    - [Purpose of A/B Testing](https://aman.ai/primers/ai/ab-testing/#purpose-of-ab-testing)
    - [How A/B Testing Works](https://aman.ai/primers/ai/ab-testing/#how-ab-testing-works)
        - [Steps Involved](https://aman.ai/primers/ai/ab-testing/#steps-involved)
    - [Stable Unit Treatment Value Assumption (SUTVA)](https://aman.ai/primers/ai/ab-testing/#stable-unit-treatment-value-assumption-sutva)
    - [Common Applications of A/B Testing](https://aman.ai/primers/ai/ab-testing/#common-applications-of-ab-testing)
    - [Benefits of A/B Testing](https://aman.ai/primers/ai/ab-testing/#benefits-of-ab-testing)
    - [Challenges and Considerations in A/B Testing](https://aman.ai/primers/ai/ab-testing/#challenges-and-considerations-in-ab-testing)
    - [Advanced Variants of A/B Testing](https://aman.ai/primers/ai/ab-testing/#advanced-variants-of-ab-testing)
    - [Parameters in A/B Testing](https://aman.ai/primers/ai/ab-testing/#parameters-in-ab-testing)
        - [Sample Size (nn)](https://aman.ai/primers/ai/ab-testing/#sample-size-n)
        - [Minimum Detectable Effect (MDE or Δ)](https://aman.ai/primers/ai/ab-testing/#minimum-detectable-effect-mde-or-%CE%B4)
        - [Significance Level (αα)](https://aman.ai/primers/ai/ab-testing/#significance-level-alpha)
        - [Statistical Power (1 - ββ)](https://aman.ai/primers/ai/ab-testing/#statistical-power-1---beta)
        - [Baseline Conversion Rate (μμ)](https://aman.ai/primers/ai/ab-testing/#baseline-conversion-rate-mu)
        - [Variance (σ2σ2)](https://aman.ai/primers/ai/ab-testing/#variance-sigma2)
        - [Test Duration](https://aman.ai/primers/ai/ab-testing/#test-duration)
        - [Confidence Interval (CICI)](https://aman.ai/primers/ai/ab-testing/#confidence-interval-ci)
        - [Effect Size (Observed Δ)](https://aman.ai/primers/ai/ab-testing/#effect-size-observed-%CE%B4)
        - [Traffic Allocation](https://aman.ai/primers/ai/ab-testing/#traffic-allocation)
        - [Parameter Interdependencies](https://aman.ai/primers/ai/ab-testing/#parameter-interdependencies)
        - [Example](https://aman.ai/primers/ai/ab-testing/#example)
    - [Statistical Power Analysis](https://aman.ai/primers/ai/ab-testing/#statistical-power-analysis)
        - [Application in A/B Testing](https://aman.ai/primers/ai/ab-testing/#application-in-ab-testing)
    - [The Interplay Between Significance Level/Probability of a Type I Error (αα) and the Probability of a Type II Error (ββ)](https://aman.ai/primers/ai/ab-testing/#the-interplay-between-significance-levelprobability-of-a-type-i-error-alpha-and-the-probability-of-a-type-ii-error-beta)
        - [Defining the Parameters](https://aman.ai/primers/ai/ab-testing/#defining-the-parameters)
        - [Factors Influencing the Relationship](https://aman.ai/primers/ai/ab-testing/#factors-influencing-the-relationship)
        - [The Perceived Severity of Errors](https://aman.ai/primers/ai/ab-testing/#the-perceived-severity-of-errors)
        - [A Misconception about ββ and αα Ratios](https://aman.ai/primers/ai/ab-testing/#a-misconception-about-beta-and-alpha-ratios)
        - [Practical Implications](https://aman.ai/primers/ai/ab-testing/#practical-implications)
    - [αα Percentile](https://aman.ai/primers/ai/ab-testing/#alpha-percentile)
        - [Definition of αα in A/B Testing](https://aman.ai/primers/ai/ab-testing/#definition-of-alpha-in-ab-testing)
        - [Percentile Representation](https://aman.ai/primers/ai/ab-testing/#percentile-representation)
        - [Context of the Alpha Percentile in A/B Testing](https://aman.ai/primers/ai/ab-testing/#context-of-the-alpha-percentile-in-ab-testing)
            - [Defining the Critical Region](https://aman.ai/primers/ai/ab-testing/#defining-the-critical-region)
            - [Interpreting Results](https://aman.ai/primers/ai/ab-testing/#interpreting-results)
            - [One-Tailed vs. Two-Tailed Tests](https://aman.ai/primers/ai/ab-testing/#one-tailed-vs-two-tailed-tests)
        - [Visualizing the Alpha Percentile](https://aman.ai/primers/ai/ab-testing/#visualizing-the-alpha-percentile)
        - [Practical Importance](https://aman.ai/primers/ai/ab-testing/#practical-importance)
    - [Measuring Long-Term Effects Using A/B Tests](https://aman.ai/primers/ai/ab-testing/#measuring-long-term-effects-using-ab-tests)
        - [Understand the Long-Term Impact](https://aman.ai/primers/ai/ab-testing/#understand-the-long-term-impact)
        - [Designing the A/B Test for Long-Term Measurement](https://aman.ai/primers/ai/ab-testing/#designing-the-ab-test-for-long-term-measurement)
        - [Addressing Potential Challenges](https://aman.ai/primers/ai/ab-testing/#addressing-potential-challenges)
        - [Analyzing Long-Term Effects](https://aman.ai/primers/ai/ab-testing/#analyzing-long-term-effects)
        - [Dealing with Seasonality](https://aman.ai/primers/ai/ab-testing/#dealing-with-seasonality)
        - [Adjusting the Test As Needed](https://aman.ai/primers/ai/ab-testing/#adjusting-the-test-as-needed)
        - [Post-Test Analysis](https://aman.ai/primers/ai/ab-testing/#post-test-analysis)
    - [Related: Dogfooding vs. Teamfooding vs. Fishfooding](https://aman.ai/primers/ai/ab-testing/#related-dogfooding-vs-teamfooding-vs-fishfooding)
        - [Dogfooding](https://aman.ai/primers/ai/ab-testing/#dogfooding)
        - [Fishfooding](https://aman.ai/primers/ai/ab-testing/#fishfooding)
        - [Why Fishfooding?](https://aman.ai/primers/ai/ab-testing/#why-fishfooding)
        - [How to Implement Fishfooding](https://aman.ai/primers/ai/ab-testing/#how-to-implement-fishfooding)
        - [The Evolution of Fishfooding at Google](https://aman.ai/primers/ai/ab-testing/#the-evolution-of-fishfooding-at-google)
    - [Managing Test Conflicts and Overlap](https://aman.ai/primers/ai/ab-testing/#managing-test-conflicts-and-overlap)
    - [Ensuring Balanced Allocation Groups in A/B Testing](https://aman.ai/primers/ai/ab-testing/#ensuring-balanced-allocation-groups-in-ab-testing)
- [FAQs](https://aman.ai/primers/ai/ab-testing/#faqs)
    - [Why is A/B Testing Necessary Despite Offline Evaluation? What Can Lead to Offline–online Misalignment?](https://aman.ai/primers/ai/ab-testing/#why-is-ab-testing-necessary-despite-offline-evaluation-what-can-lead-to-offlineonline-misalignment)
    - [What Statistical Test Would You Use to Check for Statistical Significance in A/B Testing, and What Kind of Data Would It Apply To?](https://aman.ai/primers/ai/ab-testing/#what-statistical-test-would-you-use-to-check-for-statistical-significance-in-ab-testing-and-what-kind-of-data-would-it-apply-to)
- [Further Reading](https://aman.ai/primers/ai/ab-testing/#further-reading)
- [References](https://aman.ai/primers/ai/ab-testing/#references)

## Overview

- A/B testing is a powerful statistical method used in business, marketing, product development, and UX/UI design to compare two or more versions of a variable (A and B) to determine which version performs better in achieving a specific goal. It’s also known as split testing or bucket testing. This technique is widely used in digital industries such as web development, email marketing, and online advertising to make data-driven decisions, optimize performance, and increase conversion rates.

### Purpose of A/B Testing

- A/B testing is a controlled experiment comparing two (or more) variants (e.g., A and B) to determine which performs better in achieving a specific outcome. Put simply, the primary purpose of A/B testing is to compare two versions of a variable (e.g., a web page, a product, or a marketing campaign) to determine which one performs better based on a specific objective. This testing helps businesses make data-driven decisions by identifying which version yields better results in terms of metrics such as conversion rates, user engagement, revenue, or other key performance indicators (KPIs). Instead of relying on intuition, A/B testing allows organizations to experiment and learn from real user behavior, optimizing outcomes based on empirical evidence.
- **Key purposes include:**
    - Optimizing conversion rates on websites or apps.
    - Enhancing user experience (UX) by testing design variations.
    - Maximizing the effectiveness of marketing campaigns.
    - Improving customer retention by fine-tuning messaging or product features.
    - Validating business hypotheses before rolling out large-scale changes.

### How A/B Testing Works

- A/B testing works by dividing a user base or population into two (or more) groups randomly. Each group is shown a different version of the same variable (for instance, two versions of a landing page). One group, known as the **control group**, is shown the existing version (often referred to as **Version A**), while the other group, known as the **treatment group**, is shown the new or experimental version (often referred to as **Version B**). The responses of both groups are then measured and compared to determine which version performs better according to the chosen success metric.

#### Steps Involved

1. **Identify the Objective:** Establish a clear goal, such as increasing click-through rates or conversions.
2. **Choose a Variable to Test:** Select one variable to test (e.g., headline, image, call-to-action).
3. **Create Variations:** Develop two (or more) versions of the variable—Version A (control) and Version B (variation).
4. **Split the Audience:** Randomly assign users into two groups, ensuring they are equally representative.
5. **Run the Test:** Expose each group to their respective version for a period long enough to gather statistically significant data.
6. **Measure Outcomes:** Analyze the data to see which version outperforms the other according to the predetermined success metrics.
7. **Implement Changes:** Once a winning version is identified, implement it across the entire audience.

### Stable Unit Treatment Value Assumption (SUTVA)

- **Definition:** The Stable Unit Treatment Value Assumption (SUTVA) is a foundational principle in experimental design and A/B testing. It posits that the treatment assignment of one individual does not influence the outcomes of other individuals and that each individual’s experience is solely determined by their own treatment assignment.
    
- **Key Components of SUTVA:**
    1. **No Interference Between Units:** The treatment or condition applied to one unit (e.g., a user or customer) must not affect the outcome or behavior of another unit. For example, if one user is shown Version A of a webpage, their experience should not influence another user who is shown Version B.
    2. **Single Treatment Per Unit:** Each unit must only be exposed to one version of the treatment or condition. This means a user should see only one version (either A or B) and not interact with multiple versions during the experiment.
- **Why SUTVA Matters:**
    - Violations of SUTVA can distort the results of A/B testing, making it difficult to attribute differences in outcomes to the treatment itself. This can lead to biased conclusions and ineffective decision-making.
    - Ensuring that SUTVA holds true is essential for maintaining the integrity and reliability of experimental findings.
- **Examples of SUTVA Violations in A/B Testing:**
    1. **Social Interactions:** Users discussing different versions of a product or campaign with others can introduce interference. For example, if users of Version A share their experience on social media, it might influence users exposed to Version B.
    2. **Multi-Device Usage:** If a single user accesses the experiment on multiple devices (e.g., phone, tablet, and computer), they might inadvertently be exposed to both versions, violating the single-treatment condition.
    3. **Spillover Effects:** Features or information from one version of the experiment (e.g., a promotional offer in Version A) might become accessible to users in the other group (Version B), influencing their behavior.
- **Strategies to Ensure SUTVA Compliance:**
    1. **Randomization:** Properly randomizing the assignment of users to treatment groups reduces the likelihood of systematic interference or bias.
    2. **Isolated Testing Environments:** Create controlled environments where users are exposed to only one treatment and cannot interact with users in other treatment groups.
    3. **Device and Session Tracking:** Use mechanisms to ensure that each user is consistently exposed to the same treatment, regardless of device or session.
    4. **Monitoring for Interference:** Regularly monitor and analyze potential sources of interference, such as shared user accounts or public dissemination of experiment details.
- **Handling SUTVA Violations:**
    - When SUTVA violations are unavoidable, researchers can use statistical methods to account for potential interference. For example:
        - **Network Models:** Incorporate social network data to estimate and control for the influence of user interactions.
        - **Sensitivity Analyses:** Test how robust the results are to potential violations by modeling different levels of interference.
        - **Segmented Analysis:** Analyze subsets of the data where SUTVA violations are less likely, such as geographically or demographically isolated groups.
- **Benefits of Adhering to SUTVA:**
    - **Improved Accuracy:** Ensures that the observed effects can be confidently attributed to the treatment.
    - **Greater Reproducibility:** Facilitates replication of the experiment in similar contexts, as the outcomes are not confounded by uncontrolled interference.
    - **Enhanced Decision-Making:** Provides a more reliable basis for implementing changes based on the experimental findings.
- In summary, SUTVA is a cornerstone of experimental validity in A/B testing. By understanding its principles and addressing potential violations proactively, researchers can ensure that their tests yield robust, actionable insights.

### Common Applications of A/B Testing

- A/B testing is commonly used in digital environments, particularly for improving customer experiences and marketing effectiveness. Here are some key applications:
    
    1. **Website Optimization:**
        - Testing different versions of web pages (e.g., landing pages, product pages, checkout processes).
        - Changing design elements like buttons, layout, images, or text to improve user engagement or conversions.
    2. **Email Marketing:**
        - Comparing email subject lines, content, design, or calls to action to determine which emails drive higher open or click rates.
    3. **Mobile App Optimization:**
        - Testing in-app features, user flows, or notifications to increase retention and user engagement.
    4. **Digital Advertising:**
        - Testing ad creatives, headlines, and targeting options to maximize click-through rates and conversions.
    5. **Pricing Strategies:**
        - Experimenting with different pricing models or promotions to see which drives higher revenue or customer retention.
    6. **User Experience (UX) Improvements:**
        - Testing UI/UX elements like navigation, colors, or onboarding flows to enhance overall user satisfaction.

### Benefits of A/B Testing

1. **Data-Driven Decisions:**
    - A/B testing provides empirical evidence, allowing organizations to base decisions on hard data rather than assumptions or opinions.
2. **Improved User Engagement:**
    - By experimenting with different variations, businesses can refine content, design, and interactions to better resonate with users, leading to improved engagement.
3. **Higher Conversion Rates:**
    - A/B testing can help identify the most effective elements that persuade users to take desired actions, thereby increasing conversions.
4. **Reduced Risk:**
    - Instead of overhauling a product or webpage entirely, businesses can test incremental changes in a controlled manner, reducing the risk of negatively impacting performance.
5. **Optimization Over Time:**
    - Continuous A/B testing allows for iterative improvements, enabling ongoing optimization of user experiences and marketing strategies.
6. **Cost-Effectiveness:**
    - A/B testing allows companies to improve performance without necessarily increasing marketing or product development budgets.

### Challenges and Considerations in A/B Testing

1. **Sample Size Requirements:**
    - A/B testing requires a sufficiently large sample size to achieve statistically significant results. Small sample sizes can lead to unreliable outcomes.
2. **Time Constraints:**
    - A/B tests need to run long enough to gather meaningful data. Short test durations may not account for variations in user behavior over time (e.g., seasonal effects).
3. **False Positives/Negatives:**
    - Misinterpretation of data can lead to incorrect conclusions. Without proper statistical rigor, businesses may implement changes based on results that occurred by chance.
4. **Confounding Variables:**
    - External factors (e.g., marketing campaigns, economic shifts) can influence test results, making it difficult to isolate the effect of the tested variable.
5. **Cost of Experimentation:**
    - While A/B testing can be cost-effective in the long run, setting up experiments, especially with complex platforms or technologies, can be resource-intensive.
6. **Ethical Considerations:**
    - In some cases, testing certain variations may lead to negative user experiences, which could harm the brand or customer relationships if poorly managed.
7. **Test Interference (Cross Contamination):**
    - In situations where users encounter both versions (e.g., in marketing emails or across multi-device platforms), the test results can be skewed, affecting the validity of the test.

### Advanced Variants of A/B Testing

1. **Multivariate Testing:**
    - Unlike A/B testing, which compares two versions of a single variable, **multivariate testing** allows multiple elements to be tested simultaneously (e.g., different headlines, images, and call-to-action buttons). The goal is to understand how combinations of different variables impact user behavior. Multivariate testing is more complex and requires larger sample sizes but can provide deeper insights into how various elements interact.
2. **Split URL Testing:**
    - This involves testing entirely different URLs (e.g., a different version of a website or landing page hosted on separate domains or subdomains). Split URL testing is useful for testing broader design or structural changes.
3. **Bandit Testing:**
    - **Multi-armed bandit algorithms** optimize the A/B testing process by dynamically adjusting traffic allocation to different variations in real-time. This reduces the time it takes to identify the best-performing version and minimizes the risk of losing potential conversions during the testing phase.
4. **Personalization and Segmentation:**
    - Advanced A/B tests might involve segmenting users into different groups based on behavior, demographics, or preferences. This allows for testing more personalized experiences, which can lead to better results as compared to one-size-fits-all solutions.
5. **Sequential Testing:**
    - This approach focuses on monitoring test results as they unfold, allowing for early stopping if one variation is clearly outperforming the other. Sequential testing aims to make the testing process more efficient without sacrificing statistical rigor.
6. **Adaptive Testing:**
    - In **adaptive testing**, the test dynamically adjusts as data is collected, altering the allocation of traffic to more promising variations in real-time. This approach aims to balance exploration and exploitation, potentially reaching optimal outcomes more quickly than traditional A/B tests.

### Parameters in A/B Testing

- Below, we break down the critical parameters in A/B testing, their roles, typical values, and how they interact.

#### Sample Size (nn)

- **Role**: The number of participants (users) needed in each group (A or B) to detect a meaningful difference between variants.
- **Influence**: Directly impacts the statistical power of the test. A larger sample size reduces variability and increases the likelihood of detecting a true difference.
- **Typical Values**: Determined by a power analysis based on desired significance level (α), power (1-β), and expected effect size.
- **Equation**: n=(Zα/2+Zβ)2⋅2⋅σ2Δ2n=(Zα/2+Zβ)2⋅2⋅σ2Δ2
    - where:
        - Zα/2Zα/2: Z-score for the significance level.
        - ZβZβ: Z-score for the desired power.
        - σσ: Standard deviation.
        - ΔΔ: Minimum detectable effect (MDE).

#### Minimum Detectable Effect (MDE or Δ)

- **Role**: The smallest difference between the two groups that the test is designed to detect.
- **Influence**: Drives sample size requirements. Smaller MDE requires larger samples.
- **Typical Values**: Depends on the context, e.g., a 1-5% lift in conversion rates for websites or a larger effect for significant design changes.
- **Equation**: MDE=Δμ×100%MDE=Δμ×100%
    - where μμ is the baseline metric (e.g., conversion rate).

#### Significance Level (αα)

**Role**: The probability of rejecting the null hypothesis when it is true (Type I error rate).

- **Influence**: Sets the threshold for statistical confidence. A lower α reduces the likelihood of false positives but may require a larger sample size.
- **Typical Values**: 0.05 (95% confidence) or stricter thresholds like 0.01 for high-stakes decisions.

#### Statistical Power (1 - ββ)

- **Role**: The probability of correctly rejecting the null hypothesis when the alternative hypothesis is true.
- **Influence**: Higher power requires a larger sample size but reduces the likelihood of Type II errors (false negatives).
- **Typical Values**: 0.8 (80%) or higher, with 0.9 used for highly sensitive experiments.

#### Baseline Conversion Rate (μμ)

- **Role**: The initial success rate (e.g., current conversion rate) against which changes are measured.
- **Influence**: Affects sample size. Lower baseline rates typically require larger sample sizes to detect a meaningful change.
- **Typical Values**: Depends on the domain (e.g., 2-5% for e-commerce conversion rates, 20-30% for email open rates).

#### Variance (σ2σ2)

- **Role**: The measure of data spread or variability in the metric being tested.
- **Influence**: High variance increases the noise in results, requiring a larger sample size to detect a given effect.
- **Equation**: σ2=p(1−p)σ2=p(1−p)
    - where pp is the probability of success (e.g., baseline conversion rate).

#### Test Duration

- **Role**: The length of time the test is run.
- **Influence**: Ensures sufficient data collection but should be balanced against time constraints and external factors (e.g., seasonality).
- **Typical Values**: Determined by traffic volume and sample size requirements. A/B tests typically last 1-4 weeks.

#### Confidence Interval (CICI)

- **Role**: The range within which the true effect size is likely to lie, given a specific confidence level.
- **Influence**: Wider intervals indicate more uncertainty. CI width is influenced by sample size and variance.
- **Typical Values**: 95% CI is standard, meaning the range is expected to contain the true value 95% of the time.

#### Effect Size (Observed Δ)

**Role**: The actual difference in performance observed between variants during the test.

- **Influence**: Used to determine whether the observed effect is statistically significant and practically meaningful.
- **Equation**: Effect Size=MeanB−MeanAσEffect Size=MeanB−MeanAσ

#### Traffic Allocation

- **Role**: The proportion of total traffic directed to each variant (e.g., 50%-50% split).
- **Influence**: Equal splits maximize statistical power, but unbalanced splits can be used for practical reasons, like rolling out new features.
- **Typical Values**: 50%-50% or 90%-10% for phased rollouts.

#### Parameter Interdependencies

- **Sample Size and MDE**: Smaller MDE increases required sample size.
- **α, β, and Power**: Lower α (stricter significance) and higher power increase sample size.
- **Baseline Rate and Variance**: Lower baseline rates often result in higher variance, requiring larger samples.

#### Example

- Assume you’re testing a website with a baseline conversion rate (μμ) of 3%, aiming to detect a 1% absolute improvement (MDE). At α=0.05α=0.05, power of 80%, and assuming variance from conversion rate:
- Calculate sample size: n=(1.96+0.84)2⋅2⋅0.03(1−0.03)0.012n=(1.96+0.84)2⋅2⋅0.03(1−0.03)0.012
    - This yields n≈4,312n≈4,312 per group.
- This calculation guides your traffic needs, test duration, and feasibility.
- By systematically considering each parameter, A/B testing ensures robust, reliable conclusions about the impact of changes.

### Statistical Power Analysis

- Statistical power analysis involves evaluating the likelihood that a test will correctly reject the null hypothesis (H0H0) when it is indeed false. The power calculator enables this assessment by computing the statistical power based on the sample size and generating a precise power analysis chart.

> **An increase in sample size enhances/increases statistical power.**

- Key concepts include:
    
    1. **Definition of Statistical Power:** Statistical power is the probability of rejecting the null hypothesis (H0H0) when it is false. It is mathematically expressed as 1−β1−β, where ββ represents the probability of a Type II error (failing to reject a false H0H0).
        
    2. **Standard Power Levels:** Researchers commonly aim for a statistical power of 0.8, meaning the maximum acceptable ββ level (Type II error rate) is 0.2. This ensures a reasonable balance between Type I and Type II errors.
        
    3. **Significance Level (αα):** The standard significance level is typically set at 0.05, reflecting the maximum probability of a Type I error (incorrectly rejecting a true H0H0).
        
    4. **Relationship Between ββ and αα:** The ββ level is generally four times greater than the significance level (αα), reflecting the consensus that rejecting a true null hypothesis (false negative) is considered more severe than failing to reject a false one (false positive).
        

#### Application in A/B Testing

- In A/B testing, statistical significance is often assessed using the t-test (specifically the two-sample t-test) when the data meets normality assumptions.
- If the data does not conform to normality, the Mann-Whitney U test (also known as the Wilcoxon rank-sum test) serves as a non-parametric alternative. These tests are used to compare the means or distributions of two groups to determine if there is a statistically significant difference between them.

### The Interplay Between Significance Level/Probability of a Type I Error (αα) and the Probability of a Type II Error (ββ)

- In hypothesis testing, two key probabilities are fundamental to evaluating the outcomes of statistical decisions: the significance level (αα) and the probability of a Type II error (ββ). These parameters represent distinct types of errors in statistical inference and are intrinsically interconnected. Understanding their interplay is crucial for designing robust experiments and interpreting results effectively.
- The significance level (αα) reflects the risk of incorrectly rejecting a true null hypothesis (Type I error), while ββ represents the risk of failing to reject a false null hypothesis (Type II error). The relationship between these probabilities is influenced by factors such as effect size, sample size, and study design. Striking an appropriate balance between αα and ββ requires careful consideration of the specific research context and the potential consequences of each type of error.
- This nuanced interplay highlights the trade-offs inherent in hypothesis testing. For instance, reducing αα to minimize the risk of a Type I error can increase ββ, heightening the likelihood of a Type II error, and vice versa. Understanding these dynamics is critical to design studies with optimal power and ensure their findings are both reliable and meaningful.

#### Defining the Parameters

1. **Significance Level/Type I Error Probability (αα)**: The significance level represents the probability of committing a Type I error, which occurs when a true null hypothesis is incorrectly rejected. It is often set to small values (e.g., 0.05 or 0.01) to minimize the likelihood of false positives, particularly in fields where unwarranted conclusions could have serious implications.
    
2. **Type II Error Probability (ββ)**: This parameter denotes the probability of failing to reject a false null hypothesis. In essence, a Type II error occurs when the test lacks sufficient sensitivity to detect a true effect or difference. The complement of ββ is statistical power, which quantifies the test’s ability to correctly reject a false null hypothesis.
    

#### Factors Influencing the Relationship

- The relationship between αα and ββ is not fixed but depends on several factors:
    
    1. **Effect Size**: Larger effect sizes are easier to detect, reducing ββ for a given αα. Conversely, smaller effect sizes increase ββ, as the test struggles to discern true effects from random noise.
        
    2. **Sample Size**: Increasing the sample size reduces ββ by enhancing the test’s sensitivity, thereby improving statistical power. However, larger sample sizes may also reduce variability, potentially enabling lower αα thresholds without compromising power.
        
    3. **Significance Level (αα)**: Lowering αα (e.g., from 0.05 to 0.01) decreases the likelihood of Type I errors but generally increases ββ. This inverse relationship arises because stricter thresholds for rejecting the null hypothesis require stronger evidence, making it harder to detect true effects.
        
    4. **Study Design and Variance**: Factors such as experimental design, data variability, and measurement precision also influence ββ. Well-designed studies with controlled variance reduce ββ for any chosen αα.
        

#### The Perceived Severity of Errors

- A critical aspect of determining αα and ββ is weighing the relative consequences of Type I and Type II errors. In many fields, Type I errors (false positives) are perceived as more severe, leading to a prioritization of minimizing αα. For example, falsely claiming the efficacy of a drug could have significant public health implications. However, in other scenarios—such as safety-critical engineering—failing to detect a true risk (Type II error) may be deemed more consequential, necessitating a different balance.

#### A Misconception about ββ and αα Ratios

- It is sometimes claimed that ββ is “generally” set to be four times αα, reflecting a greater tolerance for Type II errors compared to Type I errors. While this might align with some heuristic guidelines in specific contexts, it is not a universal principle. The ratio of ββ to αα should be determined based on the study’s goals, the stakes of different error types, and practical considerations like resource constraints and achievable sample sizes.

#### Practical Implications

- Balancing αα and ββ requires thoughtful consideration:
    - **Context Matters**: The severity of each error type depends on the field of application. In medical research, minimizing false positives may be paramount, while in exploratory research, greater emphasis might be placed on avoiding false negatives.
    - **Statistical Power**: Researchers often aim for a power of 0.8 or higher (i.e., β=0.2β=0.2) while maintaining a conventional αα (e.g., 0.05). This balance ensures a reasonable trade-off between detecting true effects and avoiding false positives.
    - **Iterative Design**: Simulation studies and power analysis can help optimize αα and ββ given specific study parameters, reducing the risk of overly simplistic assumptions about their relationship.

### αα Percentile

- The alpha percentile is a critical statistical threshold in A/B testing, used to determine whether the observed results are significant enough to reject the null hypothesis. It represents the point in a probability distribution where outcomes are considered sufficiently extreme to suggest that observed differences are not due to random chance.
- This concept is closely tied to statistical significance and confidence levels. Statistical significance indicates the likelihood that the observed effect in an A/B test is real rather than a product of random variation. The alpha percentile directly sets this significance level, often expressed as a value like 0.05 (or 5%). This means there is a 5% risk of incorrectly rejecting the null hypothesis, also known as a Type I error.
- Setting and interpreting the alpha percentile correctly is crucial for ensuring robust and reliable conclusions from your A/B tests. It acts as the benchmark for deciding whether the test results justify a shift in business strategy or validate a hypothesis. By understanding the alpha percentile and its relationship to statistical significance, you can make data-driven decisions with greater confidence.
- To understand it in detail, let’s break it down:

#### Definition of αα in A/B Testing

- αα is the significance level used in hypothesis testing. It represents the probability of making a Type I error, which occurs when you wrongly reject the null hypothesis (i.e., you think there is a difference between the A and B groups when there isn’t one).
- Typically, alpha is set to 0.05, meaning there is a 5% chance of rejecting the null hypothesis when it is true.

#### Percentile Representation

- The alpha percentile corresponds to the portion of the distribution in a statistical test that falls in the rejection region under the null hypothesis.
- For example:
    - If α = 0.05, then the alpha percentile is the 5th percentile in the context of a one-tailed test or split across the two tails (2.5th and 97.5th percentiles) in a two-tailed test.
    - This means that 5% of the null distribution lies in the critical region, and if your test statistic falls within this region, you reject the null hypothesis.

#### Context of the Alpha Percentile in A/B Testing

- In A/B testing, you are typically comparing two groups (control and variant) to determine if a difference in performance metrics (e.g., click-through rates, conversions) is statistically significant. Here’s how the alpha percentile comes into play:

##### Defining the Critical Region

- The alpha percentile helps define the cutoff point(s) in the test’s distribution.
- For a two-tailed test (commonly used in A/B testing), the alpha is split equally into two tails (e.g., 0.025 in each tail if α = 0.05).
- If your observed test statistic or p-value falls into one of these extreme regions (the alpha percentiles), it suggests that the result is unlikely to have occurred under the null hypothesis.

##### Interpreting Results

- Suppose you’re conducting an A/B test with a 95% confidence level (α = 0.05):
    - The alpha percentiles would be at the 2.5th percentile and 97.5th percentile for a two-tailed test.
    - If your test statistic exceeds these bounds, you reject the null hypothesis and conclude there is a statistically significant difference between groups A and B.

##### One-Tailed vs. Two-Tailed Tests

- **One-tailed test:** The entire alpha (e.g., 0.05) is in one tail, so the critical alpha percentile is the 5th percentile in the context of a left-tailed test or the 95th percentile in a right-tailed test.
- **Two-tailed test:** The alpha is split across both tails, so the critical alpha percentiles are at the 2.5th percentile (lower tail) and the 97.5th percentile (upper tail).

#### Visualizing the Alpha Percentile

- Imagine a bell curve (normal distribution) representing the null hypothesis:
- The area under the curve totals 100%.
- The alpha percentiles define the edges of the rejection regions.
- For α = 0.05 in a two-tailed test, these rejection regions cover the outermost 5% (2.5% on each side).

#### Practical Importance

- In A/B testing, understanding the alpha percentile ensures:
    - You can determine whether observed differences are due to chance or represent a real effect.
    - You control the likelihood of false positives (Type I errors).
    - Decisions are made with a clear understanding of the risk involved in rejecting the null hypothesis.

### Measuring Long-Term Effects Using A/B Tests

- A/B testing (also known as split testing) is a common method used to compare two or more variations of a variable—such as a webpage, product feature, or marketing strategy—to determine which performs better. While A/B testing is often associated with short-term evaluations, measuring long-term effects is equally critical, especially when changes are expected to have lasting impacts over time.
- Measuring long-term effects with A/B testing requires careful planning, extended test durations, appropriate metrics, and advanced analysis techniques to capture sustained impacts. These tests are more complex and demand more patience than their short-term counterparts. However, they are crucial for understanding the true value of changes and their enduring effects on user behavior. This deeper insight enables more strategic decision-making, ensuring that improvements lead to sustained success rather than temporary gains.
- Here’s a detailed explanation of how to measure long-term effects using A/B tests.

#### Understand the Long-Term Impact

Long-term effects refer to changes that persist beyond the immediate reaction to a new feature or treatment. For example, a product change might increase user engagement temporarily, but the true value lies in whether that increase persists over time. The goal of long-term A/B testing is to capture these sustained effects.

#### Designing the A/B Test for Long-Term Measurement

- **Extended Test Duration:** Long-term effects require a test duration long enough to capture the sustained impact of the change. This is crucial because initial user behavior might differ from behavior over time. A test that only runs for a few days or weeks may only capture novelty effects or short-term reactions.
    
    The duration of the test should be based on the nature of the product or service, as well as the expected time frame over which the long-term effects could manifest. For example:
    
    - **Subscription-based services:** Testing might need to last several months to measure retention rates.
    - **E-commerce changes:** The test might need to cover multiple purchase cycles.
- **Choosing the Right Metrics:** Identify the key performance indicators (KPIs) that best reflect the long-term effects of the treatment. These may differ from short-term KPIs:
    
    - **Short-term KPIs**: Immediate reactions like clicks, sign-ups, or purchases.
    - **Long-term KPIs**: Retention rates, customer lifetime value (CLV), repeat purchases, sustained engagement, etc.
    
    Long-term KPIs are often lagging indicators, meaning it takes time for their effects to be measurable. For example, increasing user engagement with a feature may only show significant impacts on retention after several months.
    
- **Randomization and Consistency:** The same principles of randomization and isolation of variables apply in long-term testing. Ensure that participants are randomly assigned to control and treatment groups at the beginning of the experiment and that they remain in their respective groups for the duration of the test. Consistency is important to ensure that the long-term impact is not confounded by external factors or user crossover.

#### Addressing Potential Challenges

- **Attrition and Sample Size Decay:** Over long periods, user attrition (people leaving the experiment) can pose a challenge. This could happen if users churn from the product altogether or simply become inactive. To maintain statistical power, you need to account for this when designing the experiment and calculating the initial sample size.
    
- **Time-Dependent Confounders:** Long-term tests are more exposed to external factors that can influence outcomes, such as seasonality, competitor actions, or economic changes. It’s important to monitor these potential confounders and adjust the analysis if necessary. Techniques like **time-series analysis** or **cohort analysis** can help isolate the effect of the treatment from other time-based factors.
    
- **Feature Decay and Novelty Effects:** Often, new features show a “novelty effect” where users initially respond positively, but that effect diminishes over time as the novelty wears off. Conversely, some changes may require an “adjustment period” where users initially react negatively but later adapt and show long-term positive effects. Monitoring long-term trends can help distinguish these phenomena.
    

#### Analyzing Long-Term Effects

- **Tracking Over Time:** Regularly monitor and track how the performance of the treatment and control groups evolve over time. You may see different phases in the results:
    - **Initial Phase:** Early responses that may reflect immediate excitement or resistance to the change.
    - **Adaptation Phase:** Users start to adapt to the change, which could either stabilize or cause shifts in behavior.
    - **Sustained Phase:** The period in which behavior stabilizes and gives insight into the long-term impact.
- **Segmenting by Time-Based Cohorts:** A common technique to analyze long-term effects is to segment users into cohorts based on when they were exposed to the treatment. For example, you can look at how user behavior evolves from the first week after exposure to the change, the second week, and so on. This approach helps in understanding how the impact develops over time.
    
- **Using Cumulative Metrics:** Cumulative metrics aggregate data over the duration of the test. For example, instead of measuring retention as a percentage of users who returned after one week, you could measure cumulative retention over months. This approach can smooth out short-term fluctuations and give a clearer picture of the long-term effects.
    
- **Statistical Significance and Confidence Intervals:** Just as with short-term A/B tests, long-term A/B tests require rigorous statistical analysis. Given the extended duration, it may take longer to reach statistical significance. Also, confidence intervals around long-term effects can be wider due to the complexity of variables involved. Using bootstrapping or Bayesian methods can provide more robust interpretations of the results.

#### Dealing with Seasonality

- **Seasonal Effects:** Long-term tests can run through different seasons, holidays, or promotional periods, which might affect user behavior. For example, e-commerce traffic typically spikes around Black Friday, while activity may dip during summer vacations. When running long-term A/B tests, it’s important to account for such seasonal effects in the analysis.
    
    - **Strategies:** You might consider running the test for at least one full cycle of seasonality (e.g., one year) to ensure that such effects are normalized. Alternatively, seasonality can be adjusted for in the analysis through regression models or by comparing relative differences in performance during and outside seasonal events.

#### Adjusting the Test As Needed

Long-term A/B tests can sometimes uncover unexpected results that require adjustments:

- **Stopping Rules:** Predefine stopping rules to determine under what conditions the test can be ended early (e.g., if significant positive or negative results emerge).
- **Interim Analysis:** Conduct periodic analysis to ensure that the test is still providing valuable insights without introducing bias. This needs to be done carefully to avoid peeking bias.

#### Post-Test Analysis

After the long-term A/B test concludes, conduct an in-depth post-test analysis:

- **Longitudinal Data Analysis:** Examine how metrics evolved throughout the test to better understand whether changes were sustained, peaked, or declined over time.
- **Generalizing Results:** Consider how the results can be generalized to future scenarios or different user segments. For example, if a feature improves engagement for certain user cohorts (e.g., new users), it might be worth exploring if the effect diminishes as users become more familiar with the product.

### Related: Dogfooding vs. Teamfooding vs. Fishfooding

- Fishfooding, dogfooding, and teamfooding represent different stages in the internal testing lifecycle of a product, each with unique benefits and purposes. These approaches help ensure product quality, enhance user satisfaction, and foster alignment among team members. By understanding the distinctions and synergies among these practices, organizations can create a robust testing framework that ensures products meet and exceed user expectations.

#### Dogfooding

- Dogfooding refers to the practice of using an app or feature shortly before it is publicly released. The term originates from the expression “eating your own dog food”, emphasizing that a company should be willing to use its own products to ensure their quality before releasing them to customers.

#### Fishfooding

- Fishfooding involves testing a product or feature much earlier in its development cycle, often before it is considered finished or ready for broader testing. The term “fishfood” originated from the Google+ team, where the internal codename for the project was Emerald Sea. During its early stages, the platform wasn’t refined enough for dogfooding, so they dubbed the initial testing phase “fishfood,” in keeping with the aquatic theme.

#### Why Fishfooding?

- Fishfooding offers distinct advantages, including:
    
    1. **Early Identification of Critical Issues**: By using the product early in development, teams can proactively address significant problems before they escalate, saving time and resources.
    2. **Enhanced User Experience**: Team members experience the product firsthand, uncovering usability and functionality issues that might be missed in later testing phases.
    3. **Team Alignment**: Having the entire team use the product fosters shared understanding and alignment around its goals and functionalities.
    4. **Customer Empathy**: Direct use of the product helps the team empathize with end-users, leading to more user-centric decisions.

#### How to Implement Fishfooding

- To implement fishfooding effectively, teams should follow these steps:
    
    1. **Set Clear Goals**: Define objectives for fishfooding, such as identifying bugs, enhancing usability, or understanding the user experience.
    2. **Set Clear Expectations**: Gain team buy-in by clearly outlining the level of effort required. Ensure team members understand their role and the importance of participation.
    3. **Create a Feedback System**: Establish a system for collecting and analyzing feedback, such as using tools like Centercode to centralize, organize, and prioritize inputs.
    4. **Iterate and Improve**: Use the feedback to refine the product, focusing not just on identifying problems but on solving them early.
    5. **Celebrate Wins and Learn from Losses**: After each cycle, acknowledge improvements made and evaluate unresolved issues to foster a culture of continuous learning.

#### The Evolution of Fishfooding at Google

- Google’s approach to fishfooding showcases how the method can evolve within an organization. During the development of Google+, the initial fishfooding phase served as an internal test of the unfinished platform. Later, Google introduced an intermediary phase called “teamfooding,” bridging the gap between fishfooding and the broader company-wide dogfooding stage. This multi-layered testing strategy enabled Google to iteratively refine the product before broader deployment.

### Managing Test Conflicts and Overlap

- An online service’s users are often part of multiple A/B tests simultaneously, as long as these tests do not conflict with one another. A conflict typically arises when two tests aim to modify or measure the same feature or area in differing ways, potentially interfering with the results. To help experiment owners identify and manage such conflicts, specialized tools or dashboards are often provided. These tools allow users to filter and view tests across various dimensions, enabling them to pinpoint overlapping areas of influence and ensure that experiments are conducted independently and effectively.
- For instance, to help test owners track down potentially conflicting tests, Netflix offers them with a test schedule view in [ABlaze](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15), the front end to their experimentation platform. This tool lets them filter tests across different dimensions to find other tests which may impact an area similar to their own.

### Ensuring Balanced Allocation Groups in A/B Testing

- When assigning participants to allocation groups in an A/B test, it is essential to ensure that each group is as homogeneous as possible. This homogeneity ensures that any observed differences between the groups can be attributed to the experimental intervention rather than pre-existing differences in the population. Key dimensions for achieving this balance often include attributes like geographical location and device type (e.g., smart TVs, game consoles, etc.).
- To maintain proportional representation across allocation groups, stratified sampling is used instead of purely random sampling. Purely random sampling can lead to imbalances, such as disproportionately assigning users from a specific country or device type to one group. Stratified sampling addresses this issue by dividing the population into subgroups (or strata) based on the key dimensions, then randomly sampling within each subgroup to preserve proportionality across all allocation groups.
- Although the implementation of stratified sampling can vary and may involve significant complexity, the approach ensures fair and reliable comparisons between allocation groups, allowing for statistically meaningful conclusions.

## FAQs

### Why is A/B Testing Necessary Despite Offline Evaluation? What Can Lead to Offline–online Misalignment?

- A common problem when developing recommender systems is that offline performance (evaluated on held-out historical data) may not reflect online performance (evaluated in an A/B test with live user interactions). For instance, deep learning models can show significant offline gains over other methods but fail to translate those gains online, or even result in worse performance. This occurs because correlational performance metrics (used offline) are not always indicative of causal behavior in a dynamic environment.
- A/B testing remains critical even after achieving improved scores in offline evaluation because offline tests, while valuable, do not fully capture the complexity of real-world environments. Here’s why:
    
    1. **Violation of the IID Assumption**: Offline evaluation typically relies on the assumption of independent and identically distributed (IID) data. However, this assumption may break when the model is deployed. In a real-world environment, the data is influenced by various factors such as user interactions, changing behaviors (due to say, interactive effects of newly introduced features for a product), and external influences that don’t appear in offline test data. For example, a new ranking model might alter user behavior (due to the items it surfaces), meaning the interactions seen post-deployment are no longer distributed in the same way as in training.
        
    2. **Non-Stationarity of Data / Staleness of Offline Evaluation Data / Distribution Mismatch**:
        - In real-world applications, data and conditions often evolve over time—a phenomenon known as non-stationarity. User preferences, trends, and behaviors can shift, leading to the staleness of the data used for offline model evaluation. Consequently, a model that performs well in static, offline tests may prove less effective in dynamic, real-world environments.
        - This issue is exacerbated by a distribution mismatch, where the data used for training and evaluation does not accurately reflect the conditions during deployment. One common example is covariate shift, where the distribution of input features changes over time. Such shifts can significantly degrade model performance, especially for deep learning models, which often require stable and representative data due to their complexity and sensitivity to input changes.
        - Addressing these challenges requires strategies such as continuous model retraining, monitoring for data drift, and leveraging robust validation techniques that account for evolving deployment conditions.
    3. **Network Effects / Feedback Loops**: When deploying models in an interconnected system like social media or e-commerce, network effects may arise. For instance, the introduction of a new ranking model may lead to a feedback loop where user behavior affects the content that is surfaced or highlighted, which in turn affects user behavior. This complexity isn’t captured in offline evaluations and requires A/B testing to detect and understand.
        
    4. **Overfitting to Proxy Objective Functions and Metrics**: Actual business goals (e.g., long-term user satisfaction, etc.) are often difficult to measure in an offline setting, so models are trained on proxy metrics like clicks or plays. These proxies may not correlate perfectly with the desired outcome, and powerful deep-learning models might overfit to these short-term metrics, deviating significantly from long-term online objectives.
        
    5. **Data Leakage**: Data leakage can occur in multiple ways, leading to an overestimation of the model’s performance during offline evaluation. Two common scenarios are:
        - **Training Data Present in Test Data**: Data leakage can happen if the training data is inadvertently included in the test set. In this case, the model might be evaluated on data it has already seen during training, artificially boosting its performance metrics. This happens because the model is effectively being tested on known data, rather than unseen data, which inflates its apparent accuracy and generalizability.
        - **Model Trained on Test Data**: Another form of data leakage occurs when test data is mistakenly included in the training set. This allows the model to learn from the test data before it is evaluated, leading to misleadingly high performance during offline evaluation. In deployment, however, the model will fail to generalize properly to new, unseen data, as it has become reliant on patterns from the test data that would not be available in a real-world scenario.
        - While the model may appear to perform well in offline tests due to these forms of leakage, its true performance may be far worse in a live environment. A/B testing helps uncover these issues by providing a realistic measure of performance without relying on flawed offline evaluations.
    6. **Unmodeled Interactions / Interactive Effects**: In an online setting, there could be interactions between different elements, such as ads or products, that were not accounted for in the offline evaluation. A new model might produce unforeseen effects when deployed, leading to interactions that negatively impact user experience or performance, even though offline metrics improved.
        
    7. **Fairness Concerns Post-Deployment**: Fairness and bias are especially critical when models impact real-world entities such as users or products. Deploying machine learning models often reveals hidden issues that were not apparent during training and offline evaluation. Offline evaluations frequently lack the nuanced data required to assess fairness comprehensively, meaning some issues only become evident after deployment. Moreover, while techniques such as LIME, SHAP, and Integrated Gradients can be utilized, the inherent complexity of deep learning models makes them difficult to explain and audit for fairness. These challenges can include biases embedded in the training or evaluation data, which might only surface when the model operates at scale. A/B testing becomes a crucial tool in such scenarios, as it enables a comparison between the model’s real-world performance and expectations derived from pre-deployment evaluations.
- While offline evaluation is useful for initial model validation, A/B testing in a live environment is essential to fully understand how the model performs in practice. It captures complexities like user interactions, feedback loops, dynamic environments, and issues such as data leakage or distribution mismatches that cannot be simulated effectively offline. Iterative refinement of metrics and models, combined with robust online testing, is crucial for ensuring effective and reliable model deployment.

### What Statistical Test Would You Use to Check for Statistical Significance in A/B Testing, and What Kind of Data Would It Apply To?

- In A/B testing, a commonly used statistical test to check for statistical significance is the t-test (specifically the two-sample t-test) if the data is normally distributed. If the data does not meet normality assumptions, the Mann-Whitney U test (also known as the Wilcoxon rank-sum test) can be used as a non-parametric alternative. You would apply these tests to continuous metrics such as the average order value, click-through rate, or conversion rate between the two groups (A and B) to determine if there is a significant difference in their means.

## Further Reading

- The multi-part [Decision Making at Netflix](https://netflixtechblog.com/decision-making-at-netflix-33065fa06481) blog from Netflix offers a great overview of the A/B testing process.

## References

- [Statistical power calculators](https://www.statskingdom.com/statistical-power-calculators.html)
- [How to set alpha when you have underpowered experiments?](https://www.linkedin.com/pulse/how-set-alpha-when-you-have-underpowered-experiments-ron-kohavi-zguqe/?trackingId=qmzxdJKNGPq9wGekpAt%2FNA%3D%3D)
- [Quasi Experimentation at Netflix](https://netflixtechblog.com/quasi-experimentation-at-netflix-566b57d2e362)
- [Selecting the best artwork for videos through A/B testing](https://netflixtechblog.com/selecting-the-best-artwork-for-videos-through-a-b-testing-f6155c4595f6)
- [Round 2: A Survey of Causal Inference Applications at Netflix](https://netflixtechblog.com/round-2-a-survey-of-causal-inference-applications-at-netflix-fd78328ee0bb)
- [Raise the Bar on Shared A/B Tests: Make them trustworthy](https://docs.google.com/document/d/1sRKParLv0UdOsdAJDTKxPRJstkEDP2Rg/edit)
- [It’s All A/Bout Testing: The Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)
- [Lessons from designing Netflix’s experimentation platform](https://www.youtube.com/watch?v=uK-Nf12Qtw8)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)