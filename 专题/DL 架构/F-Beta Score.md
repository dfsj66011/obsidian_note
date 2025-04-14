[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • F-Beta Score

- [FβFβ Score](https://aman.ai/primers/ai/f-beta/#f_beta-score)
- [References](https://aman.ai/primers/ai/f-beta/#references)
- [Further Reading](https://aman.ai/primers/ai/f-beta/#further-reading)
- [Citation](https://aman.ai/primers/ai/f-beta/#citation)

## FβFβ Score

- FβFβ score is the generalization of what people loosely call as F1F1 score, which is a harmonic mean of precision and recall. F1F1 is nothing but FβFβ with β=1β=1, where precision and recall are given equal weightage. The beta parameter determines the weight of recall in the combined score. β<1β<1 lends more weight to precision, while β>1β>1 favors recall (β→0β→0 considers only precision, β→+∞β→+∞ only recall).
    
- We use F1F1 when we want to strike a balance on getting an optimum False Positives and optimum False Negatives. What if we want to weigh them differently? The answer is F0.5F0.5 and F2.0F2.0, the two siblings of the F1F1 score.
    
- F0.5F0.5 is FβFβ with β=0.5β=0.5, which is needed when you want to weigh precision more than recall. Why? You need this when your end goal is to reduce False Positives (FP). For e.g., child safety detector for social media content. In this case, even one FP could be disastrous, so optimizing for an FP that is a theoretical zero makes sense.
    
- F2.0F2.0 is FβFβ with β=2.0β=2.0, which is needed when you want to weigh recall more than precision. Why? You need this when your end goal is to reduce False Negatives (FN). For e.g., rare cancer detector. In this case, an FN i.e. failing to catch a rare cancer for a patient who actually has it is disastrous. We are even better off having a higher rate of FPs here because when diagnosed with a rare cancer people tend to get a second checkup/opinion but if you clear them in the 1st test they naturally don’t test again to verify.
    
- The diagram below from [Prithvi Da](https://www.linkedin.com/in/prithivirajdamodaran/) summarizes the aforementioned metrics.
    

![](https://aman.ai/primers/ai/assets/f-beta/f-beta.jpeg)

## References

- [Prithvi Da on LinkedIn](https://www.linkedin.com/posts/prithivirajdamodaran_did-you-know-the-$$F_1$$-score-has-2-siblings-activity-6901841209996787712-8txI)

## Further Reading

- [FβFβ score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledKernelTrick,   title   = {Kernel Trick},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)