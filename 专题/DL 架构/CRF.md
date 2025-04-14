[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • Conditional Random Fields

- [Overview](https://aman.ai/primers/ai/conditional-random-fields/#overview)
- [Introduction](https://aman.ai/primers/ai/conditional-random-fields/#introduction)
    - [Label Bias Problem in MEMMs](https://aman.ai/primers/ai/conditional-random-fields/#label-bias-problem-in-memms)
    - [Undirected Graphical Models](https://aman.ai/primers/ai/conditional-random-fields/#undirected-graphical-models)
    - [Linear-chain CRFs](https://aman.ai/primers/ai/conditional-random-fields/#linear-chain-crfs)
    - [Inference](https://aman.ai/primers/ai/conditional-random-fields/#inference)
    - [Parameter Estimation](https://aman.ai/primers/ai/conditional-random-fields/#parameter-estimation)
    - [Wrapping Up: HMM vs. MEMM vs. CRF](https://aman.ai/primers/ai/conditional-random-fields/#wrapping-up-hmm-vs-memm-vs-crf)
    - [CRF Important Observations](https://aman.ai/primers/ai/conditional-random-fields/#crf-important-observations)
- [Software Packages](https://aman.ai/primers/ai/conditional-random-fields/#software-packages)
- [References](https://aman.ai/primers/ai/conditional-random-fields/#references)
- [Citation](https://aman.ai/primers/ai/conditional-random-fields/#citation)

## Overview

- This primer offers an introduction to Linear-Chain Conditional Random Fields, explains what was the motivation behind it’s proposal and makes a comparison with other sequence models such as Hidden Markov Models (HMM) and Maximum Entropy Markov Models (MEMM).
- You can find additional related posts here:
    - [Relationship between Hidden Markov Model and Naive Bayes](https://aman.ai/primers/ai/hmm-and-naive-bayes)
    - [Maximum Entropy Markov Models and Logistic Regression](https://aman.ai/primers/ai/maximum-entropy-markov-models-and-logistic-reg/)

## Introduction

- CRFs were proposed roughly only year after the Maximum Entropy Markov Models, basically by the same authors. Reading through the original [paper that introduced Conditional Random Fields](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers), one finds at the beginning this sentence:

> The critical difference between CRF and MEMM is that the latter uses per-state exponential models for the conditional probabilities of next states given the current state, whereas CRF uses a single exponential model to determine the joint probability of the entire sequence of labels, given the observation sequence. Therefore, in CRF, the weights of different features in different states compete against each other.

- This means that in the MEMMs there is a model to compute the probability of the next state, given the current state and the observation. On the other hand CRF computes all state transitions globally, in a single model.
    
- The main motivation for this proposal is the so called Label Bias Problem occurring in MEMM, which generates a bias towards states with few successor states.
    

### Label Bias Problem in MEMMs

- Recalling how the transition probabilities are computed in a MEMM model, from the previous post, we learned that the probability of the next state is only dependent on the observation (i.e., the sequence of words) and the previous state, that is, we have an exponential model for each state to tell us the conditional probability of the next states.
- The figure below (taken from A. McCallum et al. 2000) shows the MEMM transition probability computation.

![](https://aman.ai/primers/ai/assets/conditional-random-fields/HMM.png)

- This causes the so called **Label Bias Problem**, and Lafferty et al. 2001 demonstrate this through experiments and report it. We will not demonstrate it, but just give the basic intuition taken also from the paper. The figure below (taken from Lafferty et al. 2001) shows the label bias problem.

![](https://aman.ai/primers/ai/assets/conditional-random-fields/Label_Bias_Problem.png)

- Given the observation sequence: **_r_ _i_ _b_**

> In the first time step, r matches both transitions from the start state, so the probability mass gets distributed roughly equally among those two transitions. Next we observe i. Both states 1 and 4 have only one outgoing transition. State 1 has seen this observation often in training, state 4 has almost never seen this observation; but like state 1, state 4 has no choice but to pass all its mass to its single outgoing transition, since it is not generating the observation, only conditioning on it. Thus, states with a single outgoing transition effectively ignore their observations.

> The top path and the bottom path will be about equally likely, independently of the observation sequence. If one of the two words is slightly more common in the training set, the transitions out of the start state will slightly prefer its corresponding transition, and that word’s state sequence will always win.

- Transitions from a given state are competing against each other only.
    
- Per state normalization, i.e. sum of transition probability for any state has to sum to 1.
    
- MEMM are normalized locally over each observation where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.
    
- States with a single outgoing transition effectively ignore their observations.
    
- Causes bias: states with fewer arcs are preferred.
    
- The idea of CRF is to drop this local per state normalization, and replace it by a global per sequence normalization.
    
- So, how do we formalize this global normalization? I will try to explain it in the sections that follow.
    

### Undirected Graphical Models

- A Conditional Random Field can be seen as an undirected graphical model, or Markov Random Field, globally conditioned on XX, the random variable representing observation sequence.
    
- [Lafferty et al. 2001](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) define a Conditional Random Field as:
    
    - XX is a random variable over data sequences to be labeled, and YY is a random variable over corresponding label sequences.
        
    - The random variables XX and YY are jointly distributed, but in a discriminative framework we construct a conditional model p(Y∣X)p(Y∣X) from paired observation and label sequences:
        
- Let G=(V,E)G=(V,E) be a graph such that Y=(Yv)  v∈VY=(Yv)  v∈V, so that YY is indexed by the vertices of GG.
    
- (X,Y)(X,Y) is a conditional random field when each of the random variables YvYv, conditioned on XX, obey the Markov property with respect to the graph:
    
    P(Yv∣X,Yw,w≠v)=P(Yv∣X,Yw,w∼v)P(Yv∣X,Yw,w≠v)=P(Yv∣X,Yw,w∼v)
    
    - where w∼vw∼v means that ww and vv are neighbors in G. Thus, a CRF is a random field globally conditioned on the observation XX. This goes already in the direction of what the MEMM doesn’t give us, states globally conditioned on the observation.
- This graph may have an arbitrary structure as long as it represents the label sequences being modeled, this is also called general Conditional Random Fields.
    
- However the simplest and most common graph structured in NLP, which is the one used to model sequences is the one in which the nodes corresponding to elements of YY form a simple first-order chain, as illustrated in the figure below:
    

The figure below (taken from Hanna Wallach 2004) shows chain-structured CRFs globally conditioned on X.

![](https://aman.ai/primers/ai/assets/conditional-random-fields/Conditional_Random_Fields.png)

- This is also called linear-chain conditional random fields, which is the type of CRF on which the rest of this post will focus.

### Linear-chain CRFs

- Let x¯x¯ is a sequence of words and y¯y¯ a corresponding sequence of nn tags:

P(y¯∣x¯;w¯)=exp(w¯⋅F(x¯,y¯))∑y¯′∈Yexp(w¯⋅F(x¯,y¯′))P(y¯∣x¯;w¯)=exp⁡(w¯⋅F(x¯,y¯))∑y¯′∈Yexp⁡(w¯⋅F(x¯,y¯′))

- This can been seen as another log-linear model, but “giant” in the sense that:
    
    - The space of possible values for y¯y¯, i.e., YnYn, is huge, where nn is the since of the sequence.
    - The normalization constant involves a sum over the set YnYn.
- FF will represent a global feature vector defined by a set of feature functions f1,...,fdf1,...,fd, where each feature function fjfj can analyse the whole x¯x¯ sequence, the current yiyi and previous yi−1yi−1 positions in the y¯y¯ labels sequence, and the current position ii in the sentence:
    

F(x¯,y¯)=∑if(yi−1,yi,x¯,i)F(x¯,y¯)=∑if(yi−1,yi,x¯,i)

- We can define an arbitrary number of feature functions. The _k_’th global feature is then computed by summing the fkfk over all the nn different state transitions y¯y¯. In this way we have a “global” feature vector that maps the entire sequence: F(x¯,y¯)∈IRdF(x¯,y¯)∈IRd.
    
- Thus, the full expanded linear-chain CRF equation is (figure taken from Sameer Maskey slides):
    

![](https://aman.ai/primers/ai/assets/conditional-random-fields/CRF_Equation.png)

- Having the framework defined by the equation above we now analyze how to perform two operations: parameter estimation and sequence prediction.

### Inference

- Inference with a linear-chain CRF resolves to computing the y¯y¯ sequence that maximizes the following equation:

y¯̂ =argmaxy¯ P(y¯∣x¯;w¯)=exp(w¯⋅F(x¯,y¯))∑y¯′∈Yexp(w¯⋅F(x¯,y¯′))y¯^=arg⁡maxy¯ P(y¯∣x¯;w¯)=exp⁡(w¯⋅F(x¯,y¯))∑y¯′∈Yexp⁡(w¯⋅F(x¯,y¯′))

- We want to try all possible y¯y¯ sequences computing for each one the probability of “fitting” the observation x¯x¯ with feature weights w¯w¯. If we just want the score for a particular labelling sequence y¯y¯, we can ignore the exponential inside the numerator, and the denominator:

y¯̂ =argmaxy¯ P(y¯∣x¯;w)=∑jw¯ F(x¯,y¯)y¯^=arg⁡maxy¯ P(y¯∣x¯;w)=∑jw¯ F(x¯,y¯)

- Then, we replace F(x¯,y¯)F(x¯,y¯) by it’s definition:

y¯̂ =argmaxy¯ ∑iw¯ f(yi−1,yi,x¯,i)y¯^=arg⁡maxy¯ ∑iw¯ f(yi−1,yi,x¯,i)

- Each transition from state yi−1yi−1 to state yiyi has an associated score:

w¯ f(yi−1,yi,x¯,i)w¯ f(yi−1,yi,x¯,i)

- Since we took the expexp out, this score could be positive or negative, intuitively, this score will be relatively high if the state transition is plausible, relatively low if this transition is implausible.
    
- The decoding problem is then to find an entire sequence of states such that the sum of the transition scores is maximized. We can again solve this problem using a variant of the Viterbi algorithm, in a very similar way to the decoding algorithm for HMMs or MEMMs.
    
- The denominator, also called the partition function:
    
    Z(x¯,w)=∑y¯′∈Yexp(∑jwjFj(x¯,y¯′))Z(x¯,w)=∑y¯′∈Yexp⁡(∑jwjFj(x¯,y¯′))
    
    - … is useful to compute a marginal probability. For example, this is useful for measuring the model’s confidence in it’s predicted labeling over a segment of input. This marginal probability can be computed efficiently using the forward-backward algorithm. See the references section for demonstrations on how this is achieved.

### Parameter Estimation

- We also need to find the w¯w¯ parameters that best fit the training data, a given a set of labelled sentences:
    
    {(x¯1,y¯1),…,(x¯m,y¯m)}{(x¯1,y¯1),…,(x¯m,y¯m)}
    
    - where each pair (x¯i,y¯i)(x¯i,y¯i) is a sentence with the corresponding word labels annotated. To find the w¯w¯ parameters that best fit the data we need to maximize the conditional likelihood of the training data:
    
    L(w¯)=∑i=1mlogp(x¯1|y¯1,w¯)L(w¯)=∑i=1mlog⁡p(x¯1|y¯1,w¯)
    
- The parameter estimates are computed as:
    
    w¯∗=argmaxw¯ ∈ IRd ∑i=1mlogp(x¯i|y¯i,w¯)−λ2‖w¯‖2w¯∗=arg⁡maxw¯ ∈ IRd ∑i=1mlog⁡p(x¯i|y¯i,w¯)−λ2‖w¯‖2
    
    - where λ2‖w¯‖2λ2‖w¯‖2 is an L2 regularization term.
- The standard approach to finding w¯∗w¯∗ is to compute the gradient of the objective function, and use the gradient in an optimization algorithm like L-BFGS.
    

### Wrapping Up: HMM vs. MEMM vs. CRF

- It is now helpful to look at the three sequence prediction models, and compared them. The figure bellow shows the graphical representation for the Hidden Markov Model, the Maximum Entropy Markov Model and the Conditional Random Fields.
    
- The figure below (taken from Lafferty et al. 2001) shows the graph representation of HMM, MEMM and CRF:
    

![](https://aman.ai/primers/ai/assets/conditional-random-fields/HMM-MEMM-CRF.png)

- **Hidden Markov Models**:

P(y¯,x¯)=∏i=1|y¯|P(yi∣yi−1)⋅P(xi∣yi)P(y¯,x¯)=∏i=1|y¯|P(yi∣yi−1)⋅P(xi∣yi)

- **Maximum Entropy Markov Models**:

P(y¯,x¯)=∏i=1|y¯|P(yi∣yi−1,xi)=∏i=1|y¯|1Z(x,yi−1) exp(∑j=1Nwj⋅fj(x,yi−1))P(y¯,x¯)=∏i=1|y¯|P(yi∣yi−1,xi)=∏i=1|y¯|1Z(x,yi−1) exp⁡(∑j=1Nwj⋅fj(x,yi−1))

- **Conditional Random Fields**:

P(y¯∣x¯,w¯)=exp(w¯⋅F(x¯,y¯))∑y¯′∈Yexp(w¯⋅F(x¯,y¯′))P(y¯∣x¯,w¯)=exp⁡(w¯⋅F(x¯,y¯))∑y¯′∈Yexp⁡(w¯⋅F(x¯,y¯′))

### CRF Important Observations

- MEMMs are normalized locally over each observation, and hence suffer from the Label Bias problem, where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.
    
- CRFs avoid the label bias problem a weakness exhibited by Maximum Entropy Markov Models (MEMM). The big difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally re-normalized.
    
- The inference algorithm in CRF is again based on Viterbi algorithm.
    
- Output transition and observation probabilities are not modelled separately.
    
- Output transition dependent on the state and the observation as one conditional probability.
    

## Software Packages

- [python-crfsuite](https://github.com/scrapinghub/python-crfsuite): is a python binding for [CRFsuite](https://github.com/chokkan/crfsuite) which is a fast implementation of Conditional Random Fields written in C++.
    
- [CRF++: Yet Another CRF toolkit](https://taku910.github.io/crfpp/): is a popular implementation in C++ but as far as I know there are no python bindings.
    
- [MALLET](http://mallet.cs.umass.edu/):includes implementations of widely used sequence algorithms including hidden Markov models (HMMs) and linear chain conditional random fields (CRFs), it’s written in Java.
    
- [FlexCRFs](http://flexcrfs.sourceforge.net/) supports both first-order and second-order Markov CRFs, it’s written in C/C++ using STL library.
    
- [python-wapiti](https://github.com/adsva/python-wapiti) is a python wrapper for [wapiti](http://wapiti.limsi.fr/), a sequence labeling tool with support for maxent models, maximum entropy Markov models and linear-chain CRF.
    

## References

- [“Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data”](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
    
- [“Log-linear models and Conditional Random Fields”. Notes for a tutorial at CIKM’08 by Charles Elkan. October 20, 2008”](https://pdfs.semanticscholar.org/5f5c/171b07540cf739485967cab50fc00dd26ae1.pdf)
    
- [Video: tutorial at CIKM’08 by Charles Elkan](http://videolectures.net/cikm08_elkan_llmacrf/?q=conditional%20random%20fields)
    
- [“Conditional Random Fields: An Introduction”. Hanna M. Wallach, February 24, 2004. University of Pennsylvania CIS Technical Report MS-CIS-04-21](http://dirichlet.net/pdf/wallach04conditional.pdf)
    
- [“Statistical NLP for the Web Log Linear Models, MEMM, Conditional Random Fields” class by Sameer Maskey](http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf)
    
- [“Log-Linear Models, MEMMs, and CRFs”. Michael Collins](http://www.cs.columbia.edu/~mcollins/crf.pdf)
    
- [“An Introduction to Conditional Random Fields” Sutton, Charles; McCallum, Andrew (2010)](https://arxiv.org/pdf/1011.4088v1.pdf)
    

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledConditionalRandomFields,   title   = {Conditional Random Fields for Sequence Prediction},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)