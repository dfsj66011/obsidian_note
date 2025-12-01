

 close attention to the section above, you will have noticed a very important detail in the definition of quantum states. When we measure our quantum state, it has a certain probability of existing within any of the basis states (though some of these probabilities may be 0). In other words, if our quantum state is an n-dimensional complex vector, this single state can be used to simultaneously represent n different states! This idea, which is fundamental to quantum mechanics, is called **[superposition](https://en.wikipedia.org/wiki/Quantum_superposition).** While in classical systems our state must exist in one possible state (e.g., a bit within a computer cannot be both 1 and 0 simultaneously), in quantum systems, the state is allowed to be in multiple states at once with a certain probability. When we measure this quantum state, however, it must collapse to one of the possible states in its basis.

_**Consecutive Measurements**_

Once we measure a quantum state, this state collapses to a new state corresponding to one of the basis vectors. So, what happens if we make this same measurement on our state a second time? We will get the same result 100% of the time. Why? After we measure our state the first time, assume (without loss of generality) that this state collapses to the i-th vector in the basis. Then, our new quantum state is represented as follows.

[

![](https://substackcdn.com/image/fetch/$s_!e2ey!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F84c6552b-93c7-48f8-8c39-a6af052c0f5c_800x44.png)



](https://substackcdn.com/image/fetch/$s_!e2ey!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F84c6552b-93c7-48f8-8c39-a6af052c0f5c_800x44.png)

Quantum State after a Measurement

Clearly, making the same measurement on the quantum state shown above will yield a deterministic result (i.e., all of the probabilities are 0 except for one). This highlights a very important point in quantum mechanics. As soon as we make a measurement on our quantum state, the state has collapsed to something different (i.e., the state is modified as a result of the measurement). Therefore, the order and manner in which we perform measurements in quantum mechanics is very important. If we wanted to perform repeated measurements on our original quantum state, we would have to “prepare” this quantum state (i.e., construct a quantum state that somehow exists within this same superposition) each time before making a measurement.

#### **Example of a Quantum State: Qubits**

To solidify the concept of quantum states, I think it is useful to present a concrete example of a quantum system that is (somewhat) simple, but very useful in modern research — the quantum bit, or “qubit”. In a classical computer, we have the notion of bits, which correspond to values of 0 or 1. Each bit must exist in one of these two possible states, and many bits can be combined together to form complex computer systems. The possible states of a bit can easily be represented as follows.

[

![](https://substackcdn.com/image/fetch/$s_!b3Hu!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F6ddfe413-2ba4-48ba-b364-e99a69afc01e_600x174.png)



](https://substackcdn.com/image/fetch/$s_!b3Hu!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F6ddfe413-2ba4-48ba-b364-e99a69afc01e_600x174.png)

Vector Representation of Bits

A qubit is similar to a bit, as it shares the same basis states. However, for qubits we consider complex vector spaces (as opposed to real vector spaces). A single qubit can be represented as follows.

[

![](https://substackcdn.com/image/fetch/$s_!62ri!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcce3a240-f1e9-4500-b79b-cb4c7a64827b_800x127.png)



](https://substackcdn.com/image/fetch/$s_!62ri!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcce3a240-f1e9-4500-b79b-cb4c7a64827b_800x127.png)

Vector Representation of a Qubit

In this equation, the basis vectors are defined identically as for bits. Additionally, if we measure a qubit, the result will be either 0 or 1— same as classical bits. However, a qubit can exist in a superposition of these possible states, allowing it to encode significantly more information than a classical bit. Such a system can be visualized within something called the [Bloch Sphere](https://en.wikipedia.org/wiki/Bloch_sphere), shown below.

[

![](https://substackcdn.com/image/fetch/$s_!7Udx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0f12cad-79f4-4526-8194-bd0947f7d98d_512x581.png)



](https://substackcdn.com/image/fetch/$s_!7Udx!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0f12cad-79f4-4526-8194-bd0947f7d98d_512x581.png)

The Bloch Sphere (source: [http://akyrillidis.github.io/notes/quant_post_7](http://akyrillidis.github.io/notes/quant_post_7))

#### Observables (Measureables)

Based on the definition of quantum states and measurements given so far, you are probably thinking that our possible states (i.e., the basis we use to write our quantum state) must have come out of nowhere. How do we know what the possible states are when we take a measurement? Well, this question is answered in quantum mechanics by the concept of an observable (i.e., sometimes this is called a measureable). The name pretty much explains exactly what this is — measurables (or observables) represent the quantities within a quantum system that can be measured. Concretely, measurables are hermitian [linear operators](https://en.wikipedia.org/wiki/Operator_%28mathematics%29#Linear_operators), as shown by the equation below.

[

![](https://substackcdn.com/image/fetch/$s_!wnEe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7a861aec-3b10-48ef-b9bf-cc505a50b9f8_800x77.png)



](https://substackcdn.com/image/fetch/$s_!wnEe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7a861aec-3b10-48ef-b9bf-cc505a50b9f8_800x77.png)

Hermitian Operator

As can be seen in the equation above, a hermitian matrix is simply a matrix that is equal to its conjugate transpose. Matrices of this form have a few useful properties. Firstly, all of the eigenvalues of a hermitian matrix must be real-valued. Additionally, the eigenvectors of a hermitian matrix are a complete set (i.e., they form an [orthonormal basis](https://en.wikipedia.org/wiki/Orthonormal_basis)). In the equation above, these properties imply that the operator H has a set of normalized eigenvectors that form a basis for the n-dimensional space of complex vectors. Therefore, any vector in this space can be written as a linear combination of the eigenvectors for this hermitian operator. Therefore, given an observable (i.e., a hermitian operator), we can then expand any quantum state that is given to us in a familiar fashion.

[

![](https://substackcdn.com/image/fetch/$s_!dIFB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0e041300-8736-42b3-9a84-3dc7b6af377a_800x138.png)



](https://substackcdn.com/image/fetch/$s_!dIFB!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0e041300-8736-42b3-9a84-3dc7b6af377a_800x138.png)

Vector Representation of a Quantum State

The equation above is the same exact equation presented in the explanation of quantum states. Now, however, this representation should be seen from a slightly different perspective. Given some arbitrary quantum state, we can expand it as a linear combination of the eigenvectors associated with some observable. Then, these eigenvectors correspond to the possible states to which our quantum state could collapse after performing a measurement with said observable. The possible outputs of this measurement (i.e., the values that we get as a result) are the eigenvalues of the observable. In particular, whichever state we collapse to after making a measurement, its corresponding eigenvalue will be measured as a result. Since the eigenvalues of a hermitian operator are known to be real-valued, the result of the measurement will be a real value.

_**How do we know what the result of our measurement will be?**_

Just like before, the answer to this question is non-deterministic. However, as we know from the previous discussion on measurements, we can easily derive the probability that our quantum state will collapse into a certain basis state. Given some observable with associated eigenvectors, we can first expand the quantum state as a linear combination of eigenvectors for this observable. Then, the probability of collapse to any given state, just as before, is given by the squared magnitude of the probability amplitude associated with this state; see below.

[

![](https://substackcdn.com/image/fetch/$s_!wawl!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F53d86335-37ad-4495-a1dd-587e80feb2c4_800x62.png)



](https://substackcdn.com/image/fetch/$s_!wawl!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F53d86335-37ad-4495-a1dd-587e80feb2c4_800x62.png)

Probability of Collapse

#### Creating a Multi-Qubit System

Before moving on to the explanation of entanglement, it is important to understand how simple systems can be combined to form complex systems in quantum mechanics. Generally, multiple systems can be combined together by taking their Kronecker product. Because this statement is quite vague, I will provide a better explanation with the use of a concrete example.

From the previous discussion, we know how to represent a single qubit. However, what if we have two qubits together within a single system? Let us first consider the possible results of a measurement on this two qubit system. If we know the possible results of a measurement (i.e., the basis states for our combined system), then we can write any state of the two qubit system as a linear combination of these basis states. Because each qubit may provide a measurement of either 0 or 1, we have the following possibilities for measurements: 00, 01, 10, and 11. We can construct the vector representation of these combined states by taking the Kronecker product of their components. This is shown in the equation below, where all of the possible states for a two qubit system are constructed.

[

![](https://substackcdn.com/image/fetch/$s_!kVHS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F6fa64520-687a-4cc4-a2b0-b796ca499605_800x186.png)



](https://substackcdn.com/image/fetch/$s_!kVHS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F6fa64520-687a-4cc4-a2b0-b796ca499605_800x186.png)

Possible States for a Two Qubit System

Notice that these states form a basis for the four-dimensional space of complex vectors. The states shown above reveal a more general pattern — if we form a system of n qubits, then this system is capable of representing 2^n states (i.e., in the case above there are 2² = 4 possible states). Given these possible states for our two qubit system, a quantum state within this combined system can be easily expressed as follows.

[

![](https://substackcdn.com/image/fetch/$s_!zZ31!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff2ecad84-7d57-419c-acf5-40573aff3390_800x188.png)



](https://substackcdn.com/image/fetch/$s_!zZ31!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff2ecad84-7d57-419c-acf5-40573aff3390_800x188.png)

Two Qubit Quantum State

Once we have constructed the quantum state as shown above, everything we have learned so far applies (i.e., it is not any different than the general quantum systems we have been discussing so far). Therefore, we now know how smaller systems can be combined to form more complex systems! An understanding of this concept is pivotal for truly grapsing the concept of entanglement.

_**Observables of Complex Systems**_

Similarly to the concept outlined in the example above, if we want to combine observables for smaller systems into single observables of a combined system, we would take their Kronecker product. For example, an identity operator (i.e., this is a hermitian matrix and, therefore, an observable) for a single qubit system is a 2x2 matrix. To form an identity operator for a two qubit system, we would take the Kronecker product of two identity operators, forming a 4x4 identity matrix. This 4x4 identity matrix is an identity operator for the combined, two qubit system. Similar logic applies for different types of observables.

### Entanglement (finally…)

Now, we finally understand enough about quantum mechanics to gain a basic grasp of quantum entanglement. Quantum entanglement assumes that there exists a complex system, composed of several smaller components. Luckily, we just outlined how quantum systems can be combined together, and I will continue using the same running example — a multi-qubit system — to explain entanglement. I will begin by introducing the notion of a product state. In words, a product state is the simplest state in which a combined system can exist. It is formed by taking the product of individual component states within the system. For example, see the equation below, which combines two individual qubit states together to form a resulting product state for a two qubit system.

[

![](https://substackcdn.com/image/fetch/$s_!kLtt!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5a4fa09b-3f72-4668-9d4e-6a909dffd018_800x574.png)



](https://substackcdn.com/image/fetch/$s_!kLtt!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5a4fa09b-3f72-4668-9d4e-6a909dffd018_800x574.png)

Constructing a Two Qubit Product State

In the equation above, the first two rows represent quantum states of individual qubits, while the last row represents the resulting product state. The product state is formed by taking the product of two individual quantum states. In this case, the product of distinct qubit quantum states is taken, resulting in a product state for a two qubit system. In this equation, each of the four entries in the product state’s vector correspond to probability amplitudes associated with each basis state in a two qubit system. Interestingly, if one closely examines the product state, it will become clear that the probabilities associated with measuring either component of the two qubit system are independent. To understand this, take a look at the table below, where I use the same exact product state, but assign concrete values to the probability amplitudes shown above. All values within the tables represent measurement probabilities of a basis state (i.e., the basis state is listed in the top-left corner of each entry).

[

![](https://substackcdn.com/image/fetch/$s_!AHGY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2dfd8c6b-836e-44e1-9092-8bef6d27fb38_800x429.png)



](https://substackcdn.com/image/fetch/$s_!AHGY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2dfd8c6b-836e-44e1-9092-8bef6d27fb38_800x429.png)

Measurement Probabilities for a Product State

Examining the product state in the table above, the probability of measuring the first qubit in the combined system as 0 (or 1) is 0.5 (i.e., same as if you measured the qubit in isolation). The same observation applies to the second qubit. After closely examining the above product state, it becomes clear that if the first qubit is measured as 0 (without loss of generality), the probability of the second qubit being measured as 0 (or 1) is still 0.5. In other words, measuring one qubit in the combined system does not give us any extra information about the value of the other qubit. The measurement probabilities of the two qubits in the combined system do not depend on each other whatsoever — there is no entanglement. This property of product states follows from the fact that all of the probability amplitudes in a product state are expressed as a product of probability amplitudes for each its individual components. However, it is possible to construct states within a combined system that cannot be expressed as product states. For example, consider the following state of a two qubit system.

[

![](https://substackcdn.com/image/fetch/$s_!Kf35!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4b9190bc-dece-4167-8661-45b49645163f_800x450.png)



](https://substackcdn.com/image/fetch/$s_!Kf35!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4b9190bc-dece-4167-8661-45b49645163f_800x450.png)

Non-Product State of a Two-Qubit System

If you try for a long time, you will realize that the above two qubit quantum state cannot be expressed as the product of individual qubit states. We will call this a “non-product state”, and the vector representation of the non-product state shown above can be seen in the equation below.

[

![](https://substackcdn.com/image/fetch/$s_!hzN8!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F914414ab-1b46-4d68-a64f-fa866a2a0b34_566x346.png)



](https://substackcdn.com/image/fetch/$s_!hzN8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F914414ab-1b46-4d68-a64f-fa866a2a0b34_566x346.png)

Vector Representation of a Non-Product State

If we consider the measurement probabilities for the quantum state shown above, we will notice something interesting. Assume that we make a measurement on the first qubit and it collapses to 0. If we then measure the second qubit, what will we get as a result? Interestingly, there is a 100% chance that the second qubit will collapse to 0 when measured (i.e., examine the probabilities in the table above closely!). A similar, but reversed, phenomenon is observed if the first qubit collapses to 1. In other words, as soon as we measure the first qubit, the state of the second qubit is known— _the qubits in our two qubit system are entangled_! Therefore, through this example, we gain a basic theoretical (and practical) understanding of the meaning of entanglement. Namely, entangled quantum states are those that cannot be expressed as products of their individual components, causing measurements of the systems’ components to depend on each other (i.e., measuring one component of the system will provide extra information about the others). Obviously, this idea extends beyond the two qubit example given above, and can be observed within increasingly complex quantum mechanical systems.

Although entanglement might seem underwhelming at first glance, it is important to realize that the entanglement of a system has no dependence on distance. Therefore, if we prepare a two qubit state as shown above, then move each of the qubits in our system extremely far apart (e.g., assume we fly the apparatus containing the information of the second qubit to Mars and keep the other one on earth), as soon as we measure the first qubit, the state of the second qubit will be immediately known/fixed. Therefore, this information (i.e., the outcome of the first qubit’s measurement) travels faster than the speed of light as soon as the first qubit is measured — something famously described by Einstein as “spooky action at a distance”. Although this may seem completely absurd, these properties of quantum entanglement have been [experimentally verified](https://arxiv.org/abs/1303.0614), proving that the nature of reality is somewhat different than what we perceive. For example, does this mean teleportation is possible? I will leave this as something to think about — your guess is as good as mine!

### Conclusion

To summarize, entanglement can be described as a quantum state — composed of multiple, smaller components — with measurement probabilities that are dependent on each other. The components of the system are entangled, causing the measurement outcome of one component to impact the measurement outcomes of other components in the system.

I hope you found this post useful, and please feel free to leave any comments or feedback you have. If you are interested, you can find more about me and my research [here](https://wolfecameron.github.io/).

_Sources and Citations_

1. The cover image is from [here](https://www.cultofmac.com/408216/youll-waste-3-5-days-of-your-life-untangling-apple-headphones/).
    
2. I got most of my understanding of quantum mechanics from [this book](https://www.amazon.com/Quantum-Mechanics-Theoretical-Leonard-Susskind/dp/0465062903), which I highly recommend to anyone who is interested in learning more.
    
3. I really enjoyed the explanations in [this blog post](https://www.quantamagazine.org/entanglement-made-simple-20160428/), which inspired me to write a similar post with a more extensive background explanations.
    
4. I read through this series of blog posts on quantum computing found [here](http://akyrillidis.github.io/notes/) several times (i.e., these are written by my advisor), which helped me to develop a basic understanding of quantum mechanics and qubits.


--------

# Why 0.9? Towards Better Momentum Strategies in Deep Learning.

### How more sophisticated momentum strategies can make deep learning less painful.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Feb 26, 2021

#### **[Thoughts and Theory](https://towardsdatascience.com/tagged/thoughts-and-theory)**

#### How more sophisticated momentum strategies can make deep learning less painful.

[

![](https://substackcdn.com/image/fetch/$s_!cXah!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4c7110e2-13e5-4e2e-a6d5-97ff1ba0e6b4_800x314.png)



](https://substackcdn.com/image/fetch/$s_!cXah!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4c7110e2-13e5-4e2e-a6d5-97ff1ba0e6b4_800x314.png)

(from “Introduction to Optimization” By Boris Polyak)

### Introduction

Momentum is a widely-used strategy for accelerating the convergence of gradient-based optimization techniques. Momentum was designed to speed up learning in directions of low curvature, without becoming unstable in directions of high curvature. In deep learning, most practitioners set the value of momentum to 0.9 without attempting to further tune this hyperparameter (i.e., this is the default value for momentum in many popular deep learning packages). However, there is no indication that this choice for the value of momentum is universally well-behaved.

Within this post, we overview recent research indicating that decaying the value of momentum throughout training can aid the optimization process. In particular, we recommend a novel _Demon_ strategy for momentum decay. To support this recommendation, we conduct a large-scale analysis of different strategies for momentum decay in comparison to other popular optimization strategies, proving that momentum decay with _Demon_ is practically useful.

_**Overview**_

This post will begin with a summary of relevant background knowledge for optimization in deep learning, highlighting the current go-to techniques for training deep models. Following this introduction, the _Demon_ momentum decay strategy will be introduced and motivated. Finally, we will conclude with an extensive empirical analysis of _Demon_ in comparison to a wide scope of popular optimization strategies. _**Overall, we aim to demonstrate through this post that significant benefit can be gained by developing better strategies for handling the momentum parameter within deep learning.**_

### Background

For any deep learning practitioner, it is no surprise that training a model can be computationally expensive. When the hyperparameter tuning process is taken into account, the computational expense of model training is even further exacerbated. For example, some state-of-the-art language models can cost millions of dollars to train on public cloud resources when hyperparameter tuning is considered (see [here](https://arxiv.org/abs/2004.08900) for more details). To avoid such massive training expenses, the deep learning community must discover optimization strategies that _(i)_ facilitate quick convergence, _(ii)_ generalize well, and _(iii)_ are (relatively) robust to hyperparameter tuning.

Stochastic gradient descent with momentum (SGDM) is a widely-used tool for deep learning optimization. In the computer vision (CV) domain, SGDM is used to achieve state-of-the-art performance on several well-known benchmarks. However, the hyperparameters of SGDM are highly-tuned on well-known datasets (e.g., ImageNet) and, as a result, the performance of models trained with SGDM is often sensitive to hyperparameter settings.

To mitigate SGDM’s weaknesses, adaptive gradient-based optimization tools were developed, which adopt a different learning rate for every parameter within the model (i.e., based on the history of first-order gradient information). Although many such adaptive techniques have been proposed, Adam remains the most popular, while variants such as AdamW are common in domains like natural language processing (NLP). Despite their improved convergence speed, adaptive methodologies have historically struggled to achieve comparable generalization performance to SGDM and are still relatively sensitive to hyperparameter tuning. Therefore, even the best approaches for deep learning optimization are flawed — _no single approach for training deep models is always optimal_.

_**What can we do about this?**_

Although no single optimization strategy is always best, we demonstrate that decaying momentum strategies, which are largely unexplored within the deep learning community, offer improved model performance and hyperparameter robustness. In fact, the _Demon_ momentum decay strategy is shown to yield significantly more consistent performance in comparison to popular learning rate strategies (e.g., cosine learning rate cycle) across numerous domains. Although some recent research has explored tuning momentum beyond the naive setting of 0.9, no large-scale empirical analysis has yet been conducted to determine best practices for the momentum parameter during training. _Through this post, we hope to solve this issue and make momentum decay a well-known option for deep learning optimization._

### Optimal Momentum Decay with Demon

Though several options for decaying momentum exist, we recommend the _Demon_ strategy, proposed in _[this paper](https://arxiv.org/abs/1910.04952)_. _Demon_ is extensively evaluated in practice and shown to outperform all other momentum decay schedules. Here, we take time to describe this momentum decay strategy, its motivation, and how it can be implemented in practice.

#### _What is Demon?_

[

![](https://substackcdn.com/image/fetch/$s_!dYbV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5b61438-be14-4fce-bdaa-69392d011158_274x196.png)



](https://substackcdn.com/image/fetch/$s_!dYbV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5b61438-be14-4fce-bdaa-69392d011158_274x196.png)

(Image by Author)

Numerous, well-known decay strategies, including the _Demon_ strategy, are depicted in the figure above, where the x-axis represents the progression of training from beginning to end. Most of these strategies were originally popularized as learning rate decay schedules. In this post, however, we also evaluate the effectiveness of each for momentum decay. In comparison to other decay schedules, _Demon_ waits until the later training stages to significantly decrease the value of momentum. Therefore, for the majority of training, _Demon_ keeps the value of momentum near 0.9, decaying swiftly to zero during later training stages. The _Demon_ strategy is motivated by the idea of decaying the total contribution of a gradient to all future training updates. For a more rigorous description of the motivation for _Demon_, one can refer to the [associated paper](https://arxiv.org/abs/1910.04952), which provides a more extensive theoretical and intuitive analysis. The exact form of the _Demon_ decay schedule is given by the equation below.

[

![](https://substackcdn.com/image/fetch/$s_!iaAT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5f696cad-966a-4408-bf61-31a6b02a610d_460x77.png)



](https://substackcdn.com/image/fetch/$s_!iaAT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5f696cad-966a-4408-bf61-31a6b02a610d_460x77.png)

Demon Decay Schedule (Image by Author)

Here, `t` represents the current iteration index,`T` represents the total number of iterations during training, and beta represents the momentum parameter. Therefore, the above equation yields the value of the momentum parameter at iteration `t` of training for the _Demon_ decay schedule. The initial value for beta represents the starting value of the momentum parameter. In general, beta can be initialized using a value of 0.9, but it is observed in practice that a slightly higher value yields improved performance (e.g., 0.95 instead of 0.9). In all cases, _Demon_ is used to decay the momentum parameter from the initial value at the beginning of training to zero at the end of training.

#### Adding Demon to Existing Optimizers

Although the decay schedule for _Demon_ is not complicated, determining how to incorporate this schedule into an existing deep learning optimizer is not immediately obvious. To aid in understanding how _Demon_ can be used with popular optimizers like SGDM and Adam, we provide a pseudocode description of _Demon_ variants for these optimizers below.

[

![](https://substackcdn.com/image/fetch/$s_!DhHy!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56738c9e-275f-4dfa-a925-2f91a9138a58_800x309.png)



](https://substackcdn.com/image/fetch/$s_!DhHy!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56738c9e-275f-4dfa-a925-2f91a9138a58_800x309.png)

Demon in SGDM and Adam (Image by Author)

As can be seen in the algorithm descriptions above, the SGDM and Adam optimizers are not significantly modified by the addition of _Demon_. In particular, _Demon_ is just used to modify the decay factor for the first moment estimate (i.e., the sliding average over the stochastic gradient during training) within both SGDM and Adam. Nothing else about the optimizers is changed, thus demonstrating that adopting _Demon_ in practice is actually quite simple.

#### Demon in Code

The code for implementing the Demon schedule is also extremely simple. We provide it below in python syntax.

[

![](https://substackcdn.com/image/fetch/$s_!spU2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F8f42afdb-5851-408c-85dd-f5cdeba4c809_800x83.png)



](https://substackcdn.com/image/fetch/$s_!spU2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F8f42afdb-5851-408c-85dd-f5cdeba4c809_800x83.png)

Code for Demon in Python (Image by Author)

### Experimental Analysis of Demon

We evaluate _Demon_ extensively in practice. Experiments are provided on several datasets, such as MNIST, CIFAR-10/100, FMNIST, STL-10, Tiny ImageNet, PTB, and GLUE. Furthermore, _Demon_ is tested with numerous popular model architectures, including ResNets and Wide ResNets, non-residual CNN architectures (i.e., VGG), LSTMs, transformers (i.e., BERT fine-tuning), VAEs, noise conditional score networks, and capsule networks. We conduct baseline experiments using both SGDM and Adam with 10 different variants of learning rate and momentum decay each. Furthmore, baseline experiments are provided for a wide scope of recent optimization strategies such as YellowFin, AMSGrad, AdamW, QHAdam, quasi-hyperbolic momentum, and aggregated momentum. We aim to summarize all of these experiments in the following section and demonstrate the benefit provided by momentum decay with _Demon_.

#### The Big Picture

Across all experiments that were performed (i.e., all datasets, models, and optimization strategy combinations), we record the number of times each optimization strategy yields top-1 or top-3 performance in comparison to all other optimization strategies. Intuitively, such a metric reflects the consistency of an optimization strategy’s performance across models and domains. These performance statistics are given below for the best-performing optimization strategies.

[

![](https://substackcdn.com/image/fetch/$s_!ddni!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4aa516fc-8696-42a5-ac89-92b2c65d332c_800x477.png)



](https://substackcdn.com/image/fetch/$s_!ddni!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4aa516fc-8696-42a5-ac89-92b2c65d332c_800x477.png)

Top-1 and Top-3 Performance Ratios of Different Optimization Strategies (Image by Author)

As can be seen above, _Demon_ yields extremely consistent performance across domains. In particular, it yields the best performance of any optimization strategy in nearly 40% of all experiments — a >20% absolute improvement over the next-best cosine learning rate schedule. Furthermore, _Demon_ obtains top-3 performance in comparison to other optimizers in more than 85% of total experiments, highlighting that _Demon_ still performs well even when it is not the best.

Interestingly, in addition to yielding more consistent performance in comparison to many widely-used optimization strategies, _Demon_ significantly outperforms other schedules for momentum decay. In fact, top-1 performance is never achieved by any other momentum decay strategy across all experimental settings. Such a finding highlights the fact that momentum decay is most effective when the proper decay strategy is chosen. Based on these results, _Demon_ is clearly the best momentum decay strategy of those that were considered.

#### Detailed Experimental Results

_Demon_ has been tested on numerous models and datasets within several different domains. Here, we provide detailed results for all experiments that were run with _Demon_. For each of these experiments, _Demon_ is compared to numerous baseline optimization methods (i.e., outlined at the beginning of this section). It should be noted that the results from the experiments shown below were used to generate the summary statistics outlined above, thus revealing that _Demon_ has by far the most consistent performance of any optimization method that was considered.

[

![](https://substackcdn.com/image/fetch/$s_!iJ1A!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1a18f62d-a83b-4a6c-9e52-0901f5f43484_800x629.png)



](https://substackcdn.com/image/fetch/$s_!iJ1A!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1a18f62d-a83b-4a6c-9e52-0901f5f43484_800x629.png)

(Image by Author)

[

![](https://substackcdn.com/image/fetch/$s_!OmMx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0af7163f-6108-437d-91fd-8656a4dd09a8_800x623.png)



](https://substackcdn.com/image/fetch/$s_!OmMx!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0af7163f-6108-437d-91fd-8656a4dd09a8_800x623.png)

(Image by Author)

#### Hyperparameter Robustness

In addition to accelerating training and generalizing well, optimization strategies that decrease hyperparameter sensitivity are desirable because they can decrease the need for hyperparameter tuning, which is computationally expensive in practice. We evaluate the robustness of _Demon_ to different hyperparameter settings in comparison to both SGDM and Adam optimizers (i.e., arguably the most widely-used optimizers in deep learning) by recording model performance (i.e., test accuracy) across a wide range of possible hyperparameters.

[

![](https://substackcdn.com/image/fetch/$s_!0McQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd706de2b-76d5-4304-98f4-04e076b4be47_800x360.png)



](https://substackcdn.com/image/fetch/$s_!0McQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd706de2b-76d5-4304-98f4-04e076b4be47_800x360.png)

Hyperparameter Robustness Results (Image by Author)

From left to right, the experiments depicted above are Wide ResNet on STL-10 with SGDM, VGG on CIFAR-100 with SGDM, and ResNet-20 on CIFAR10 with Adam. _Demon_ performance is depicted in the top row, while the performance of vanilla optimizer counterparts (i.e., SGDM and Adam) are depicted in the bottom row. Lighter color indicates higher performance and a separate model was trained to generate the measurement within each of the above tiles. For all experimental settings that were tested, it can be seen that utilizing _Demon_ during training yields a noticeably larger band of light color across different hyperparameter settings. For example, on the STL-10 dataset, _Demon_, in addition to achieving better top performance in comparison to SGDM, has an average of 5-6 light-colored tiles in each column, while vanilla SGDM has only 1–3 (roughly). These results demonstrate that _Demon_ yields reasonable performance across a wide scope of hyperparameters, implying that it is more robust to hyperparameter tuning.

#### Other Notable Empirical Results

The experimental support for _Demon_ is vast, and we recommend anyone who is interested in specific experimental metrics and results to refer to [the paper](https://arxiv.org/abs/1910.04952). However, there are a few additional experimental results for _Demon_ that are worth mentioning in particular.

_**Fine-Tuning with Demon on GLUE**_

Transformer models are one of the most computationally expensive models to train in deep learning. To test whether improved transformer performance can be achieved with _Demon_, we fine-tune BERT on several GLUE tasks using both _Demon_ and Adam. The results are shown below.

[

![](https://substackcdn.com/image/fetch/$s_!0Bn0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9fbbb926-6c09-4b9a-938a-59cd2aa0ceea_800x84.png)



](https://substackcdn.com/image/fetch/$s_!0Bn0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9fbbb926-6c09-4b9a-938a-59cd2aa0ceea_800x84.png)

Demon vs. Adam on GLUE (Image by Author)

As can be seen, _Demon_ outperforms Adam for BERT fine-tuning on the GLUE dataset. Furthermore, to achieve these results, no extra fine-tuning was required for _Demon_. We simply employ the same hyperparameters used for other experiments and achieve better performance with minimal effort. This result is interesting, especially because Adam, which is one of the go-to optimizers in the NLP domain, has been tuned extensively on the GLUE dataset.

_**Better Qualitative Results for NCSN**_

[

![](https://substackcdn.com/image/fetch/$s_!TAv4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fab16c5eb-f3d3-43cb-9e3c-21898a88269c_469x148.png)



](https://substackcdn.com/image/fetch/$s_!TAv4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fab16c5eb-f3d3-43cb-9e3c-21898a88269c_469x148.png)

NCSN Inception Score on CIFAR-10 (Image by Author)

We find that _Demon_ is quantitatively outperformed by Adam for [Noise Conditional Score Networks](https://arxiv.org/abs/1907.05600) (NCSN) trained on CIFAR-10, as shown in the table above. However, when the results of models trained with Adam and _Demon_ are qualitatively examined, we notice an interesting pattern.

[

![](https://substackcdn.com/image/fetch/$s_!6X_i!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc49a2393-3dbf-4701-ac82-09bbf966ebec_800x244.png)



](https://substackcdn.com/image/fetch/$s_!6X_i!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc49a2393-3dbf-4701-ac82-09bbf966ebec_800x244.png)

NCSN Results for Adam (left) and Demon (right) (Image by Author)

As can be seen above, the NCSN trained with Adam, despite achieving a slightly improved inception score, produces seemingly unnatural images (i.e., all images appear to have green backgrounds). In comparison, the images produced by the NCSN trained with _Demon_ appear significantly more realistic.

### Conclusion

Most practitioners in the deep learning community set the momentum hyperparameter to 0.9 and forget about it. We argue that this is not optimal, and that significant benefit can reaped by adopting more sophisticated momentum strategies. In particular, we introduce the _Demon_ momentum decay schedule and demonstrate that is yields significantly improved empirical performance in comparison to numerous other widely-used optimizers. _Demon_ is extremely easy to use, and we encourage the deep learning community to try it. For more details not included in this post, feel free to read [the paper](https://arxiv.org/abs/1910.04952) that was written for _Demon_.

Thanks so much for reading this post, any feedback is greatly appreciated. For anyone who is interested in learning more about similar research, the project presented in this post was conducted by the optimization lab at Rice University, Deparment of Computer Science. See [here](http://akyrillidis.github.io/group/) for more details on the lab, which is led by [Dr. Anastasios Kyrillidis](http://akyrillidis.github.io/about/).


------

# Effortless Distributed Training of Ultra-Wide GCNs

### An overview of GIST, a novel distributed training framework for large-scale GCNs.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Jul 02, 2021

[

![](https://substackcdn.com/image/fetch/$s_!3NNc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff57a4ab4-e65c-40cb-8dc8-e97a01d65db3_800x536.png)



](https://substackcdn.com/image/fetch/$s_!3NNc!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff57a4ab4-e65c-40cb-8dc8-e97a01d65db3_800x536.png)

Figure 1: A depiction of the training pipeline for GIST. sub-GCNs divides the GCN model into multiple sub-GCNs. Every sub-GCN is trained by subTrain using mini-batches constructed with the Cluster operation. Sub-GCN parameters are intermittently aggregated into the global model through the subAgg operation. [Figure created by author.]

In this post, I will overview a recently proposed distributed training framework for large-scale graph convolutional networks (GCNs), called graph independent subnetwork training (GIST) [1]. GIST massively accelerates the GCN training process for any architecture and can be used to enable training of large-scale models, which exceed the capacity of a single GPU. I will aim to cover the most pivotal aspects of GIST within this post, including relevant background information, a comprehensive description of the training methodology, and details regarding the experimental validation of GIST. A full, detailed manuscript was written to describe GIST and is available [on Arxiv](https://arxiv.org/abs/2102.10424). Furthermore, the source code for all experiments performed with GIST is publicly available [on GitHub](https://github.com/wolfecameron/GIST).

### Introduction

Machine learning and deep learning have been already popularized through their many applications to industrial and scientific problems (e.g., self-driving cars, recommendation systems, person tracking, etc.), but machine learning on graphs, which I will refer to as graphML for short, has just recently taken the spotlight within computer science and artificial intelligence research. Although many reasons for the popularization of graphML exist, a primary reason is the simple fact that not all data can be encoded in the Euclidean space. Graphs are a more intuitive data structure in numerous applications, such as social networking (i.e., nodes of the graph are people and edges represent social connections) or chemistry (i.e., nodes of the graph represent atoms and edges represent chemical bonds). As such, generalizing existing learning strategies on Euclidian data (e.g., convolutional neural networks, transformers, etc.) to work on graphs is a problem of great value.

Towards this goal, several (deep) learning techniques have been developed for graphs, the most popular of which is the graph convolutional network (GCN) [2]. The GCN implements a generalization of the convolution operation for graphs, inspired by a first-order approximation of spectral graph convolutions. Despite the popularity of the GCN and its widespread success in performing node and graph-level classification tasks, the model is notoriously inefficient and difficult to scale to large graphs. Such an issue catalyzed the development of node partitioning techniques, including both neighborhood sampling (e.g., LADIES and FastGCN) and graph partitioning (e.g., ClusterGCN and GraphSAGE), that divide large graphs into computationally-tractable components. Nonetheless, the data used within graphML research remains at a relatively small scale, and most GCN models are limited in size due to the problem of oversmoothing in deeper networks [3]. Such use of smaller data and models in graphML experimentation is in stark contrast to main stream deep learning research, where experimental scale is constantly expanding.

To bridge the gap in scale between deep learning and graphML, GIST aims to enable experiments with larger models and datasets. GIST, which can be used to train any GCN architecture and is compatible with existing node partitioning techniques, operates by decomposing a global GCN model into several, narrow sub-GCNs of equal depth by randomly partitioning the hidden feature space within the global model. These sub-GCNs are then distributed to separate GPUs and trained independently and in parallel for several iterations prior to having their updates aggregated into the full, global model. Then, a new group of sub-GCNs is created/distributed, and the same process is repeated until convergence. In cases of very large graphs, we adopt existing graph partitioning approaches to form mini-batches, which allows GIST to train GCN models on arbitrarily large graphs.

Put simply, GIST aims to provide a distributed training framework for large-scale GCN experiments with minimal wall-clock training time. Furthermore, because GIST trains sub-GCNs instead of ever training the global model directly, it can be used to train models with extremely large hidden layers that exceed the capacity of a single GPU (e.g., we use GIST to train a “ultra-wide” 32,768-dimensional GraphSAGE model on Amazon2M). It should be noted that we choose to focus on scaling model width, rather than depth, due to the fact that deep GCN models are known to suffer from oversmoothing [3].

### What’s the GIST?

Here, we explain the general training methodology employed by GIST. This training methodology, which aims to enable fast-paced, large-scale GCN experimentation, is compatible within any GCN architecture or sampling methodology. We assume in our explanation that the reader has a general understanding of the GCN architecture. For a comprehensive overview of the GCN architecture, we recommend [this article](https://tkipf.github.io/graph-convolutional-networks/). A global view of the GIST training methodology is provided in Figure 1, and we further explain each component of this methodology within the following sections.

#### Creating Sub-GCNs

[

![](https://substackcdn.com/image/fetch/$s_!U5jS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F25b25e18-4d15-4bff-b25c-2ed286f85677_800x307.png)



](https://substackcdn.com/image/fetch/$s_!U5jS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F25b25e18-4d15-4bff-b25c-2ed286f85677_800x307.png)

Figure 2: GCN partitioning with m = 2. Orange and blue colors represent different features partitions. Both hidden dimensions (d1 and d2) are partitioned. The output dimension (d3) is not partitioned, and partitioning the input dimension (d0) is optional. GIST does not partition the input dimension. [Figure created by author.]

The first step in training a GCN model with GIST is partitioning the hidden dimensions of the global model to form multiple, narrow sub-GCNs of equal depth (i.e., `sub-GCNs` in Figure 1). The number of sub-GCNs, denoted as `m`, must be known prior to partitioning. Then, for each hidden layer of the global model, the indices of neurons within the layer are randomly partitioned into `m` disjoint groups of equal size, corresponding to different sub-GCNs. Once these partitions are constructed, a sub-GCN weight matrix at an arbitrary layer can be constructed by indexing the rows of the global weight matrix with the partition indices from the previous layer and columns of the global weight matrix with partition indices of the current layer. As such, this partitioning methodology creates smaller weight matrices for each sub-GCN that correspond to the random feature partition that has been selected.

The above methodology, depicted in Figure 2, is performed for all layers of the global GCN model, but the input and output dimensions are excluded from partitioning. The input dimension (i.e., d0 in Figure 2) is not partitioned because it would result in each sub-GCN having access only to a portion of the input vector for each node, which causes a drastic performance decrease with larger values of `m`. Similarly, the output dimension of the global model (i.e., d3 in Figure 2) is not partitioned so that each sub-GCN produces an output vector of the same size. As a result, no modification to the loss function is needed, and all sub-GCNs can be trained to minimize the same global loss function.

Once sub-GCNs are constructed, they are each sent to separate GPUs to be trained independently and in parallel. It should be noted that the full model is never communicated (i.e., only sub-GCNs are communicated between devices), which drastically improves the communication efficiency of distributed training with GIST. The process of sub-GCN partitioning is illustrated within the Figure 2, where different sub-GCN partitions are denoted with orange and blue colors. Recall that the input and output dimensions are not partitioned within GIST, which is shown in (b) of Figure 2.

#### Training Sub-GCNs

After sub-GCNs are constructed and sent to their respective devices, they are each trained independently and in parallel for a set number of iterations (i.e., `subTrain` in Figure 1), referred to as local iterations. When sub-GCNs have completed their local iterations, the parameter updates of each sub-GCN are aggregated into the global model, and a new group of sub-GCNs is created. This process repeats until convergence. As previously mentioned, sub-GCNs are trained to minimize the same global loss function. Additionally, each sub-GCN is trained over the same data (i.e., no non-iid partition of the data across devices is assumed).

To ensure the total amount of training is kept constant between models trained with GIST and using standard, single-GPU methodology, GCN models trained with GIST have the total number of training epochs split across sub-GCNs. For example, if a vanilla, baseline GCN model is trained for 10 epochs using a single GPU, then a comparable GCN model trained with GIST using two sub-GCNs would conduct 5 epochs of training for each sub-GCN. Because sub-GCNs are trained in parallel, such a reduction in the number of training epochs for each sub-GCN results in a large training acceleration.

If the training graph is small enough, sub-GCNs conduct full-batch training in parallel. However, in cases where the training graph is too large for full-batch training to be performed, a graph partitioning approach is employed to decompose the training graph into smaller, computationally-tractable sub-graphs as a pre-processing step (i.e., `Cluster` in Figure 1). These sub-graphs are then used as mini-batches during independent training iterations, which loosely reflects the training approach proposed by clusterGCN [4]. Although any partitioning approach can be used, GIST employs METIS due to its proven efficiency on large-scale graphs [5].

#### Aggregating Sub-GCNs

After sub-GCNs complete independent training, their parameters must be aggregated into the global model (i.e., `subAgg` in Figure 1) before another independent training round with new sub-GCNs may begin. Such aggregation is performed by simply copying the parameters of each sub-GCN into their corresponding locations within the global model. No collisions occur during this process due to the disjointness of the feature partition created within GIST. Interestingly, not all parameters are updated within each independent training round. For example, within (b) of Figure 2, only overlapping orange and blue blocks are actually partitioned to sub-GCNs, while other parameters are excluded from independent training. Nonetheless, if sufficient independent training rounds are conducted, all parameters within the global model should be updated multiple times, as each training round utilizes a new random feature partition.

### Why is GIST useful?

At first glance, the GIST training methodology may seem somewhat complex, causing one to wonder why it should be used. In this section, I outline the benefits of GIST and why it leads to more efficient, large-scale experimentation on graphs.

#### Architecture-Agnostic Distributed Training

GIST is a distributed training methodology that can be used for any GCN architecture. In particular, GIST is used to train vanilla GCN, GraphSAGE, and graph attention network (GAT) architectures within the original manuscript, but GIST is not limited to these models. Therefore, it is a generic framework that can be applied to accelerate the training of any GCN model.

#### Compatibility with Sampling Methods

The feature partitioning strategy within GIST is orthogonal to the many node partitioning strategies that have been proposed for efficient GCN training. Therefore, any of these strategies can be easily combined with GIST for improved training efficiency. For example, graph partitioning is used to enable training of GCNs over larger graphs with GIST, and GIST is even used to train GraphSAGE models. Such experiments demonstrate the compatibility of GIST with existing approaches for graph and neighborhood sampling.

#### Ultra-Wide GCN training

GIST indirectly updates the global GCN model through the training of smaller sub-GCNs, which enables models with extremely large hidden dimensions (i.e., exceeding the capacity of a single GPU) to be trained. For example, when training a GCN model with GIST using 8 sub-GCNs, the model’s hidden dimension can be made roughly 8X larger in comparison to models at the capacity limit of a single GPU. Such a property enables the training of “ultra-wide” GCN models with GIST, as is demonstrated in experiments with GIST.

#### Improved Model Complexity

GIST reduces both communication and computational complexity of distributed GCN training significantly, resulting in a drastic acceleration of wall-clock training time. Such a complexity reduction is created by the fact that only sub-GCNs, which are significantly smaller than the global model, are communicated and trained by GIST. More precise expressions for the complexity reductions provided by GIST are available within the originaln manuscript.

### How does GIST perform in practice?

Within this section, I overview the experiments performed using GIST, which validate its ability to train GCN models to high performance with significantly reduced wall-clock time. Experiments are performed over numerous datasets, including Cora, Citeseer, Pubmed, OGBN-Arxiv, Reddit, and Amazon2M. However, I focus upon experiments with Reddit and Amazon2M within this post, as these datasets are much larger and more relevant to practical graphML applications. The smaller datasets are mostly used as design/ablation experiments for the GIST methodology, and more details are available within the manuscript.

#### Reddit Dataset

[

![](https://substackcdn.com/image/fetch/$s_!XUY_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F320e8a72-338d-42c0-a62d-c78a7448d2e7_800x400.png)



](https://substackcdn.com/image/fetch/$s_!XUY_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F320e8a72-338d-42c0-a62d-c78a7448d2e7_800x400.png)

Figure 3: F1 Score and wall-clock training time of GraphSAGE and GAT models trained with both GIST and standard, single-GPU methodology on the Reddit dataset. [Figure created by author.]

Experiments with GIST on the Reddit dataset are performed with 256-dimensional GraphSAGE and GAT models with two to four layers. Models are trained with GIST using multiple different numbers of sub-GCNs, where each sub-GCN is assumed to be distributed to a separate GPU (i.e., 8 sub-GCN experiments utilize 8 GPUs in total). 80 epochs of total training are performed using the Adam optimizer and no weight decay, and the number of local iterations is set to 500. The training graph is partitioned into 15,000 sub-graphs during training. Baseline models are trained using standard, single-GPU methodology, and all other experimental details are held constant. As can be seen in Figure 3, all models trained with GIST achieve performance that matches or exceeds that of models trained with standard, single-GPU methodology. Additionally, the training time of GIST is significantly reduced in comparison to standard, single-GPU training.

#### Amazon2M Dataset

[

![](https://substackcdn.com/image/fetch/$s_!oLXB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1c296cfb-3abc-49a5-ba5b-c86bc06e9c99_800x412.png)



](https://substackcdn.com/image/fetch/$s_!oLXB!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1c296cfb-3abc-49a5-ba5b-c86bc06e9c99_800x412.png)

Figure 4: F1 score and wall-clock training time of GraphSAGE models with different hidden dimensions and numbers of layers trained with both GIST and standard, single-GPU methodology on the Amazon2M dataset. [Figure created by author.]

Experiments with GIST on the Amazon2M dataset are performed using GraphSAGE models with hidden dimensions of 400 and 4096 (i.e., narrow and wide models) and different numbers of layers. Again, models are trained with GIST using multiple different numbers of sub-GCNs and the training graph is decomposed into 15,000 partitions. Baseline experiments are performed using standard, single-GPU training methodology. Training is conducted using the Adam optimizer with no weight decay for 400 total epochs, and the number of local iterations is set to 5,000. As can be seen in Figure 4, models trained with GIST complete training significantly faster in comparison to baseline models trained with standard, single-GPU methodology. Furthermore, models trained with GIST perform comparably to those trained with standard methodology in all cases.

#### Ultra-Wide GCNs

[

![](https://substackcdn.com/image/fetch/$s_!hn14!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4a84b646-64e0-4b8a-bfae-a634e1654f2f_800x379.png)



](https://substackcdn.com/image/fetch/$s_!hn14!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4a84b646-64e0-4b8a-bfae-a634e1654f2f_800x379.png)

Figure 5: Performance metrics for GraphSAGE models of numerous hidden dimensions between 400 and 32,768 trained with both GIST and single-GPU methodology on the Amazon2M dataset. Cases marked with “OOM” caused an out-of-memory error during training. [Figure created by author.]

As previously mentioned, GIST can be used to train incredibly-wide GCN models due to the fact that the global model is indirectly updated through the independent training of sub-GCNs. To demonstrate this capability, GraphSAGE models with increasingly-large hidden dimensions are trained over the Amazon2M dataset. As is shown in Figure 5, GIST can be used to train GraphSAGE models with hidden dimensions as large as 32,768 to high-performance on the Amazon2M dataset with relatively minimal training time. Single-GPU training methodologies reach an out-of-memory error in these cases (even in GCN models that are significantly smaller), thus demonstrating that GIST can be used to train models that far-exceed the capacity of a single GPU. Furthermore, the wall-clock training time of models trained with only a single GPU becomes quite prohibitive in comparison to models trained with GIST, thus highlighting its ability to accelerate large-scale GCN experiments. As demonstrated through these experiments, GIST enables GCN experimentation at scales that were previously not feasible.

### Conclusion

In this blog post, I outlined GIST, a novel distributed training methodology for large GCN models. GIST operates by partitioning a global GCN model into several, narrow sub-GCNs that are distributed across separate GPUs and trained independently and in parallel before having their parameters aggregated into the global model. GIST can be used to train any GCN architecture, is compatible with existing sampling methodologies, and can yield significant accelerations in training time without decreasing model performance. Furthermore, GIST is capable of enabling training of incredibly-wide GCN models to state-of-the-art performance, such as a 32,768-dimensional GraphSAGE model on the Amazon2M dataset.

I truly appreciate your interest in this blog post. If you have any comments or questions, feel free to contact me or leave a comment (contact information is available on [my website](https://wolfecameron.github.io/)). GIST was developed as part of the independent subnetwork training (IST) initiative within [my research lab](http://akyrillidis.github.io/group/) at Rice University. More information about related projects can be found [here](https://akyrillidis.github.io/ist/).

### _Citations_

[1] [https://arxiv.org/abs/2102.10424](https://arxiv.org/abs/2102.10424)

[2] [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)

[3] [https://arxiv.org/abs/1801.07606](https://arxiv.org/abs/1801.07606)

[4] [https://arxiv.org/abs/1905.07953](https://arxiv.org/abs/1905.07953)

[5] [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.106.4101](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.106.4101)


------

# The Quantum Approximate Optimization Algorithm from the Ground Up

### Understanding QAOA, its motivation, and how it was derived from past algorithms in the quantum computing community.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Jul 16, 2021

[

![](https://substackcdn.com/image/fetch/$s_!ME9N!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F05e06d4c-ca19-41c6-b1c3-b9db14f4d840_778x519.jpeg)



](https://substackcdn.com/image/fetch/$s_!ME9N!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F05e06d4c-ca19-41c6-b1c3-b9db14f4d840_778x519.jpeg)

[Source](https://www.newscientist.com/article/2252933-quantum-computers-may-be-destroyed-by-high-energy-particles-from-space/)

### **Introduction**

In recent years, the advent of quantum computing has brought great excitement within numerous research communities (e.g., physics, computer science, mathematics, etc.). Interest in the topic of quantum computing led to numerous theoretical contributions in the space, catalyzing the development of influential algorithms such as [Grover’s search](https://en.wikipedia.org/wiki/Grover%27s_algorithm), [Shor’s algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm), [adiabatic optimization](https://en.wikipedia.org/wiki/Adiabatic_quantum_computation), and many more. Despite these theoretical demonstrations of quantum supremacy and usefulness, such algorithms go beyond the current reality of quantum hardware, utilizing an excessive circuit depth and number of qubits. As of now, such requirements are unrealizable within a physical system.

Currently, quantum computing is said to exist within the Noisy Intermediate-Scale Quantum (NISQ) era, in which hardware is limited in the number of available qubits and circuit depth is limited by increasing noise that arises within a quantum system. As such, many of the most eminent quantum algorithms (such as those mentioned above) have limited practical applications, leading researchers to focus on the development of NISQ-friendly quantum algorithms. Two of the most widely-studied algorithms in this space are the variational quantum eigensolver (VQE) and the quantum approximate optimization algorithm (QAOA), which are both variational quantum algorithms (VQA) that use parametric quantum circuits (with parameters optimized classically) to optimize functions (e.g., minimum eigenvalue/eigenvector pair, maximum cut on a graph, maximum vertex cover on a graph, etc.). In this post, I aim to explain QAOA in depth, assuming only a mild background understanding of quantum computing (i.e., qubits, quantum states, and basic linear algebra); see [this blog post](https://www.mustythoughts.com/variational-quantum-eigensolver-explained) if you are interested to learn more about VQE.

I will begin by explaining QAOA at a high level, including the components of the algorithm and how it can be used to solve problems of interest. Following an initial description of QAOA, I will overview numerous concepts in quantum mechanics and mathematics that are relevant to developing an in-depth understanding of QAOA. Once such background is clear, I will explain the details of the adiabatic algorithm for quantum computing, which is intricately related to QAOA, in an attempt to understand how QAOA is related to the algorithms that preceded it. At the end of the post, I will return to a discussion of QAOA, aiming to motivate the components of the algorithm by drawing connections to the adiabatic algorithm for quantum computing.

### The Quantum Approximate Optimization Algorithm

QAOA is a variational quantum algorithm (i.e., a hybrid quantum-classical algorithm involving parametric quantum circuits that are classically optimized) that was proposed for finding approximate solutions to [combinatorial optimization problems](https://en.wikipedia.org/wiki/Combinatorial_optimization) (e.g., MaxCut, Maximum Vertex Cover, MaxSAT, etc.). At a high level, QAOA applies a series of parametric quantum gates to some initial state, and optimizes the parameters of these gates such that the final state (after the quantum gates have been applied) encodes an approximate solution to some problem of interest (i.e., if you take a measurement of the final state with respect to the [computational basis](https://quantumcomputing.stackexchange.com/questions/1410/what-is-meant-by-the-term-computational-basis) it will produce a high-quality solution). The QAOA framework involves several components. Here, I simply define the relevant components and illustrate how they are combined together without deeply considering their motivation, thus providing a basic understanding of QAOA as a whole.

#### Cost and Mixer Hamiltonians

Assume we are studying a quantum system with `n` total [qubits](http://akyrillidis.github.io/notes/quant_post_7). Within QAOA, there are two [Hamiltonians](https://en.wikipedia.org/wiki/Hamiltonian_%28quantum_mechanics%29) of interest — the mixer Hamiltonian and the cost Hamiltonian. Each of these Hamiltonians are complex-valued matrices of dimension `2^n x 2^n`. But, to understand their purpose, we must first understand the role of a Hamiltonian in a quantum system. In quantum mechanics, a Hamiltonian is a [hermitian matrix](https://en.wikipedia.org/wiki/Hermitian_matrix) that encodes the energy of a quantum system. In particular, the eigenvalues of the Hamiltonian, which are real-valued because the matrix is hermitian, represent the possible energy levels of the system (i.e., the set of possible outcomes when measuring the system’s total energy).

A common pattern within quantum algorithms is to encode a problem of interest (e.g., minimization/maximization of some function) within a Hamiltonian matrix and find a way to generate a quantum state that, when measured with respect to that Hamiltonian, produces the lowest/highest possible energy. Within QAOA, the cost Hamiltonian, which we will denote as `H_C` is constructed in this way, encoding the solution to some problem of interest (e.g., a MaxCut or MaxSAT instance) within its lowest energy state. For QAOA, the cost Hamiltonian is typically constructed using [Ising models](https://en.wikipedia.org/wiki/Ising_model), which can be used to encode combinatorial optimization problems as Hamiltonians with single-qubit [Pauli-Z](https://en.wikipedia.org/wiki/Pauli_matrices) terms. Although we will not detail the construction of cost Hamiltonians with Ising models in this post, [this paper](https://arxiv.org/pdf/1302.5843.pdf) covers the topic pretty extensively. Unlike the cost Hamiltonian, the mixer Hamiltonian is unrelated to the problem definition and is defined as follows.

[

![](https://substackcdn.com/image/fetch/$s_!qXFz!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1de6fc00-da9a-4ca0-becb-7c7091b0e6cf_211x99.png)



](https://substackcdn.com/image/fetch/$s_!qXFz!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1de6fc00-da9a-4ca0-becb-7c7091b0e6cf_211x99.png)

Default mixer Hamiltonian for QAOA

In the equation above, all `sigma` variables correspond to single qubit [Pauli-X](https://en.wikipedia.org/wiki/Pauli_matrices) matrices on the `i`-th qubit. One important property of the mixer Hamiltonian is that it should not [commute](https://en.wikipedia.org/wiki/Commuting_matrices) with the cost hamiltonian. If these matrices do not commute, then they cannot share any eigenvectors, which prevents the QAOA ansatz (see below) from getting stuck in non-optimal states with higher energy.

#### Initial State

From here, we must also define the initial quantum state used within QAOA, which is given below.

[

![](https://substackcdn.com/image/fetch/$s_!qNkb!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7b92e6de-3daa-487b-9b17-726916692fd6_286x91.png)



](https://substackcdn.com/image/fetch/$s_!qNkb!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7b92e6de-3daa-487b-9b17-726916692fd6_286x91.png)

Default initial state for QAOA

Here, `z` is used to denote the computational basis for our quantum system. This initial state may also be written as follows (i.e., the two definitions are identical).

[

![](https://substackcdn.com/image/fetch/$s_!6DVy!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3fdc8c57-a5a8-4a20-a43c-c30444e77305_681x83.png)



](https://substackcdn.com/image/fetch/$s_!6DVy!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3fdc8c57-a5a8-4a20-a43c-c30444e77305_681x83.png)

Alternate definition of the intial state for QAOA

As can be seen above, the initial state within QAOA is simply a product of maximally-superposed states across each qubit. Interestingly, this initial state is the maximum eigenvector of the mixer Hamiltonian — it will become clear why this is important later in the post. Additionally, this state is extremely easy to construct — just apply a [Hadamard gate](https://qiskit.org/textbook/ch-states/single-qubit-gates.html#hgate) separately to each qubit within the system — which makes it a desirable choice if QAOA should be implemented on actual quantum hardware.

#### QAOA Ansatz

Now, we have defined all relevant components of QAOA, but it is unclear how these components can be combined together to solve an optimization problem. As previously mentioned, QAOA evolves the initial state using a series of parametric quantum gates, where each quantum gate is described as multiplication of the quantum state by a [unitary](https://en.wikipedia.org/wiki/Unitary_matrix) matrix. Quantum gates are always unitary to ensure that evolution of the quantum state is adiabatic and that the [normalization property](http://farside.ph.utexas.edu/teaching/qmech/Quantum/node34.html) of the quantum state is preserved. The evolution of the initial state in QAOA is described by the following [ansatz](https://pennylane.ai/qml/glossary/circuit_ansatz.html), which produces a final state that (hopefully) encodes an approximately optimal solution to the problem of interest.

[

![](https://substackcdn.com/image/fetch/$s_!2-76!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc6313908-b423-4a61-b2a1-3a92853a4558_800x83.png)



](https://substackcdn.com/image/fetch/$s_!2-76!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc6313908-b423-4a61-b2a1-3a92853a4558_800x83.png)

QAOA Ansatz

Here, `beta` and `gamma` are real-valued, scalar parameters. The QAOA ansatz, containing `2p` parameters in total, is comprised of a series of alternating [unitary](https://en.wikipedia.org/wiki/Unitary_matrix) gates with different scalar parameters. As can be seen, the “depth” of this ansatz is `p`, as the alternating gate structure is applied to the initial state `p` times consecutively. The unitary matrices that evolve the intial state are constructed by taking [matrix exponentials](https://en.wikipedia.org/wiki/Matrix_exponential) of the cost and mixer Hamiltonians (i.e., the exponential of a hermitian matrix is always unitary). Therefore, QAOA, using the above ansatz, produces a new quantum state by applying a series of parametric unitary gates, based on the cost and mixer hamiltonians, to an initial state. If we want this final state to encode a solution to our problem of interest, all we have to do is set `beta` and `gamma` parameters such that the output of QAOA is a low energy state of the cost Hamiltonian. But… how do we find the correct parameters?

#### Optimizing QAOA Parameters

The expected energy of our final state with respect to the cost Hamiltonian can be expressed as follows, where `Tr()` denotes the [trace](https://mathworld.wolfram.com/MatrixTrace.html) of a matrix.

[

![](https://substackcdn.com/image/fetch/$s_!Z9mr!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c48dbff-1e18-46b9-b00e-a5816ad71de8_800x26.png)



](https://substackcdn.com/image/fetch/$s_!Z9mr!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c48dbff-1e18-46b9-b00e-a5816ad71de8_800x26.png)

Expected energy of the QAOA output state

This expected value can be computed in practice by taking repeated, identical measurements of the state produced by the QAOA ansatz. Then, using a classical computer, the gradient of this function can be computed (e.g., using the [parameter shift rule](https://pennylane.ai/qml/glossary/parameter_shift.html)), allowing the `beta` and `gamma` parameters to be updated in the direction that mimizes the expectation with respect to the cost Hamiltonian. By iteratively repeating this process, the parameters of the QAOA ansatz are updated such that the energy of the final state is minimized. As a result, an approximately optimal solution can be generated by producing this final state and taking a measurement with respect to the computational basis (i.e., this can be done multiple times to ensure a good solution is found, due to the fact that measurements in quantum systems are stochastic). Note that the solution provided by QAOA is approximate, and not guaranteed to be globally optimal.

### Background Information for Understanding QAOA

Now, we understand what QAOA is, but the motivation behind its components is unclear. If you are like me, the constructions within QAOA may seem arbitrary, leaving you wondering why it works and how someone even devised such an algorithm. In order to clarify some of this confusion, there is some background information that must be understood, which I aim to outline in this section.

#### Matrix Exponentials

Although matrix exponentials were briefly outlined in the description of QAOA, it is worthwhile to futher outline some of their major properties, due to their relevance to the QAOA ansatz. As stated before, the exponential of a Hermitian matrix is always unitary. In fact, any unitary matrix can be written in the following form.

[

![](https://substackcdn.com/image/fetch/$s_!M2ps!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7cca1860-0caf-42c8-8ee3-9a1d66fa45af_142x34.png)



](https://substackcdn.com/image/fetch/$s_!M2ps!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7cca1860-0caf-42c8-8ee3-9a1d66fa45af_142x34.png)

Unitary matrix as an exponential

where `H` is an arbitrary hermitian matrix, `i` is the imaginary unit, and `gamma` is an arbitrary scalar parameter. Typically, quantum gates are constructed by performing time evolution of a quantum state with respect to some Hamiltonian of interest. Such time evolution takes the form of a unitary matrix as shown below, where `h` denotes the Planck constant.

[

![](https://substackcdn.com/image/fetch/$s_!qiUn!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4850945-a6e7-43de-a503-5980b04b7a52_278x43.png)



](https://substackcdn.com/image/fetch/$s_!qiUn!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4850945-a6e7-43de-a503-5980b04b7a52_278x43.png)

Time evolution with respect to an arbitrary Hamiltonian

As demonstrated by the expression above, an intricate connection exists between the evolution of a quantum state and the (exponentiated) Hamiltonian matrix. Matrix exponentials of Hamiltonians arise often as a way of performing time evolution of a quantum state within some system of interest. Additionally, the scalar parameter used within the matrix exponent can be interpreted as the total time for which such evolution is performed. In the case of QAOA, evolution occurs with respect to the cost and mixer Hamiltonians in an alternating fashion and we perform classical optimization to determine the optimal times for which each of our alternating unitary gates are applied.

Aside from their conection to the evolution of quantum state, there is one more property of matrix exponentials that will be useful. If two matrices commute with each other, then the following property of matrix exponentials is true.

[

![](https://substackcdn.com/image/fetch/$s_!1nps!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffbf2969f-ebfd-4db1-9312-cdf5f5ce2f84_209x34.png)



](https://substackcdn.com/image/fetch/$s_!1nps!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffbf2969f-ebfd-4db1-9312-cdf5f5ce2f84_209x34.png)

This property becomes relevant when we discuss Trotterization later in the post. Additionally, based on this property, one may notice that the order of unitary matrices within the QAOA ansatz would be irrelevant if the cost and mixer Hamiltonians commute, thus providing extra reasoning regarding the requirement that the two Hamiltonians must not commute.

#### The Schrödinger Equation

It may seem like the above expression for time evolution with respect to some Hamiltonian came out of nowhere, but it is actually derived from the [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) — an extremely famous partial differential equation for expressing quantum states and how they change over time. There are many ways that this equation can be written, but for the purposes of this blog post I will express it as follows.

[

![](https://substackcdn.com/image/fetch/$s_!4y4C!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fadd7c751-9e39-4e37-bc2b-0ee6aba16a0e_269x75.png)



](https://substackcdn.com/image/fetch/$s_!4y4C!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fadd7c751-9e39-4e37-bc2b-0ee6aba16a0e_269x75.png)

Schrödinger Equation

Here, the Schrödinger equation describes the dynamics of a quantum state at time `t` with respect to some Hamiltonian `H`. Interestingly, if `H` is time-independent (i.e., the Hamiltonian does not change with respect to the time `t`), the above equation can be solved to yield the following.

[

![](https://substackcdn.com/image/fetch/$s_!0Vaf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5ec6cf2d-1c84-4a0f-a661-d3e469d6ef7c_283x43.png)



](https://substackcdn.com/image/fetch/$s_!0Vaf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5ec6cf2d-1c84-4a0f-a661-d3e469d6ef7c_283x43.png)

Schrödinger equation with time-independent Hamiltonian

As can be seen, the unitary matrix given by the Schrödinger equation to evolve the initial quantum state for time `t` is identical to the expression given in the previous section.

At this point, it seems like determining the dynamics of a quantum state is pretty easy — what’s the problem? Well, the problem arises when the Hamiltonian in question is not time-independent. Namely, if the Hamiltonian changes with respect to the time `t` the Schrödinger equation becomes extremely difficult (oftentimes impossible) to solve. So, we must find an approximation to its solution in order to evolve a quantum state with respect to a time-dependent Hamiltonian.

#### Approximate Evolution through Discretization in Time

So, how do we find such an approximate solution? Well, we know that the evolution of our quantum state for time `t` can be described by a unitary matrix. Let us denote this unitary matrix as shown below, where evolution occurs from time `t0` to time `t`.

[

![](https://substackcdn.com/image/fetch/$s_!rszl!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5cf67bac-4807-4280-ad0e-59e152989a05_109x37.png)



](https://substackcdn.com/image/fetch/$s_!rszl!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5cf67bac-4807-4280-ad0e-59e152989a05_109x37.png)

Often, the first step in approximating something that is continous in time (such as the unitary matrix shown above) is dividing it into discrete time steps that can be combined together. This is shown below (note that the product of two unitary matrices is always unitary).

[

![](https://substackcdn.com/image/fetch/$s_!OlkW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F43cf689c-1092-4cbf-ae7a-a010ad83783a_633x37.png)



](https://substackcdn.com/image/fetch/$s_!OlkW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F43cf689c-1092-4cbf-ae7a-a010ad83783a_633x37.png)

Discretizing the unitary matrix in time

The discretization of our unitary matrix into many, smaller timesteps is analogous to approximating a continuous function with many piecewise components. Namely, as the number of discrete timesteps increases, the approximation becomes more accurate. Going further, the unitary matrices for each discrete timestep can be expressed with respect to a time-dependent Hamiltonian, as shown below.

[

![](https://substackcdn.com/image/fetch/$s_!_GBE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F40fb3843-e062-41bc-a8df-8d41eee3d55d_800x56.png)



](https://substackcdn.com/image/fetch/$s_!_GBE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F40fb3843-e062-41bc-a8df-8d41eee3d55d_800x56.png)

Discrete evolution with respect to a time-dependent Hamiltonian

As can be seen above, by taking discrete steps in time, we can approximate continuous time evolution with respect to a time-dependent hamiltonian by repeatedly doing the following: (1) getting the Hamiltonian at some fixed time, (2) assuming the Hamiltonian is fixed for a short time interval, and (3) performing evolution with respect to this fixed Hamiltonian for a short time. By performing such discrete evolution over very short time periods, we create a cascade of unitary matrices that, when combined together, approximate the desired time evolution with respect to a time-dependent Hamiltonian.

**Trotterization (Suzuki-Trotter Decomposition)**

Discretization in time may oftentimes not be the only required component for approximating evolution with respect to a time-dependent Hamiltonian. This is because Hamiltonians are often expressed as sums of multiple, non-commuting Hamiltonians, yielding an expression resembling the one below.

[

![](https://substackcdn.com/image/fetch/$s_!AssU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F97b752c0-f907-4dc7-8b5a-b93784076d70_294x35.png)



](https://substackcdn.com/image/fetch/$s_!AssU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F97b752c0-f907-4dc7-8b5a-b93784076d70_294x35.png)

Unitary with a sum of non-commuting Hamiltonian components

In the expression above, the matrix exponential cannot be decomposed as a product of several unitary matrices because each of the Hamiltonian’s components do not commute (i.e., recall the final property from the previous explanation of matrix exponentials). As a result, we encounter the following problem.

[

![](https://substackcdn.com/image/fetch/$s_!v17f!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F27275609-5abf-4ac2-b23b-55d59aae1c41_704x43.png)



](https://substackcdn.com/image/fetch/$s_!v17f!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F27275609-5abf-4ac2-b23b-55d59aae1c41_704x43.png)

Because implementing a quantum gate based on a sum of multiple, non-commuting Hamiltonian components is difficult, practitioners often rely upon the [Suzuki-Trotter decomposition](https://en.wikipedia.org/wiki/Time-evolving_block_decimation) (i.e., commonly referred to as “Trotterization”) to decompose such an expression as a product of multiple unitary matrices. This decomposition has the following form.

[

![](https://substackcdn.com/image/fetch/$s_!AaSD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1a555fd-2d75-4508-9232-77ab394dbb3b_406x71.png)



](https://substackcdn.com/image/fetch/$s_!AaSD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1a555fd-2d75-4508-9232-77ab394dbb3b_406x71.png)

Suzuki-Trotter Decomposition

Using the Suzuki-Trotter decomposition, the unitary introduced at the beginning of this section can be approximated by a repeating cascade of separate matrix exponentials for each of the Hamiltonian’s non-commuting components. Such a decomposition greatly simplifies the implementation of this unitary matrix as a quantum gate, which is extremely relevant for NISQ-friendly quantum algorithms such as QAOA. As a result, Trotterization arises very frequently within quantum computing.

### Quantum Adiabatic Algorithm

The final step in gaining and in-depth understanding of QAOA is understanding the algorithm from which it was derived: the quantum adiabatic algorithm (QAA). [QAA](https://arxiv.org/abs/quant-ph/0001106) is an algorithm that was originally proposed for finding exact solutions to combinatorial optimization problems with the use of adiabatic evolution. At a high-level, QAA is similar to QAOA, as it begins in some initial state and evolves this state to produce the low-energy state of a cost hamiltonian. However, QAA is not an algorithm that can be implemented on NISQ devices, as it requires evolution be performed for a (potentially) very long time, exceeding the capabilities of current hardware.

#### Adiabatic Theorem

QAA relies upon the [adiabatic theorem](https://en.wikipedia.org/wiki/Adiabatic_theorem) of quantum mechanics. Given a time-dependent Hamiltonian and a low-energy eigenstate of that Hamiltonian at time `t0`, the adiabatic theorem states that, if the quantum state is evolved slowly enough for time `t`, it will remain as the low-energy state of the Hamiltonian through time (i.e., the resulting state after evolution is the low energy state of our Hamiltonian at time `t0 + t`). This theorem relies on the assumption that a gap exists between the two lowest eigenvalues of the Hamiltonian at any given time. Additionally, the amount of time required for evolution is dependent upon this gap, and may become arbitrarily large if the gap is small. Furthermore, the value of this gap cannot be estimated in general, making it difficult to determine to total amount of time required to perform such an evolution.

#### Solving Optimization Problems with the Adiabatic Theorem

Based on the adiabatic theorem, QAA makes the following proposal. Assume that we have some combinatorial optimization problem instance that we want to solve. Remember from the previous description of QAOA that we can encode this problem instance within a cost Hamiltonian, such that the solution to the problem is given by the Hamiltonian’s low-energy state. Additionally, recall that the previously-defined mixer Hamiltonian has a low-energy state that is easy to construct (i.e., the initial state within QAOA). Then, assume we define the following time-dependent Hamiltonian, based on the cost and mixer Hamiltonians defined within the description of QAOA.

[

![](https://substackcdn.com/image/fetch/$s_!ZDFw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F84293221-ccb2-484c-98d5-34119a641e30_460x87.png)



](https://substackcdn.com/image/fetch/$s_!ZDFw!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F84293221-ccb2-484c-98d5-34119a641e30_460x87.png)

Interpolation between cost and mixer Hamiltonians

Given the above time-dependent Hamiltonian, we can easily construct the low-energy state at time 0 (i.e., it is just the low-energy state of the mixer Hamiltonian). Additionally, the time-dependent Hamiltonian at time T is equal to the cost Hamiltonian. Therefore, if we evolve this intial state for long enough, we will eventually produce the low-energy state of the cost Hamiltonian according to the adiabatic theorem (i.e., we are guaranteed a non-zero eigengap by the [Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)). As a result, such an evolution produces an exact solution to our problem of interest!

### Back to QAOA

Now that we have covered relevant background information, it is time to return to QAOA and gain a more comprehensive understanding of its motivation. In particular, QAOA can be interpreted as an approximation of QAA, which can be derived by leveraging the tools we have discussed so far.

#### From QAA to QAOA

Both QAOA and QAA begin with the same initial state (i.e., the equal superposition state) and attempt to evolve this initial state to produce the low-energy state of the cost Hamiltonian. Consider the following time-dependent Hamiltonian, which is an interpolation between the mixer and cost Hamiltonians.

[

![](https://substackcdn.com/image/fetch/$s_!h0V9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F92d0446f-9d25-47fa-8cc2-22a45b4fc9a4_593x37.png)



](https://substackcdn.com/image/fetch/$s_!h0V9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F92d0446f-9d25-47fa-8cc2-22a45b4fc9a4_593x37.png)

As previously outlined, expressing evolution with respect to a time-dependent Hamiltonian in closed form is very difficult. So, we must approximate this evolution using discrete time steps, as shown below. We denote the unitary that exactly expresses the evolution of our quantum state with respect to the time-dependent Hamiltonian for time `t` as `U` .

[

![](https://substackcdn.com/image/fetch/$s_!VdfN!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3639472d-0313-4905-ace2-05556e059c7b_776x41.png)



](https://substackcdn.com/image/fetch/$s_!VdfN!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3639472d-0313-4905-ace2-05556e059c7b_776x41.png)

where `Delta` represents the length of each discrete timestep between time 0 and `t`. Furthermore, each time-dependent Hamiltonian term within the expression above can be expressed with respect to the cost and mixer Hamiltonians (from this point onward, only the first term in the sequence is listed to ease readability — other terms follow the same pattern).

[

![](https://substackcdn.com/image/fetch/$s_!banp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F51f11584-2675-471d-9501-fca334408d8f_500x39.png)



](https://substackcdn.com/image/fetch/$s_!banp!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F51f11584-2675-471d-9501-fca334408d8f_500x39.png)

From here, we can take the Suzuki-Trotter decomposition to arrive at the following expression, where we take `p` to be some large, positive integer. Recall, that the cost and mixer Hamiltonians do not commute. Thus, we are required to perform Trotterization to decompose the matrix exponential in this way.

[

![](https://substackcdn.com/image/fetch/$s_!mi8b!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff066d9fa-f944-4d22-9eaa-f83c2e105d1f_689x89.png)



](https://substackcdn.com/image/fetch/$s_!mi8b!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff066d9fa-f944-4d22-9eaa-f83c2e105d1f_689x89.png)

This is starting to look somewhat familiar… By denoting the constants in front of each Hamiltonian (exluding the imaginary components) as `beta` and `gamma` , we arrive at the following expression.

[

![](https://substackcdn.com/image/fetch/$s_!TIs7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb87f0bb0-f1e5-4066-ac0b-63c8674ae33a_394x48.png)



](https://substackcdn.com/image/fetch/$s_!TIs7!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb87f0bb0-f1e5-4066-ac0b-63c8674ae33a_394x48.png)

So, after decomposing our evolution into discrete time steps and applying Trotterization, we arrive at an expression that (almost) exactly resembles our ansatz from QAOA! Therefore, QAOA can be motivated as an approximation of QAA in which the`beta` and `gamma` parameters are not fixed and the depth `p` can be set arbitrarily! Intuitively, the quality of this approximation must improve as the depth of QAOA is increased, as the Suzuki-Trotter decomposition is only true in the limit of infinite `p`. Interestingly, several theoretical results for QAOA are derived by assuming an ansatz of infinite depth and drawing connections to QAA and the adiabatic theorem.

#### Important Distinctions

Although QAOA can be interpreted as an approximation of QAA, there are several important distinctions to be made between these algorithms. First, QAOA is a variational quantum algorithm, containing parametric quantum circuits that are classically optimized to satisfy some objective. In contrast, the corresponding “parameters” within QAA are fixed based on the time intervals used within the approximation of the unitary matrix that encodes the time-dependent evolution of the quantum state. As a result, QAOA does not necessarily approximate QAA, as its parameters can be set arbitrarily and may not reflect the discrete time interval constants that arise within QAA. Additionally, the depth of QAA may be arbitrarily large, while the depth of the QAOA ansatz is usually much smaller in order to accommodate quantum hardware limitations (i.e., the [deepest QAOA ansatz](https://arxiv.org/abs/2004.04197) that has been physically implemented has a depth of only three) — remember, QAOA is a practical algorithm for NISQ-era quantum computers. As a result, the connection between practical QAOA algorithms and QAA are, in reality, quite limited. Finally, QAOA finds an approximate solution to the problem of interest, while QAA is finds the globally optimal solution given enough time, making the goal of the two algorithms are somewhat different.

### Conclusion

In this post, I tried to provide a relatively comprehensive understanding of QAOA from the ground up. By providing relevant background, I drew a connection between QAA and QAOA, revealing that QAOA can be interpretted as an approximation of QAOA (despite the practical differences that exist between the two algorithms). This sheds light on the motivation for various aspects of QAOA, such as the choice of mixer Hamiltonian (i.e., derived from QAA) and the alternating unitary structure (i.e., can be interpreted as a discrete, Trotterized approximation of QAA).

Thank you for reading, and I hope you enjoyed the post. If you find any errors or have suggestions, feel free to [contact me](https://wolfecameron.github.io/). Furthermore, if you are interested in this topic, I would encourage you to check out my [research lab](http://akyrillidis.github.io/group/) at Rice University, which performs research related to numerous topics in quantum computing.


------

# A Broad and Practical Exposition of Online Learning Techniques

### An overview of online learning techniques, focusing on those that are most effective for the practitioner.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Sep 18, 2021

[

![](https://substackcdn.com/image/fetch/$s_!W1jO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F8fcc1c76-0812-4a74-ac85-2c34f17322b7_800x533.jpeg)



](https://substackcdn.com/image/fetch/$s_!W1jO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F8fcc1c76-0812-4a74-ac85-2c34f17322b7_800x533.jpeg)

[source](https://unsplash.com/photos/wO42Rmamef8)

In this blog post, I will take a deep dive into the topic of [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) — a very popular research area within the deep learning community. Like many research topics in deep learning, online learning has wide applications in the industrial setting. Namely, the scenario in which data becomes sequentially available to a learner is very common; see examples such as dynamic e-commerce recommendations, on-device learning scenarios, or even federated learning as examples where the full dataset might not be all available at the same time. My goal is to tackle the topic of online learning from the viewpoint of a practitioner, answering questions such as:

- What are problems faced by online learning models?
    
- What are go-to solutions for training a model online?
    
- What level of performance should I expect from a model that is trained online?
    

The goal of this post is to make practitioners aware of the options that exist within the online learning space, providing a viable solution to the scenario in which a model must learn from new data that is constantly becoming available. Not only does such a training setup eliminate latency-ridden, offline re-training procedures (i.e., the model is updated in real time), but it more closely reflects how intelligent systems learn in the world around us — when a human learns a skill, they do not require several hours of GPU training to leverage their new knowledge!

The structure of the post is as follows. I will first introduce the topic of online learning and, from a broad view, outline the problems and solutions that have been identified and proposed in the space. Although a significant number of publications on the topic of online learning exist, only some of them provide tangible benefits on the modern, large-scale architectures and datasets that would be of most interest to deep learning practitioners today. I will overview a majority of existing approaches to online learning within this post, but I will take special note of the approaches that will give practitioners the best “bang for their buck”, thus providing the needed context regarding the most useful methodologies.

### **What is online learning?**

Within this post, I define online learning as a training scenario in which the full dataset is never available to the model at the same time. Rather, the model is exposed to the portions of the dataset sequentially and expected to learn the full training task through such partial exposures. Typically, after being exposed to a certain portion of the dataset, the model is not allowed to re-visit this data later on. Otherwise, the model could simply loop over the dataset and perform a normal training procedure.

Anyone who has read papers studying online learning is probably confused what this training setup should be called, as many names have been assigned over the years of research. E.g., lifelong learning [1, 2], continual learning [3, 4], incremental learning [5, 6], and streaming learning [7, 8] — many more exist even in addition to this! Many of these names refer to very similar scenarios with slight differences in the experimental setup. For example, lifelong learning typically refers to learning multiple tasks in sequence, while incremental learning tends to learn batches of data in sequence. Somewhat differently, streaming learning takes a single pass through the dataset, never allowing the model to view multiple data examples at once. Nonetheless, I use “online learning” as a general term that refers to all of such experimental setups that share the common property of partial, sequential exposure to data.

#### _Does the setup matter?_

Given that so many different experimental scenarios exist for the study of online learning techniques, one may easily start to wonder whether studying online learning in general is useful. In other words, can we just study online learning, or does the choice of experimental setup have a significant impact on the effectiveness of certain training methodologies? Will an approach that works well in lifelong learning also work well for incremental learning? In short, _the choice of experimental setup is important_. For example, previous work has shown that models trained for incremental learning scenarios rarely perform well in the streaming setting [7]. For this reason, I will try to be specific about the exact learning scenario being discussed, especially when studying a particular paper. Luckily, however, many of the training techniques used for all types of online learning are very similar [9, 10] — they just might require slight modifications to make them more impactful in a given setting.

### Why is online learning difficult?

Before diving into current methodologies for online learning, one may wonder why any specialized training methodology is even required. _Can’t we just train the model normally as the data becomes available?_ The short answer to this question is no, but understanding why requires some background understanding.

#### A Naive Approach…

One of the simplest approaches to online learning is to maintain a single model that is fine-tuned on new data as it arrives. In this case, data becomes sequentially available to the model, and previous data cannot be re-visited. Therefore, the model is updated/fine-tuned in real-time as new data arrives and slowly learns the task of interest over time. If the incoming stream of data were [i.i.d.](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables), this fine-tuning approach would work quite well! Namely, the stream of data would be sampled evenly from the distribution of training data (e.g., an even sampling of all the different classes within a classification task), and the model would get exposed to all data in a balanced manner. Over time, if data continues to become available, the model will begin to perform quite well.

So, what’s the problem? Well, in many practical applications, the incoming data stream is non-i.i.d, meaning that it does not sample from the underlying dataset in an unbiased manner. For example, all data examples exposed to the model could be from a single class within a classification task. More practically, consider a deep learning model being used on a fashion-based e-commerce site. On a normal day, the model learns from customer activity on the site, sampling from the same group of products. However, one day a new line of products may be added to the e-commerce site, catalyzing a lot of customer activity around a single product group. In this case, the model is exposed to a massive amount of data in relation to a single topic/product, leading to an imbalanced exposure between the new and existing products on the site. Obviously, this will complicate the online learning process, but how?

#### Catastrophic Forgetting

In the online learning community, the main issue that has been identified for training models in an online fashion over non-i.i.d. data streams is catastrophic forgetting [11, 12]. Catastrophic forgetting is a property of online learning models in which the model forgets how to classify previous data as it is exposed to new data. For example, consider a dataset with 10 classes (e.g., CIFAR10), and assume that the online learning model has already been trained on classes one and two. Then, assume that the model receives new data that is only sampled from classes three and four. If the model is fine-tuned on this new data without access to any of the previously-learned data, it will begin to perform well on classes three and four but most likely deteriorate in performance on classes one and two. In other words, it will suffer from catastrophic forgetting!

The goal of online learning is to figure out how to eliminate catastrophic forgetting. As such, nearly all research in this area (1) assumes that the incoming data is non-i.i.d. (i.e., i.i.d. data streams can be trivially handled with vanilla fine-tuning) and (2) proposes some methodology that defends against catastrophic forgetting. Generally, the most common method of generating non-i.i.d. data streams (for classification-based learning problems) is to divide all possible output classes into multiple, disjoint groups and expose these groups of classes to the online learner sequentially [5, 12]. In this case, the online learning model must learn how to handle new classes while maintaining knowledge of perviously-encountered classes. For tasks other than classification (e.g., object detection or reinforcement learning), many other training setups have been derived for generating non-i.i.d. data streams that induce catastrophic forgetting [8, 13].

### What Approaches Exist?

Many approaches have been proposed for reducing catastrophic forgetting in the online learning domain. For the purposes of this post, I will broadly partition these methods into the following categories: architectural modification, regularization, distillation, replay, rebalancing, and other. For each of these categories, I will provide a brief overview of the approach, a summary of relevant papers, and a discussion practical effectiveness. I will typically consider both incremental (i.e., task/class-incremental learning) and streaming setups within these descriptions, as they are the most common within the literature.

#### Architectural Modification

_**Overview.**_ The idea behind architectural modification is simple: as you receive new data, add more parameters to your model to increase its capacity. Such parameters can be added in a structured (e.g., adding entirely new neurons or filters to the architecture) or unstructured (e.g., adding new connections between existing neurons) manner. Moreover, updating the augmented model (i.e., here I use “augmented” to refer to the model with added parameters) after new data is received can be done in two ways: (1) simply update the model without restrictions or (2) use masking/selective plasticity strategies to ensure only non-important neurons (i.e., those that don’t impact performance on previous data) are updated. In both cases, the goal of such an approach is to allow the model to perform well on both new and old data by ensuring it is never kept from expanding its knowledge of the underlying learning problem due to restricted capacity. By always adding more parameters, we ensure the model can continue to learn from new data.

_**Methods.**_ Many different approaches have been proposed for architectural modification that enumerate all possible options listed above. [13] studies the task-incremental setting and instantiates an entirely new neural network for each task (i.e., each task has its own hidden/output layers of equal size that are connected to input). Then, to ensure previous knowledge is maintained and utilized, lateral connections are formed to the hidden layers of previous tasks, allowing new tasks to draw upon previously-learned feature representations to augment the learning process. Similarly, [14] studies the task-incremental setting and proposes extra neurons to be added to the network each time a new task is introduced, while [15] studies a similar (but structured) approach of adding entire “blocks” of neurons into the network each time new data is encountered in the task-incremental setting. Finally, [16] adds new parameters into the network during the online learning process, but explores various masking and plasticity strategies to control the model’s parameter updates to avoid the destruction of knowledge from previous tasks.

_**Discussion.**_ Although architectural modification techniques have seen success in small-scale online learning problems, they have two major properties that limit their potential. First, because the architecture is being constantly expanded or enlarged, the memory requirements of these techniques is generally large/unbounded, which may become intractable in the case of large-scale models or datasets. Ideally, it would be better if the memory-usage of an online learning algorithm was not dependent upon the amount of data being received. Additionally, the majority of such methodologies are dependent upon the existence of “task boundaries” (i.e., pre-determined break points in the incoming data stream, such as the batches present in incremental learning). Such task boundaries provide obvious points during the online training process during which parameters/modules can be added to the network. Some architectural modification methodologies are completely dependent on the existence of such task boundaries [13, 14, 15], but these boundaries are not always present (e.g., during streaming learning). As such, reliance upon tasks boundaries limits the applicability of such techniques in some scenarios.

#### Regularization

_**Overview.**_ Regularization techniques for online learning typically try to (i) identify parameters that are “important” and (ii) induce regularization terms during training that prevent such parameters from being changed too much. Typically, important parameters are defined as those that while deteriorate network performance when updated/perturbed. Numerous different heuristics for importance have been proposed, but they all share the common goal of characterizing whether modifying that parameter will harm the network’s performance with respect to old data that is no longer present. By ensuring that important parameters are not modified during online training, the performance of the network on old data is preserved, as parameter updates for new data are consolidated to regions that are not relevant to the network’s behavior.

_**Methods.**_ In [17], parameter importance is defined as the contribution of a parameter to network accuracy on previous tasks, which is estimated using the approximate posterior probability of a parameter (computed using the Fisher information matrix). Then, dynamic regularization is imposed as the network is trained on new data such that parameters with high importance incur large penalties if they are updated from their original value, thus encouraging training to only update parameters that are not important to maintaining performance on previous data. [18, 19] follow a nearly identical approaches, but use different heuristics for parameter importance. Namely, [18] defines parameter importance using the gradient magnitudes of each parameter throughout training, while [19] considers the change in loss related to each parameter as a proxy for parameter importance when learning a new task. Although not exactly regularization approaches, [20, 21] propose constrained optimization formulations of online learning that ensure parameter updates during online training cannot damage performance on previous tasks.

_**Discussion.**_ Similar to architectural modification approaches, regularization-based online learning methodologies show promise at smaller scales. However, when used in large-scale experiments, such approaches tend to not be super effective and computing parameter importances becomes extremely expensive for large, deep learning models. For this reason, regularization-based approaches to online learning are typically not considered to be useful for large-scale online learning applications [9, 10].

#### Distillation

_**Overview.**_ Distillation methods for online learning are inspired by the concept of knowledge distillation within deep learning [22]. Originally, knowledge distillation was proposed to “distill” the knowledge of a large “teacher” network into the parameters of a smaller “student” network, by training the student to match the output of the teacher over a dataset. Somewhat differently, online learning methodologies adopt distillation so that the knowledge of previous models (i.e., those trained on older data) can be distilled into the current network being learned to ensure historical knowledge is not lost. The methodology is quite similar to that of normal knowledge distillation. The main difference is that the teacher and student networks are typically of the same size/architecture, but taken from different points in the online training phase. Such a process is similar to the technique of self distillation [39, 40].

_**Methods.**_ Numerous distillation methodologies have been proposed in the online learning space. In fact, because it tends to not interfere with other methodologies (i.e., it is just a modification to the loss function!), distillation is often combined with other methodologies to augment the online learning process [5, 24]. [23] was one of the first works to propose using distillation for online learning in the batch incremental setting. This work showed that if a distillation loss with respect to previous network settings was added to the loss computed over new data, one can better maintain the network’s performance on previous tasks. Such a methodology aimed to replace the previous go-to methodology for avoiding catastrophic forgetting in batch incremental learning— adding a fine-tuning stage that includes both old and new data in a balanced ratio (i.e., balanced fine-tuning) [23, 25]. Subsequent works discovered that performing distillation with new data may cause performance deterioration if distributions of new data are significantly different from old data. Such an issue was then solved by directly caching old data examples to be used in computing the distillation loss [2, 5, 24, 26].

_**Discussion.**_ Distillation is a commonly-used approach within the online learning community that has worked well even at larger-scales [5]. However, followup work, when considering the use of distillation for online learning, showed that distillation is less effective when previous data is cached for use during fine-tuning [27]. In fact, some work even argued that adding distillation to the loss is unnecessary — possibly even harmful — when an explicit memory of previous data is maintained for use in online updates [9, 10, 28]. As such, distillation methodologies, although they still remain popular, are questionably effective when memory of previous data examples is allowed. As such, methods that store previous data examples for use during online updates — collectively referred to as replay (or rehearsal) techniques — have become a go-to approach.

#### Replay

_**Overview.**_ The term “replay” broadly refers to online learning methodologies that store exemplars from previous portions of the dataset. Then, when new data arrives, these stored exemplars can be incorporated into the online learning process to prevent catastrophic forgetting. For example, these previous data examples could be added into a distillation loss to avoid network output from deviating too much from previous settings. More commonly, previous data exemplars could be simply sampled (i.e., as in a mini-batch) for combination with new data during the online learning process. In the batch incremental setting, previous examples would be mixed with the batch of new data during fine-tuning to ensure old knowledge is not lost. Similarly, streaming approaches would incorporate randomly-sampled exemplars from previous classes into online updates, thus ensuring knowledge is maintained.

_**Methods.**_ Numerous approaches have been proposed for replay-based online learning. Firstly, many distillation approaches store previous data examples for use during distillation, which loosely falls into the category of replay [2, 5, 6, 24]. Pure replay, however, typically performs model updates directly with old data. [3] discovered that retaining even a very small number of previous data examples (i.e., a few examples per class) to be used directly in online training drastically reduced the impact of catastrophic forgetting. Similarly, [29] demonstrated that full replay (i.e., maintaining all previous data in a buffer) completely eliminates catastrophic forgetting, while partial replay (even with a significantly reduced number of examplars) provides significant benefits for online learning — this result is similar to the findings of [3]. More recently, [7] extensively explored replay techniques in the streaming setting, allowing compressed feature representations to be stored in the replay buffer instead of the data itself. Such an approach, which was modified for object detection in [8], was highly successful in achieving good results for streaming learning on large-scale image classification tasks.

_**Discussion.**_ Replay mechanisms are a now a core component of most online learning methodologies due to the scale-agnostic success of these methodologies in various applications. Although storing previous data examples can be memory intensive, performing replay provides drastic benefits to online learning performance. In fact, replay has been shown to completely eliminate catastrophic forgetting if sufficient data exemplars are maintained in the buffer [29]. Due to its simplicity and practical effectiveness, replay has become extremely popular in the online learning community.

#### Rebalancing

_**Overview.**_ Several recent works in batch-incremental learning have noticed that models learned in an online fashion tend to be biased towards the most recently-observed data (i.e., those in the most recent batch). As such, several techniques, which I refer to as rebalancing techniques, were proposed to eliminate such imbalance. The core idea behind such approaches is to ensure predictions are not biased towards newer data (e.g., in the classification setting, a biased model would predict nearly all data as one of the classes within the most recently-observed batch of training data). Rather, the magnitude of predictions should be balanced between all classes or types of data, agnostic of when such data was encountered during the training processs.

_**Methods.**_ Techniques for eliminating bias in network classification layers for the batch-incremental setting were originally explored in [6] and [37]. In [6], cosine normalization is employed on the softmax layer of the model, thus ensuring output vectors for each class have an equal magnitude. [37] proposes a similar, learnable approach that leverages a small validation dataset to train a linear “correction” module to eliminate bias in the classification layer. Beyond these initial works, several other methodologies were proposed for fixing bias in the classification layer for incremental learning — typically inspired by the idea of making classification weights for a class similar magnitude to when the class was first learned. [38] stores classification statistics in a small memory buffer when a class is first encountered during online learning, then uses these statistics to make class scores more comparable later in training. [27] follows a similar approach, but directly re-uses the classification weights from when a class was first learned during online training.

_**Discussion.**_ Prediction bias is a known, measurable issue within incremental learning (i.e., both task and batch/class incremental). Furthermore, adding rebalancing is shown to drastically improve incremental learning performance, even on large-scale datasets such as ImageNet [6, 37]. As such, rebalancing methods are worth employing in this domain. In general, it is probably best to utilize methods that do not require any validation set to perform rebalancing (i.e., this simply avoids the trouble of creating a validation set). For example, [6] does not require any validation set, while [37] does require a validation set. Beyond the incremental learning setting (e.g., in streaming learning), it is not clear whether classification bias follows the same patterns as in incremental learning. However, adding rebalancing is unlikely to damage the performance of the online learning model.

#### Other Techniques

Somewhat similarly to replay, several works have proposed the use of generative models to “hallucinate” examples from previous classes. In [16], a generative adversarial network (GAN) [30] is employed for batch-incremental online learning, in which the generator creates data examples to use for replay and the discriminator both discriminates and solves the underlying learning problem. [31] adopts a similar approach with an auto encoder, but incroperates an extra distillation term into the loss function. [32] leverages GANs in the task incremental setting, where each task trains both a disciminator and generator. Then, the generator can be communicated to the next task so that old data representations can be constructed through the generator for replay during learning of the new task. A similar approach is adopted by [33] to solve the domain expansion problem (i.e., task incremental learning with only two tasks).

Another somewhat popular area of study is dual memory techniques for online learning. Such methodologies are inspired by the brain and try to mimic the biological process of memory consolidation. At a high-level, the dual-memory approach — which is typically coupled with replay [12] — dedicates separate model components to newly-formed and long-term knowledge. For example, two separate networks could be maintained, where one network is solely devoted to learning new data and the other tries to solve the overall learning problem (i.e., both old and new data) [12]. [35] proposes a similar approach that maintains two separate models: a probabilistic model for short-term memory and an autoencoder for long-term memory. Somewhat differently, highly-uncertain examples could be stored within a separate memory buffer that is later incorperated into the network [34].

Some other less studied (but still notable) approaches to online learning include sparse-coding methodologies [12], ensemble-based methods [36], and methodologies that modify the activation function within the neural network to avoid catastrophic forgetting [26]. Although these methodologies are less popular and have fallen out of favor in comparison to common approaches like replay and distillation, it is still useful to have such techniques in mind to gain a holistic understanding of the area and (hopefully) provide ideas for future innovation.

### So…What should I use?

Given the extensive discussion of online learning methodologies within this post, it is important to provide a summary of the existing approaches that highlights the most useful methods in large-scale deep learning applications. The utility of existing methods can be summarized simply as follows:

- Architectural modification and regularization are less-used because they suffer certain drawbacks (as mentioned in the discussion) and tend to not perform as well at scale.
    
- Distillation is very popular, but is questionably effective when replay is allowed.
    
- Replay is widely considered to be the go-to approach for mitigating catastrophic forgetting, and has been shown to work extremely well in large-scale online learning experiments.
    
- Rebalancing is important in the incremental learning setting, as it eliminates biases that form towards recently-observed data.
    

Therefore, the best “bang for your buck” in the online learning domain would be using replay-based online learning methods. For example, [7, 8] propose a replay-based approach for large-scale deep learning scenarios that performs surprisingly well and is memory efficient. Similarly, approaches such as [3, 29] demonstrate that simply maintaining a buffer of previous data for use during online updates is an extremely powerful tool. With this in mind, performing replay seems to be a good choice in almost all online learning scenarios.

In addition to replay, using distillation may improve performance in some cases, though some work argues that distillation is not useful when combined with replay. For example, [5] performs a combination of distillation and replay that seems to perform very well even at larger scales, thus showing that distillation can positively impact online learning performance in certain scenarios. Furthermore, if one is training a model using incremental learning, it is important to utilize rebalancing, as bias within the classification layer can significantly deteriorate performance.

Thank you so much for reading, and I hope you enjoyed the post. Feel free to visit my [research page](https://wolfecameron.github.io/), or reach out to me with any question/comments you may have about the post. You can also [follow me on twitter](https://twitter.com/cwolferesearch) if you want to stay up to date with my publications/blog posts. If you are interested in this topic, I encourage you to visit the [webpage for my lab](http://akyrillidis.github.io/group/) at Rice University, where I am a PhD student focusing in the empirical and theoretical foundation of deep learning.

_Bibliography  
_[1] [https://arxiv.org/abs/1704.01920](https://arxiv.org/abs/1704.01920)

[2] [http://home.ustc.edu.cn/~saihui/papers/eccv2018_lifelong.pdf](http://home.ustc.edu.cn/~saihui/papers/eccv2018_lifelong.pdf)

[3] [https://arxiv.org/abs/1902.10486](https://arxiv.org/abs/1902.10486)

[4] [https://arxiv.org/abs/1908.04742](https://arxiv.org/abs/1908.04742)

[5] [https://arxiv.org/abs/1807.09536](https://arxiv.org/abs/1807.09536)

[6] [https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)

[7] [https://arxiv.org/abs/1910.02509](https://arxiv.org/abs/1910.02509)

[8] [https://arxiv.org/abs/2008.06439](https://arxiv.org/abs/2008.06439)

[9] [https://arxiv.org/abs/2011.01844](https://arxiv.org/abs/2011.01844)

[10] [https://arxiv.org/abs/1909.0838](https://arxiv.org/abs/1909.08383)

[11] [https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)

[12] [https://arxiv.org/abs/1708.02072](https://arxiv.org/abs/1708.02072)

[13] [https://arxiv.org/abs/1606.04671](https://arxiv.org/abs/1606.04671)

[14] [https://arxiv.org/abs/1612.03770](https://arxiv.org/abs/1612.03770)

[15] [https://arxiv.org/abs/1908.08017](https://arxiv.org/abs/1908.08017)

[16] [https://arxiv.org/abs/1904.03137](https://arxiv.org/abs/1904.03137)

[17] [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)

[18] [https://arxiv.org/abs/1711.09601](https://arxiv.org/abs/1711.09601)

[19] [https://arxiv.org/abs/1703.04200](https://arxiv.org/abs/1703.04200)

[20] [https://arxiv.org/abs/1706.08840?source=post_page---------------------------](https://arxiv.org/abs/1706.08840?source=post_page---------------------------)

[21] [https://arxiv.org/abs/1812.00420](https://arxiv.org/abs/1812.00420)

[22] [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

[23] [https://arxiv.org/abs/1606.09282](https://arxiv.org/abs/1606.09282)

[24] [https://arxiv.org/abs/1606.02355](https://arxiv.org/abs/1606.02355)

[25] [https://arxiv.org/abs/1903.12648](https://arxiv.org/abs/1903.12648)

[26] [https://arxiv.org/abs/1607.00122](https://arxiv.org/abs/1607.00122)

[27] [https://arxiv.org/abs/2001.05755](https://arxiv.org/abs/2001.05755)

[28] [https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)

[29] [https://arxiv.org/abs/1809.05922](https://arxiv.org/abs/1809.05922)

[30] [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

[31] [https://arxiv.org/abs/1704.01920](https://arxiv.org/abs/1704.01920)

[32] [https://arxiv.org/abs/1705.08690](https://arxiv.org/abs/1705.08690)

[33] [https://arxiv.org/abs/1705.00744](https://arxiv.org/abs/1705.00744)

[34] [https://hal.archives-ouvertes.fr/hal-01418123/document](https://hal.archives-ouvertes.fr/hal-01418123/document)

[35] [https://arxiv.org/abs/1711.10563](https://arxiv.org/abs/1711.10563)

[36] [https://arxiv.org/abs/1611.06194](https://arxiv.org/abs/1611.06194)

[37] [https://arxiv.org/abs/1905.13260](https://arxiv.org/abs/1905.13260)

[38] [https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)

[39] [https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)

[40] [https://arxiv.org/abs/1805.04770](https://arxiv.org/abs/1805.04770)