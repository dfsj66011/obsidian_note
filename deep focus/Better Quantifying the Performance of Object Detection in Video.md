
#### Hungarian Algorithm

Within any form of object detection, we are given during evaluation a set of ground truth and predicted objects. But, it is not immediately clear which of these predictions corresponds to which of the ground truths. Therefore, some work must be done to determine the best mapping between ground truth and predicted objects before the quality of predictions can be evaluated. More specifically, a one-to-one mapping between predicted and ground truth objects must be produced. However, it is also possible for predicted or ground truth objects to have no match (i.e., in such a case the mapping is not technically one-to-one, as there exists unpaired elements).

The [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is a combinatorial optimization algorithm that can be used to solve the [set matching](https://en.wikipedia.org/wiki/Assignment_problem) problem in polynomial time. Within the object detection domain, it is commonly used to produce the mapping between predicted and ground truth objects, typically based on pairwise [intersection over union](https://en.wikipedia.org/wiki/Jaccard_index) (IoU) scores between objects. Additionally, some extra requirements are typically imposed for objects to be matched with each other within the Hungarian algorithm (e.g., IoU of two bounding box object detections must be greater than 0.5 to be a considered a viable match). The Hungarian algorithm is desirable for such an application due to its performance and efficiency.

#### Mean Average Precision

As previously stated, mean average precision (mAP) is the standard metric for evaluating object detectors in static images. In order to compute mAP, one must first compute the number of true positive/negatives and false positive/negatives within an image. After running the Hungarian algorithm, computing such metrics is quite simple (i.e., we just check if detections are missing a pair, pairs are wrong, etc.). From here, precision and recall can be computed at different “confidence” levels within an image to determine the average precision (AP). Such a process is repeated separately for every different semantic class in a given problem domain. Then, the mean of the average precisions is computed across all semantic classes, forming the mean average precision (mAP) metric.

For a more in-depth description of mAP, I recommend reading [this article](https://blog.roboflow.com/mean-average-precision/). However, understanding that (1) mAP is the go-to metric for evaluating object detection performance in static images and (2) mAP is computed by looking at prediction metrics within a single image should be sufficient for understanding the remainder of this post.

### What metrics do we have already?

Before I introduce the metric that is (in my opinion) most appropriate for evaluating object detectors in video, it is worthwhile to consider some other metrics that could be used. I will divide this discussion into two categories based on whether metrics were used for object detection or object tracking and provide a brief, high-level description of each metric, aiming to demonstrate its shortfalls in the context of evaluating video object detection. First, recall that measuring the performance of object detection in video has four major components: detection, localization, association, and classification. As will be seen in this section, object tracking does not consider the classification component of evaluation, _but classification-aware metrics for object tracking have the potential to capture all relevant components of video object detection performance_.

#### Object Detection

As previously mentioned, the most-common metric for evaluating object detection in video (at the time of writing) is **mAP**. In particular, mAP is computed separately across each frame of a video and an average of mAP scores is taken across all video frames to form a final performance metric. Such an approach completely fails to capture the temporal aspect of video object detection (i.e., no notion of association) and is, therefore, insufficient as an evaluation protocol. Nonetheless, mAP is currently the go-to metric for object detection in video and discussions of changing/modifying this metric are seemingly minimal.

One interesting variant of mAP that incorporates temporal information is **speed-based mAP** [3] (i.e., I made this name up for the context of this blog post; the associated reference does not provide a specific name). To compute this metric, objects are first divided into three different groups (i.e., slow, medium, and fast) based on how fast they are moving between frames (i.e., computed using motion IOU of adjacent frames). Then, mAP is computed separately for objects in each of these three groups, and the three mAP scores are presented separately. Although speed-based mAP provides a more granular view of object detection performance based on objects’ speed in the video, it provides three metrics instead of one and still does not capture the temporal aspect of video object detection; again, mAP contains no notion of association. Therefore, speed-based mAP is not a suitable metric for video object detection.

One final metric that has been proposed for video object detection is **Average Delay (AD)** [4]. AD captures the delay between an object entering a video and being “picked up” by the detector, which is referred to as “algorithmic latency”. This delay is measured in frames, such that an AD of 2 means that, on average, an object will exist in the video for two full frames until it is actually detected by the model. Although AD captures temporal information within video object detection, _it is proposed as a metric to be used in union with mAP_. Therefore, it is not a standalone metric that can be used to comprehensively evaluated the performance of object detectors in video.

#### Object Tracking

Although no comprehensive metrics have yet been proposed for video object detection, many useful metrics exist within the object tracking community from which inspiration can be drawn. Object tracking requires that objects (either one or multiple objects within each video frame) be identified, localized, and associated in a consistent manner throughout a video. Although object tracking contains no notion of classification, sufficient overlap exists between object tracking and video object detection to warrant a more in-depth examination of current evaluation metrics in object tracking.

**Multiple Object Tracking Accuracy (MOTA)** [5] is one of the most widely-used metrics in object tracking. MOTA matches ground truth to predicted objects per-detection, meaning that each predicted and ground truth detection is treated as a separate entity during evaluation. At a high level, MOTA (based on a matching provided by the Hungarian algorithm) determines the number of identity switches (i.e., the same object is assigned a different identifier in adjacent video frames), false positives, and false negative detections across all video frames. Then, MOTA is computed by normalizing the aggregate sum of these components by the total number of ground truth objects in the video, as outlined in the equation below.

[

![](https://substackcdn.com/image/fetch/$s_!aYny!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F64c098d5-bd79-45a5-8a0a-61be7d1a4b1c_800x121.png)



](https://substackcdn.com/image/fetch/$s_!aYny!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F64c098d5-bd79-45a5-8a0a-61be7d1a4b1c_800x121.png)

The MOTA Tracking Metric

MOTA captures association performance through identity switches (i.e., denoted as `IDSW` above), while detection performance is captured through false positives and negatives. However, MOTA does not consider localization. Rather, localization must be measured by a separate metric, **multiple object tracking precision (MOTP)** [5], which averages localization scores across all detections within a video. Despite being a long-time, standardized metric for object tracking, MOTA has several shortcomings that limit its applicability to video object detection. Namely, it overemphasizes detection performance (i.e., the impact of identity switches on the above score is minimal), does not consider association beyond adjacent frames, does not consider localization, provides multiple scores instead of a unified metric, is highly-dependent on frame rate, and provides an unbounded score (i.e., MOTA can have a value of [-∞, 1]) that may be difficult to interpret. As such, _MOTA is not sufficient as a metric for video object detection_.

Other widely-used metrics within the object tracking community are **IDF1** [6] and **track-mAP** (also known as 3D-IoU), which match ground truth to predictions on a trajectory level (i.e., trajectories are defined as sequences of predicted or ground truth objects throughout video frames that share the same, unique identifier). In comparison to MOTA, IDF1 and track-mAP are not as widely-used within the object tracking community, so I will not provide an in-depth discussion of these metrics (see [7] for a more comprehensive discussion and comparison between metrics). However, _IDF1 and track-mAP both have numerous limitations, which impede them from being adopted as standard metrics for video object detection_. Namely, IDF1 overemphasizes association performance, does not consider localization, and ignores all association/detection outside of trajectories that are not matched with each other. Similarly, track-mAP requires each trajectory prediction to contain a confidence score, requires a trajectory distance metric to be defined by the user, and can be easily “gamed” (i.e., simple counterexamples can be provided which perform poorly but achieve a high track-mAP score).

### Towards a Comprehensive Metric

Despite the limitations of metrics outlined in the previous section, there is a new standard metric within the object tracking community as of mid-2020: **higher order tracking accuracy (HOTA)** [7]. In this section, I will introduce the HOTA metric and explain why I believe it is an appropriate metric for evaluating video object detectors. Although HOTA does not consider classification performance, its classification-aware counterpart, CA-HOTA, effectively captures all relevant aspects of video object detection performance within a single evaluation metric.

#### Why HOTA?

The goal of HOTA is to (1) provide a single score that captures all relevant aspects of tracking performance, (2) enable long-term association performance to be measured (i.e., association beyond two adjacent frames), and (3) decompose into different sub-metrics that capture more detailed aspects of tracking performance. HOTA, which aims to mitigate issues with previous tracking metrics, provides a single score within the range [0, 1] that captures all relevant aspects of tracking performance (i.e., detection, association, and localization) in a balanced manner. Additionally, this single, interpretable score can be decomposed into sub-metrics that characterize different aspects of tracking performance on a more granular level. The benefits of HOTA in comparison to other widely-used tracking metrics are summarized by the following figure.

[

![](https://substackcdn.com/image/fetch/$s_!6XqP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9d5a7cc0-2a95-4237-8f1d-4ecbd3abf460_800x382.png)



](https://substackcdn.com/image/fetch/$s_!6XqP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9d5a7cc0-2a95-4237-8f1d-4ecbd3abf460_800x382.png)

HOTA in Comparison to Other Tracking Metrics [7]

Although HOTA does not consider classification performance, variants have been proposed that incorporate classification into the HOTA score (e.g., CA-HOTA). CA-HOTA captures all aspects of performance for video object detection (i.e., association, localization, detection, and classification) within a single, interpretable metric. **As such, CA-HOTA can be considered a (relatively) comprehensive metric for video object detection.**

#### What is HOTA?

[

![](https://substackcdn.com/image/fetch/$s_!jl3Z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa35a03e2-b8aa-4f7b-865c-8082c3529bad_800x551.png)



](https://substackcdn.com/image/fetch/$s_!jl3Z!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa35a03e2-b8aa-4f7b-865c-8082c3529bad_800x551.png)

`, An Illustration of the HOTA Metric [7]

Similar to MOTA, HOTA matches predictions and ground truth objects at a detection level (i.e., as opposed to a trajectory level like in IDF1 or track-mAP). Within HOTA, two categories of metrics are computed from the ground truth and predicted object matches (again, produced by the Hungarian algorithm): detection components and association components. Detection components are simply true positives, false negative, and false positives, which have been discussed previously. Association components, which are somewhat different, include true positive associations (TPA), false negative associations (FNA), and false positive associations (FPA).

Consider an object with a valid match in a particular frame (i.e., a true positive detection), which we denote as `c`. Within this true positive detection, both the predicted and ground truth objects must be assigned a unique identifier. To compute the number of TPAs, one simply finds the number of true positives in other frames that share the same ground truth and predicted identifier as `c` and repeats this process for every possible `c` within the video (i.e., every true positive detection). FNAs and FPAs are defined in a similar manner, but one must find detections in other frames that have the same ground truth identifier and a different predicted identifier or the same predicted identifier and a different ground truth identifier, respectively. Essentially, TPAs, FPAs, and FNAs allow association performance to be measured across all frames within a video, instead of only between adjacent frames. Hence, the name “higher order” tracking accuracy.

Given a certain localization threshold (i.e., the localization threshold modifies matches produced by the Hungarian algorithm), we can compute all detection and association components of HOTA. Then, the aggregate HOTA score at a given localization threshold (which we denote as alpha) can be computed as follows.

[

![](https://substackcdn.com/image/fetch/$s_!lOsK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F666cb92d-e554-4e0a-84cd-a2a057f21692_800x65.png)



](https://substackcdn.com/image/fetch/$s_!lOsK!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F666cb92d-e554-4e0a-84cd-a2a057f21692_800x65.png)

HOTA Metric at a Single Localization Threshold

Once HOTA has been computed at a specific localization threshold, the aggregate HOTA score can be derived by averaging the above metric across several different localization thresholds. Typically, a sample of localization thresholds are taken in the range [0, 1], as shown in the equation below.

[

![](https://substackcdn.com/image/fetch/$s_!zqNH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F21a2c714-6fe1-4a08-b052-651dd4a169ec_800x150.png)



](https://substackcdn.com/image/fetch/$s_!zqNH!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F21a2c714-6fe1-4a08-b052-651dd4a169ec_800x150.png)

HOTA Metric

As revealed by the equations above, HOTA captures detection (through detection components), association (through association components), and localization (through localization thresholds) performance. Furthermore, HOTA can be decomposed into numerous sub-metrics. Each of these sub-metrics, outlined in the figure below, can be used to analyze a more specific aspect of tracker performance (e.g., localization, detection, or association performance in isolation).

[

![](https://substackcdn.com/image/fetch/$s_!c5u4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1434bbb7-0ac3-4176-b621-9fadd1553538_800x400.png)



](https://substackcdn.com/image/fetch/$s_!c5u4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1434bbb7-0ac3-4176-b621-9fadd1553538_800x400.png)

Sub-Metrics within HOTA [created by author]

For a more comprehensive discussion of how the HOTA metric is computed, one can also read the associated paper [7] or blog post [8].

#### What about classification?

I previously mentioned that HOTA can be modified to also capture classification performance, but never clarified exactly how this is done. Although multiple variants for classification-aware HOTA are proposed [7], one possible method is to modify the HOTA score at a certain localization threshold as follows.

[

![](https://substackcdn.com/image/fetch/$s_!UOdP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3b306c3b-99eb-497d-9462-5463f7d7fd68_800x226.png)



](https://substackcdn.com/image/fetch/$s_!UOdP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3b306c3b-99eb-497d-9462-5463f7d7fd68_800x226.png)

Classification Aware HOTA (CA-HOTA)

As can be seen above, classification performance is incorporated by scaling all contributions to the HOTA metric by the associated confidence of the correct class for a given true positive detection. As a result, the HOTA metric will deteriorate if classification performance is poor. Given the above definition, the aggregate CA-HOTA metric can then be computed as an average of scores over different localization thresholds (i.e., just as for vanilla HOTA), yielding the CA-HOTA metric.

### Conclusion

Currently, the video object detection community lacks a unified evaluation metric that correctly captures model performance. Commonly, mAP metrics are used and averaged across video frames to characterize performance, but such a technique completely disregards the temporal aspect of model performance. Drawing upon recent progress in the object tracking community, I claim that the higher order tracking accuracy (HOTA) metric is a suitable evaluation criterion for object detection in video. The classification aware variant of HOTA, CA-HOTA, captures all relevant aspects of video object detection performance, including detection, association, localization, and classification. As such, _it is a comprehensive metric (especially in comparison to static metrics like mAP) that can and should be used for benchmarking different methodologies in video object detection_.

I hope this writeup will spark discussion within the community and lead to more standardized and comprehensive benchmarking for object detection in video. This work was done as part of my job as a Research Scientist at [Alegion](https://www.alegion.com/). If you are interested in the topic, I encourage you to check out the company, or maybe even apply for an open position. We are always looking for more people interested machine learning-related topics! To keep up with my blog posts, feel free to [visit my website](https://wolfecameron.github.io/) or follow me on [twitter](https://twitter.com/cwolferesearch).

_References_

[1] [https://cocodataset.org/#home](https://cocodataset.org/#home)

[2] [https://www.mdpi.com/2076-3417/10/21/7834](https://www.mdpi.com/2076-3417/10/21/7834)

[3] [https://arxiv.org/abs/1703.10025](https://arxiv.org/abs/1703.10025)

[4] [https://www.researchgate.net/publication/335258204_A_Delay_Metric_for_Video_Object_Detection_What_Average_Precision_Fails_to_Tell](https://www.researchgate.net/publication/335258204_A_Delay_Metric_for_Video_Object_Detection_What_Average_Precision_Fails_to_Tell)

[5] [https://arxiv.org/abs/1603.00831](https://arxiv.org/abs/1603.00831)

[6] [https://arxiv.org/abs/1609.01775](https://arxiv.org/abs/1609.01775)

[7] [https://arxiv.org/abs/2009.07736](https://arxiv.org/abs/2009.07736)

[8] [https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1)


------

# How to Train Deep Neural Networks Over Data Streams

### We know how to train neural networks on regular datasets. But, what do we do if we only have access to a bit of data at a time?

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Nov 04, 2021

[

![](https://substackcdn.com/image/fetch/$s_!Woi9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa7537463-e8a7-4e5e-b8b2-2be4482c83db_800x533.jpeg)



](https://substackcdn.com/image/fetch/$s_!Woi9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa7537463-e8a7-4e5e-b8b2-2be4482c83db_800x533.jpeg)

[source](https://unsplash.com/photos/rB7-LCa_diU)

Historically, many machine learning algorithms have been developed to handle, and learn from, incoming streams of data. For example, models such as SVMs and logistic regressors have been generalized to settings in which the entire dataset is not available to the learner and training must be conducted over an incoming, sequential stream of data [1, 2]. Similarly, many clustering algorithms have been proposed for learning over data streams [3]. These methodologies force the underlying model to learn from a continuous data stream that becomes available one example at a time, eliminating the need for the entire dataset to be available at once. Interestingly, although approaches for streaming learning have been developed for more traditional machine learning algorithms, streaming learning is not widely explored for deep neural networks, where offline training (i.e., performing several loops/epochs over the full dataset) dominates.

Aiming to close this gap, recent work in the deep learning community has explored the possibility of training deep networks via streaming. Within the deep learning domain, streaming can be described as a learning setup in which _(i)_ the dataset is learned one example at a time (i.e., the dataset is presented as an incoming stream of data), _(ii)_ each unique example in the dataset is seen only once, _(iii)_ the ordering of data in the stream is arbitrary, and _(iv)_ the model being trained can be evaluated at any point within the data stream. In comparison to the typical, offline method of training neural networks, this setup may seem pretty harsh, which explains why achieving high performance in the streaming setting is often difficult. Nonetheless, streaming learning is reflective of many applications in industry and, if utilized correctly, provides a powerful tool for the deep learning practitioner.

Within this post, I will provide an overview of streaming within deep learning. I will begin by motivating the use of streaming learning, focusing upon the utility of streaming learning in practical applications. I will then overview existing approaches for training deep neural networks in the streaming setting, emphasizing the approaches that are most useful in practice. Through this post, I aim to _i)_ illustrate the relationship between streaming learning and deep learning in general and _ii)_ outline useful findings for leveraging the power of streaming learning in actual applications.

### Why streaming learning?

**How does streaming relate to other training paradigms?** Prior to the proposal of streaming learning [4], numerous training setups were studied that explore different strategies of sequentially exposing partial, disjoint subsets of data to the model instead of training over the entire dataset in an offline fashion. Within this post, I will refer to all of such methodologies (including streaming learning) collectively as “online” learning setups. Typically, online learning divides the dataset into several disjoint “batches” of data, and the model is exposed to one batch at a time. Once a batch of data is learned, the model cannot return to it later in the training process (i.e., only “current” data can be directly accessed to train the model).

Many different variants of online learning have been proposed (e.g., lifelong learning, continual learning, batch/class-incremental learning, streaming learning, etc.). Each of these different variants refers to the same concept of online learning described above, usually with a minor change to the experimental setup. For example, lifelong learning tends to learn different tasks in sequence (i.e., each task could be considered a subset/batch of data from the dataset containing all tasks), while class incremental learning learns a strict subset of classes for an overall classification problem within each batch (e.g., the first 20 classes of CIFAR100). Notably, each batch of data during the online learning process can be quite large. For example, class-incremental learning experiments are often performed on ImageNet, where each batch contains a 100-class subset of the full dataset (i.e., ~130K data examples).

Streaming learning can also be interpreted as a variant of the above description for online learning, where each “batch” of data is just a single example from the dataset. Yet, streaming learning seems to deviate from the usual online learning setup more than other, related methodologies. Namely, streaming learning, because it is restricted to learning the dataset one example at a time, cannot perform arbitrary amounts of offline training each time a new batch of data becomes available. This is because only a single data example is made available to the model at a time, and performing several updates over the same data in sequence can quickly deteriorate model performance. Thus, _streaming learning methodologies tend to perform brief, real-time model updates as new data becomes available_, whereas other online learning methodologies often perform expensive, offline training procedures to learn each new batch.

**Why is online learning difficult?** When one first encounters the topic of online learning, they may think that solving this problem is quite easy. _Why not just fine-tune the model on new data as it becomes available?_ Such a naive approach works in some cases. In particular, it will work if the incoming data stream is [i.i.d.](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables), meaning that each piece of incoming data is sampled uniformly from the full space of possible data examples. In this case, learning from the data stream is the same as uniformly sampling an example from the dataset for each training iteration (i.e., this is just stochastic gradient descent!), and naive fine-tuning works quite well. However, we cannot always ensure that incoming data is i.i.d. In fact, many notable practical applications are characterized by non-i.i.d. streams of data (e.g., video streams, personalized behavior tracking, object tracking, etc.).

Although online learning is easy to solve when the incoming data is i.i.d., an interesting phenomenon occurs when the data is made non-i.i.d. — the model learns from the new data, but quickly forgets everything that was learned previously. For example, in the case of class-incremental learning, a model may begin learning how to classify horses (i.e., some class it has not encountered before), but completely forget how to classify dogs, cats, squirrels, and all other animals that is had learned to classify in the past. This problem — commonly referred to as catastrophic forgetting [5, 6] — is the fundamental issue faced by all online learning techniques. Namely, because data streams are often non-i.i.d., models learned in an online fashion typically suffer from catastrophic forgetting, which significantly impacts their performance (especially on data that is not recently-observed).

**Streaming Learning is more practical.** Now that we have a better understanding of streaming learning, online learning in general, and the issues faced by both, one may ask the question: _why focus on streaming in particular?_ The main reasons are that streaming learning _i)_ occurs in real-time and _ii)_ better reflects common learning paradigms that arise in practice.

Because learning occurs one example at a time within streaming, model updates tend to be brief (i.e., one or a few forward/backward passes per example). As a result, minimal latency exists between the arrival of a new data example and the adaptation of the underlying model to that data example — _the model is updated in real time as new data becomes available_. In comparison, other commonly-studied online learning setups may suffer latency when _i)_ waiting for a sufficiently-large batch of new data to accumulate or _ii)_ updating the model after the new batch of data becomes available — many forward/backward passes must be performed to update the model over a large batch of data, especially if several loops over the data are performed. Though such alternative experimental setups for online learning are interesting from a research perspective, _why would any practitioner wait for data to accumulate when they have the ability to update the model after the arrival of each new sample?_

The ability of streaming learning to adapt models to streams of data in real time also has wide applications in industry. Consider, for example, a recommendation system that performs dynamic updates each time a user interacts with a website (e.g., a purchase, a click, or even a movement of the mouse). Alternatively, one could utilize a deep network to perform video interpolation (e.g., predict where a person will be in the next frame given the previous several frames) and leverage streaming learning to update this model over the video stream based on the mistakes it makes in each frame. The possibilities of streaming learning are nearly endless, as it applies to any situation in which a deep network should immediately learn from incoming data. Thus, it is (in my opinion) a topic worthy of focus for deep learning practitioners.

### Methodologies for Deep Streaming Learning

Now that streaming learning has been defined and motivated, it’s time to learn how neural networks can actually be trained over data streams without severely degrading their performance. Recently, several algorithms for training deep neural networks in a streaming fashion have been proposed within the deep learning community [4, 7, 8]. For each algorithm, I will overview the main components and details of the methodology, as well as highlight major practical considerations for implementing the algorithm in practice. Throughout this section, I focus on the major details of each approach that allow practitioners to better understand which methodologies are most suitable for a given application.

#### ExStream [4]

ExStream, proposed in February of 2019, is a replay-based methodology for streaming learning. Here, “replay” (also referred to as rehearsal) is used to describe methods that store previously-encountered data from the incoming data stream within a buffer. Then, when new data becomes available, replay-based methodologies mix the new data with samples of data from the replay buffer and use this mix of new and old data to update the model, thus ensuring that previous knowledge is retained. Put simply, these methodologies train the network with a mix of new and old data to ensure that the network is exposed to a balanced set of examples from the data stream. Replay is a widely-used methodology within online learning that is both simple and effective, but it does require storage of previous data, which creates a non-negligible memory footprint.

Although ExStream demonstrates that “full” replay (i.e., storing all incoming data in the replay buffer and looping through all examples in the replay buffer each time new data is encountered) eliminates catastrophic forgetting, more memory-efficient replay mechanisms that don’t require storage of the entire data stream are also explored. Using a ResNet50 [9] to perform image classification, ExStream pre-trains and fixes the convolutional layers of the underlying model (i.e., none of the convolutional layer parameters are updated during the streaming process) and focuses upon learning the final, fully-connected layer in a streaming fashion. Thus, all examples stored within the replay buffer are simply vectors (i.e., the output of the ResNet’s feature extractor/backbone). Using these feature vectors as input, ExStream maintains a separate, fixed-size replay buffer for each class of data (i.e., only `c` vectors can be stored per class) and aims to discover an algorithm that _i)_ minimizes the number of vectors that must be stored for replay (i.e., this limits memory overhead) and _ii)_ maintains state-of-the-art classification performance (i.e., comparable to full replay).

To achieve these goals, ExStream leverages the following rule set for maintaining its replay buffer during streaming:

- Maintain `c` cluster centroids (i.e., just vectors!) per class, each of which has a “count” associated with it.
    
- Until `c` examples for a class have been encountered within the data stream, simply add each vector to the replay buffer with a count of one.
    
- Once the buffer for a given class is full and a new example for that class arrives, find the two closest clusters centroids (based on Euclidean distance) and merge them together by taking a weighted average of the vectors based on their respective counts. Set the count of the resulting centroid to the sum of the counts for the two previous centroids.
    
- Once the two centroids have been merged (thus making room for a new one), add the new data example into the buffer with a count of one.
    

Using the replay buffer that’s maintained as described above, ExStream then performs replay by sampling and mixing cluster centroids from different classes with incoming data to perform updates on model parameters. In comparison to several other algorithms for maintaining cluster centroids within the replay buffer (e.g., online k-means, CluStream [10], HPStream [11], etc.), ExStream is shown to yield the best performance. Further, the memory footprint of the algorithm can be easily tuned by adjusting the number of centroids permitted for each class within the replay buffer (although a replay buffer that is too small could yield poor performance). ExStream is shown to perform well on Core50 and iCUB datasets, but is not applied to large-scale classification problems (e.g., ImageNet) until later work was published.

#### Deep SLDA [7]

Deep Streaming Linear Discriminant Analysis (SLDA), proposed in April of 2020, is another methodology for deep streaming learning that moves away from replay-based methodologies (i.e., it does not maintain any replay buffer). As a result, it is quite memory efficient in comparison to methods like ExStream that require a replay buffer, thus (potentially) making it appropriate for memory constrained learning scenarios (e.g., on-device learning). SLDA is an already-established algorithm [12] that has been used for classification of data streams within the data mining community. Within Deep SLDA, the SLDA algorithm is combined with deep neural networks by _i)_ using a fixed ResNet18 [9] backbone to obtain a feature vector for each data example and _ii)_ employing SLDA to incrementally classify such feature vectors in a streaming fashion. Again, all network layers are fixed during the streaming process for Deep SLDA aside from the final classification module (i.e., the SLDA component).

The specifics behind classifying feature vectors with SLDA in an incremental fashion is beyond the scope of this post — the algorithm is complex and probably deserves and entire blog post of its own to truly develop an understanding. At a high level, however, SLDA operates by _i)_ maintaining a single mean vector per class with an associated count and _ii)_ constructing a shared covariance matrix that characterizes the relationships between class representations. The mean vector is updated for each class as new data becomes available, while the covariance matrix can either be kept fixed (after some base initialization over a subset of training data) or updated incrementally during the streaming process. At test time, the class output for a new data example can be inferred using a closed-form matrix multiplication of mean class vectors with the inverse of the covariance matrix.

SLDA has benefits over replay-based methodologies like ExStream because memory requirements are reduced significantly — only a single vector per-class and a shared covariance matrix must be stored. Furthermore, SLDA is shown to yield impressive classification performance even on large-scale datasets like ImageNet, outperforming popular methods like ExStream [4], iCarl [13], and End-to-End Incremental Learning [14]; see Table 1 of [8]. Additionally, in comparison to normal, offline neural network training over large-scale datasets, the wall-clock training time of Deep SLDA is nearly negligible. Overall, _the method is surprisingly effective at scale given its minimal computation and memory requirements_.

#### REMIND [8]

REMIND, published in July of 2020, is a recently-proposed, replay-based methodology for deep streaming learning. Instead of maintaining cluster centroids within the replay buffer like ExStream, REMIND stores separate buffer entries for each data example that is encountered within the incoming stream. In previous work, this is done by storing raw images within the replay buffer [13, 14]. Within REMIND, however, the authors propose that intermediate activations within the neural network (i.e., these activations are not just a vector — they may have spatial dimensions) should be stored within the replay buffer instead of raw images, which _i)_ drastically reduces per-sample memory requirements and _ii)_ mimics the replay of compressed memories within the brain as outlined by hippocampal indexing theory.

To make the above strategy possible, REMIND adopts a ResNet18 [9] architecture and freezes the initial layers of the network so that the parameters of these layers do not change during the streaming process. Similar to Deep SLDA, the values of these frozen parameters are set using some base initialization phase over a subset of training data. Then, for each example encountered during streaming, REMIND _i)_ passes the example through the network’s frozen layers to extract an intermediate activation, _ii)_ compresses the activation tensor using a product quantization (PQ) strategy [15], and _iii)_ stores the quantized vector within the replay buffer. Then, online updates are performed on the final network layers (i.e., those that are not frozen) with a combination of new data (after it has been quantized) and sampled activations from the replay buffer as input. In practice, REMIND performs random crops and mixup on activations sampled from the replay buffer, which provides regularization benefits and yields moderate performance improvements.

Due to REMIND’s memory efficient approach to replay, it can maintain incredibly large replay buffers with limited overhead. For example, on the ImageNet dataset, REMIND can maintain a buffer of ~1M examples with the same memory footprint as a replay buffer of 10K raw images. As a result, REMIND outperforms both ExStream and Deep SLDA significantly on numerous common benchmarks for online learning; see Table 1 in [8]. The performance benefits of REMIND are especially noticeable on large-scale datasets (e.g., Imagenet), where the ability to store many replay examples with limited memory allows REMIND to truly differentiate itself. Currently, REMIND is the best-performing approach for training deep networks in the streaming domain.

#### Connections to general online learning

Now that the existing approaches to deep streaming learning have been outlined, one may begin to ask how these approaches relate to methodologies that have been proposed for other online learning setups (e.g., batch-incremental learning or lifelong learning). For a more comprehensive description of all methodologies that have been proposed for online learning, I recommend [my previous post](https://towardsdatascience.com/a-broad-and-practical-exposition-of-online-learning-techniques-a4cbc300dcd4) on this topic. However, I try to overview these methodologies at a high level below.

- **Replay:** Widely-used in both streaming learning and online learning in general.
    
- **Knowledge Distillation:** Widely utilized within online learning techniques, but not yet explored for streaming learning. Although knowledge distillation may provide some benefit to streaming learning techniques, several recent papers argue that knowledge distillation provides minimal benefit when combined with replay [16, 17].
    
- **Bias Correction:** Not tested within the streaming setting, but very beneficial for incremental learning.
    

Several other methodologies for online learning exist that have not been explored within the streaming setting (e.g., architectural modifications or regularization-based approaches). However, many of these methodologies are less popular within current online learning research, as they do not perform well when applied to larger-scale problems. As such, it is unlikely that these approaches would outperform proven methodologies for large-scale streaming learning, such as REMIND.

#### What’s best to use in practice?

Though all of the methodologies for deep streaming learning overviewed within this post are useful, a practitioner may wonder which method is most appropriate to implement for their application. In terms of performance, REMIND is the best-performing methodology that has been proposed for deep streaming learning to date. This can be seen within Table 1 of the paper for REMIND [8], where REMIND is shown to significantly outperform both ExStream and Deep SLDA. Additionally, REMIND has even been extended to problem domains such as object detection [16], thus showcasing its utility in applications beyond image classification.

Both ExStream and REMIND require the storage of a replay buffer and have similar computational efficiency, thus making REMIND the obvious choice between the two. However, Deep SLDA does not require such a replay buffer to be maintained and can be trained very quickly. As such, even though REMIND achieves better performance, Deep SLDA may be favorable in scenarios with limited memory or computational resources. Otherwise, REMIND is the best option for deep streaming learning in practice, as it achieves impressive performance and minimizes the memory footprint of the replay buffer through quantization.

For those who are interested, implementations of REMIND [17] and Deep SLDA [18] are both publicly available via github.

### Conclusions

In this post, I overviewed the training of deep neural networks over data streams, including a discussion of why such a training paradigm is practically relevant and a description of existing methodologies for training deep networks in this fashion. Of the relevant methodologies for deep streaming learning, REMIND achieves the best performance, while approaches such as Deep SLDA may be useful in cases with limited memory or computational resources.

Thank you so much for reading this post. If you have any feedback or generally enjoyed the post and want to keep up with my future work, feel free to follow me on [twitter](https://twitter.com/cwolferesearch) or visit [my website](https://wolfecameron.github.io/). This work was done as part of my job as a Research Scientist at [Alegion](https://www.alegion.com/) and a PhD student at Rice University. If you enjoyed the material in this post, I encourage you to check out [open positions](https://www.alegion.com/company/careers) at Alegion, or get in contact with [my research lab](http://akyrillidis.github.io/group/)!

_**Citations**_

[1] [https://arxiv.org/abs/1412.2485](https://arxiv.org/abs/1412.2485)

[2] [https://ieeexplore.ieee.org/abstract/document/8622392](https://ieeexplore.ieee.org/abstract/document/8622392)

[3] [https://epubs.siam.org/doi/abs/10.1137/1.9781611974317.7](https://epubs.siam.org/doi/abs/10.1137/1.9781611974317.7)

[4] [https://arxiv.org/abs/1809.05922](https://arxiv.org/abs/1809.05922)

[5] [https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)

[6] [https://arxiv.org/abs/1708.02072](https://arxiv.org/abs/1708.02072)

[7] [https://arxiv.org/abs/1909.01520](https://arxiv.org/abs/1909.01520)

[8] [https://arxiv.org/abs/1910.02509](https://arxiv.org/abs/1910.02509)

[9] [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

[10] [https://www.vldb.org/conf/2003/papers/S04P02.pdf](https://www.vldb.org/conf/2003/papers/S04P02.pdf)

[11] [https://www.semanticscholar.org/paper/A-Framework-for-Projected-Clustering-of-High-Data-Aggarwal-Han/317c0f3a829ee8f0b3d7e5a915a93c52baa7a9e8](https://www.semanticscholar.org/paper/A-Framework-for-Projected-Clustering-of-High-Data-Aggarwal-Han/317c0f3a829ee8f0b3d7e5a915a93c52baa7a9e8)

[12] [https://ieeexplore.ieee.org/document/1510767](https://ieeexplore.ieee.org/document/1510767)

[13] [https://arxiv.org/abs/1611.07725](https://arxiv.org/abs/1611.07725)

[14] [https://arxiv.org/abs/1807.09536](https://arxiv.org/abs/1807.09536)

[15] [https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)

[16] [https://arxiv.org/abs/2008.06439](https://arxiv.org/abs/2008.06439)

[17] [https://github.com/tyler-hayes/REMIND](https://github.com/tyler-hayes/REMIND)

[18] [https://github.com/tyler-hayes/Deep_SLDA](https://github.com/tyler-hayes/Deep_SLDA)


-------

# Deep Learning on Video (Part One): The Early Days…

### A comprehensive overview of early, deep learning-based approaches for learning tasks on video.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Dec 22, 2021

[

![](https://substackcdn.com/image/fetch/$s_!j1-G!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F97d3f48a-fe8b-4b7c-9349-07c182e0a330_800x261.png)



](https://substackcdn.com/image/fetch/$s_!j1-G!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F97d3f48a-fe8b-4b7c-9349-07c182e0a330_800x261.png)

(created by Author)

Within this series of blog posts, I will explore the topic of deep learning on video data. Beginning from some of the earliest publications on the topic, I aim to outline how this problem was initially approached — prior to the mass popularization of deep learning methodologies— and follow the evolution of video deep learning over time. In doing this, my goal is not only to overview the history of deep learning on video, but also to provide relevant context and understanding for researchers or practitioners looking to become involved in the field.

I will separate this topic into several blog posts. This part will overview the “early days” of deep learning on video. Publications within this period were the first to leverage 3D convolutions to extract features from video data in a learnable fashion, moving away from the use of hand-crafted image and video feature representations. Examples of these approaches include two-part architectures that pass hand-crafted video features into 3D convolutional layers, unsupervised learning techniques based upon 3D convolutions, or even 3D convolutional networks that operate directly upon video data.

This post will be separated into three sections. I will first provide relevant background information (e.g., how 3D convolutions work, methodologies before 3D convolutions, how video data is structured). Then, I will overview the relevant literature, providing a high-level (but comprehensive) understanding of early methodologies for deep learning on video. Finally, I will conclude the post and overview the limitations of such methodologies, looking toward the better-performing approaches that were later developed.

### Preliminaries

Before diving into relevant literature, there are several concepts that must be outlined. In describing these concepts, I assume the reader has a basic knowledge of convolution operations and the typical structure of image data within computer vision applications. However, I will still try to provide all relevant information, such that important concepts can be understood with minimal prior knowledge.

#### **How is video data structured?**

[

![](https://substackcdn.com/image/fetch/$s_!V3mP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd0efc34c-252a-4585-852d-90a60d9046c2_800x384.png)



](https://substackcdn.com/image/fetch/$s_!V3mP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd0efc34c-252a-4585-852d-90a60d9046c2_800x384.png)

Comparison between the structure of Image and Video Data (created by Author)

In developing deep learning applications for video, the first question one may ask themselves is how video data is typically structured (e.g., in comparison to images). For images, this question has a simple answer. In most cases, input images are RGB-format (i.e., each image has three color channels associated with it) and have a certain height and width. For example, within the figure above, the input image has three color channels, each with a height and width of five. Thus, input into an image-based deep learning model will usually be a [tensor](https://discuss.pytorch.org/t/what-exactly-is-a-tensor/8861) of size `3 x Height x Width`. Several images can then be stacked into a mini-batch, forming a tensor of size `Batch x 3 x Height x Width`.

For videos, the data structure is not much different. Namely, a video is simply a collection of temporally-ordered images, called frames. When viewed in the correct temporal order, these frames reveal the movement of a scene through time, forming a video. Similar to images, each of these frames will typically be represented in RGB-format and have the same spatial resolution (i.e., all frames have the same height and width). In the figure above, the video is comprised of three frames, each with three color channels and a height and width of five.

All frames within a video can be combined into a single “volume”, forming a tensor of size `3*Frames X Height X Width` by “stacking” all frames on top of each other (i.e., frames can also be aggregated into a tensor of size `Frames X 3 X Height X Width` depending on the application). Combining several videos into a mini-batch must be done carefully, as the number of frames within each video may differ, forming video tensors of different sizes. Nonetheless, different videos can be “chunked” into equal numbers of contiguous frames (or a padding approach could be adopted) to form videos of equal size that can be combined together to form a mini-batch of size `Batch x 3*Frames x Height x Width`.

#### What datasets were out there?

Within the literature that will be covered within this post, the main datasets used for evaluation include: [TRECVID](https://trecvid.nist.gov/), [KTH](https://www.csc.kth.se/cvap/actions/), [Hollywood2](https://www.di.ens.fr/~laptev/actions/hollywood2/), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), and [YouTube Action](http://www.cs.ucf.edu/~liujg/YouTube_Action_dataset.html). All of these datasets are focused upon the problem of human action recognition (HAR), in which a categorical label must be assigned to a video clip based upon the action being performed by the human in the clip. HAR, which has numerous applications to problem domains such as autonomous driving and surveillance, was, by far, the most widely-studied application within the early days of video deep learning. Although later research considered other, more complicated applications, HAR remains a notable problem to this day. Interestingly, however, it can be shown that image-based models that operate independently on frames in a video perform well on HAR [1, 2], revealing that only minimal temporal reasoning is required to provide a reasonable solution.

#### From 2D to 3D…

Prior to describing proposed methodologies for deep learning on video, it is useful to consider how constructions from deep learning on images can be generalized to the video domain. In particular, we must consider how 2D convolutions — the workhorse behind convolutional neural networks (CNNs) — can be applied to video. CNNs, which are simply sequences of parallel convolution operations separated by nonlinearities, have long been applied to popular, image-based tasks like image classification, object detection, and key point estimation, yielding great success. As such, it makes sense that early work in deep learning on video would try to extend such a successful approach to operate on video data. A 2D convolution operation is depicted below.

[

![](https://substackcdn.com/image/fetch/$s_!zJSj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F20257bbf-484b-4dff-9786-06f69b64425b_800x200.png)



](https://substackcdn.com/image/fetch/$s_!zJSj!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F20257bbf-484b-4dff-9786-06f69b64425b_800x200.png)

2D Convolution Operation on an RGB Image (created by Author)

As can be seen, within a 2D convolution, a single convolutional kernel, which spans all channels of the input (i.e., the three color channels in the image above), is moved over each spatial location of the input to individually compute corresponding output elements. Here, each output element is derived by multiplying pixel values at the correct spatial locations within the input with their corresponding weights in the kernel and summing over all such multiplications. Note that this description considers only the simplest case of a 2D convolution (i.e., RGB input, no stride, etc.). I recommend [this blog post](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) to learn further details about 2D convolutions.

When we begin to consider video data, the input into the convolution operation changes, as we now have multiple frames of video instead of a single image. As such, the input is no longer a single RGB image, but a volume of RGB images stacked on top of each other (i.e., this can be viewed as a similar input with a larger number of channels). To account for multiple frames within the input, the convolution operation must span not only spatial dimensions (e.g., our 2D convolution in the image above spans three pixels in height and width), but also temporal dimensions. Namely, a single kernel must consider multiple frames of the input in computing its corresponding output representation. This formulation, which we refer to as a 3D convolution operation, is depicted below.

[

![](https://substackcdn.com/image/fetch/$s_!JB0H!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4fdfbe0d-16f4-49ee-a2ef-873eeedf5f1c_800x341.png)



](https://substackcdn.com/image/fetch/$s_!JB0H!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4fdfbe0d-16f4-49ee-a2ef-873eeedf5f1c_800x341.png)

3D Convolution Operation on Three RGB Frames (created by Author)

As can be seen, the 3D convolution above has a kernel that spans two frames within the input. Initially considering the first two frames, this kernel is moved over each spatial dimension to compute corresponding output elements (i.e., through element-wise multiplication and summing similarly to 2D convolutions). After moving over possible spatial dimensions, the 3D convolution takes a step in time, moving to the next span of frames it must consider, and repeats the process of traversing spatial dimensions. In the image above, the 3D convolution initially considers the first two frames within the video, then moves on to compute representations corresponding to frames two and three. If more frames existed in the video, the 3D convolution would continue traversing time in this fashion.

A 3D convolution is a spatio-temporal operation that considers both spatial features within each individual input frame and temporal features within consecutive frames. Although 3D convolutions are not the only way to learn features from video data, they are the main learnable transformation leveraged by early approaches to deep learning on video. Thus, 3D convolutions are a primary focus of this post.

#### What was used before deep learning?

Although CNNs were widely successful within the image domain, extending their applications to video took time, especially due to the lack of large, diverse video databases (relative to the size and diversity of image-based datasets) for training purposes [1]. As such, one may begin to wonder how HAR was usually tackled prior to the popularization of deep learning-based systems for action recognition.

Initial systems for HAR extracted fixed, hand-crafted features from video data (e.g., using motion history images, SIFT descriptors, HoG features, template matching, filter banks, etc.) to aid in classification. Typically, the video would be first segmented into interest points (i.e., shorter clips with the human cropped from the background). Then, fixed features were extracted from these interest points (i.e., comprised of several cropped frames) and classified with some machine learning model (e.g., a support vector machine). Such methodologies were later extended to form more complex systems that learned features hierarchically through alternating applications of hand-crafted feature extraction and sub-sampling [3, 4].

Although hand-crafted approaches worked relatively well, such features struggle to generalize across datasets and are difficult to adapt to different sensor domains (e.g., video or radar) [5]. Additionally, hand-crafted features tend to make simplifying assumptions about the underlying data distribution (e.g., fixed viewpoint, human at the center of frame, etc.) that degrade performance in real-world applications [6]. Thus, researchers in this domain began to search for alternative feature extraction methodologies that can be learned directly from data. Due to the ability of CNNs to extract useful features from images, using 2D/3D convolutions to learn features from video data quickly became a primary area of focus. This post overviews the methodologies that emerged as a result.

### Early Methodologies for Deep Learning on Video

I will now overview relevant, early methodologies that pioneered the use of 3D convolutions to extract features from video data in a learnable fashion. The publications that will be covered within this post can be roughly categorized as methods that combine learnable feature extraction with fixed feature extraction, methods that apply learnable feature extraction directly to video data, and unsupervised methods for learning spatio-temporal features in video.

#### A Hybrid Approach: Combining Hand-Crafted and Learned Features

Although CNNs were known to be capable of learning useful, discriminative features when trained on large image datasets, early extensions of CNNs into the video domain typically combined the use of 3D convolutions with fixed feature extraction instead of applying 3D convolutions directly to the raw video data. In particular, the video data would first be pre-processed by a fixed feature extractor, such as by computing optical flow or vertical and horizontal gradients over the raw video frames. Then, after pre-processing, the resulting representation would be passed through layers of 3D convolutions to extract learnable features for classification.

This approach was adopted by initial HAR systems applied to the TRECVID surveillance dataset [7, 8]. Such methodologies, instead of operating directly on the raw video data, computed [optical flow](https://nanonets.com/blog/optical-flow/) and [directional gradients](https://en.wikipedia.org/wiki/Image_gradient) within the input video. These hand-crafted features were then concatenated with the original video frames and passed through several layers of 3D convolutions, which are separated by sub-sampling operations and non-linear activation functions. Within these 3D CNN architectures, each data modality (i.e., optical flow, directional gradient, and normal frames) is processed independently by separate 3D convolutional kernels. Then, the output of the 3D CNN is used for classification, either with a support vector machine [7] or a fully-connected, linear module [8]. Within these works, several different CNN architectures were explored for HAR, leading to the conclusion that combining the output of several, unique 3D CNN architectures typically leads to the best performance [8].

Such approaches were highly-successful at the time, even outperform single-frame CNNs in HAR (i.e., a strong baseline approach) and nearly all baselines that utilize hand-crafted feature extraction. Nonetheless, such methodologies were still inherently dependent upon the quality of the optical flow and gradient features computed within the initial pre-processing stage. Such features are not learned and, therefore, may not generalize well across datasets or applications. This observation led to the proposal of methodologies that completely eliminate fixed feature extraction in an attempt to extract fully-learnable features that have no dependence upon any hand-crafted heuristic.

#### Extracting Learnable Video Features with 3D CNNs

To avoid limitations associated with approaches based upon hand-crafted features, later deep learning methodologies for video passed raw input volumes (i.e., groups of several frames) directly into layers of 3D convolutions separated by pooling and activation layers [9]. Such 3D CNNs were structurally similar to image-based CNNs, but modified to contain 3D convolutions [9]. Typically, these approaches extract a small subset of frames from the full video, crop the frames to focus upon the humans within each frame, then pass the sub-sampled and cropped volume into the 3D CNN [9, 10], where the output is then used to perform classification. Interestingly, the performance of hybrid 3D CNNs (i.e., those that combine hand crafted and learned features) can be easily exceeded by such an approach, even with the use of a much smaller network [10].

Notably, because such approaches segment the underlying video into shorter clips (e.g., containing 10–15 frames) that are passed into the 3D CNN, the final classification does not consider relationships between larger or temporally-distant groups of frames within the full video. To solve this shortcoming, the approach described above was combined with recurrent neural networks — a long short-term memory (LSTM) network [11] in particular — to synthesize information extracted with 3D CNNs from each clip in a video. In particular, the video was first segmented into smaller groups of frames, which are again passed into the 3D CNN to extract features. However, instead of performing classification on each of these features independently, the features associated with each video segment are passed, in temporal order, as input to an LSTM, which then performs the final classification [10]. Such a methodology was shown to consider the context of the full video and, as a result, achieve significantly improved performance on HAR.

The use of 3D CNNs to extract features directly from video data was revolutionary, as it paved the way for deep learning to become a dominant approach for learning tasks on video. These works moved away from traditional video learning methodologies and followed the approach of related deep learning literature for image recognition, choosing to let all features be obtained in a learnable fashion. As such, systems for video learning tasks (e.g., HAR) became less domain-dependent and video-related research could evolve at a faster pace, as the need to constantly determine the best hand-crafted features for each possible dataset was eliminated [12]. However, one major limitation still exists for such approaches — their performance is entirely dependent on the quality and abundance of labeled data for the underlying task.

#### Unsupervised Representation Learning for Video

Despite the incredible value provided by 3D CNN models, their performance is limited by the amount of data available for training. Especially during the early days of deep learning, the availability of labeled video data was minimal [1]. As such, the abilities of 3D CNN models to learn high-quality features was always somewhat limited due to a lack of sufficient data. Such a limitation led to the simultaneous development of unsupervised deep learning methodologies for extracting meaningful video features [12, 13].

The first of such unsupervised learning methodologies for video drew upon a gated, restricted Boltzmann machine (GRBM) architecture to learn features from adjacent frames in a video. Initially, the GRBM architecture could not handle processing high-resolution images (i.e., it was too computationally expensive), due to the flattening of images into a vector before being passed as input to the model. To mitigate this problem, the GRBM was modified in [13] to use convolution operations, allowing for higher-resolution images to be processed at a lower computational cost. This convolutional GRBM model could then be trained on adjacent video frames in an unsupervised manner to learn useful spatial and temporal information, thus allowing high-quality features to be extracted from the video without the need for labeled data. From here, it was then shown that the features extracted from the convolutional GRBM could be passed into a 3D CNN to solve tasks like HAR, where performance was greatly improved by the unsupervised pre-training of the GRBM [13].

Although the convolutional GRBM was shown to achieve impressive performance, the unsupervised pre-training process was quite computationally inefficient, leading to the proposal of an alternative, unsupervised learning methodology for video, inspired by independent subspace analysis (ISA) [12]. ISA is a biologically-inspired, sparse learning methodology, based upon [independent component analysis](https://en.wikipedia.org/wiki/Independent_component_analysis), that was previously used to extract meaningful features from static images and can be trained in an unsupervised manner. Similar to GRBMs, ISA was too computationally inefficient to be applied to high-resolution images. To solve this issue, [12] proposed that ISA be first trained (i.e., in an unsupervised manner) on smaller image patches from the frames of a video. Then, the dimensionality of ISA output on these initial frame patches could be reduced — using principal component analysis (PCA) — enough to be passed through another ISA module. Then, the output of these modules could be fed into a 3D CNN to extract features for classification [12]. Such an approach greatly improved the efficiency of unsupervised representation learning on video and achieved state-of-the-art performance on nearly all datasets for HAR.

### Conclusions and Looking Forward

Within this post, we overviewed early methodologies that utilized 3D convolution operations to extract learnable features from video. Originally, such work adopted a hybrid approach, in which hand-crafted features were extracted and mixed with learnable features from 3D CNNs. Later, all hand-crafted components were eliminated, and video features were extracted in a completely learnable fashion by applying 3D CNNs directly to raw video data. Then, in parallel, several unsupervised approaches for learning meaningful video features were proposed, thus mitigating dependence upon labeled video data in solving common learning tasks (e.g., HAR).

Although early deep learning approaches for video worked relatively well, they were quite inefficient and did not match the impressive performance achieved by deep learning on image-based tasks. The approaches outlined in this work only narrowly outperformed single-frame baseline models and were even occasionally outperformed by hand-crafted baseline methodologies on certain datasets. As such, a more performant approach was needed that maintains or improves the computational efficiency of the 3D CNN methodologies outlined within this post. Eventually, this need led to the proposal of two-stream 3D CNN architectures, which will be overviewed in the next post of this series.

Thank you so much for reading this post! I hope you found it helpful. If you have any feedback or concerns, feel free to comment on the post or reach out to me via [twitter](https://twitter.com/cwolferesearch). If you’d like to follow my future work, you can follow me on Medium or check out the content on my [personal website](https://wolfecameron.github.io/). This series of posts was completed as part of my background research as a research scientist at [Alegion](https://www.alegion.com/). If you enjoy this post, feel free to check out the company and any relevant, open positions — we are always looking to discuss with or hire motivated individuals that have an interest in deep learning-related topics!

_Bibliography_

[1][https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)

[2] [https://ieeexplore.ieee.org/document/1495508](https://ieeexplore.ieee.org/document/1495508)

[3] [https://mcgovern.mit.edu/wp-content/uploads/2019/01/04069258.pdf](https://mcgovern.mit.edu/wp-content/uploads/2019/01/04069258.pdf)

[4] [https://www.cs.tau.ac.il/~wolf/papers/action151.pdf](https://www.cs.tau.ac.il/~wolf/papers/action151.pdf)

[5] [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5995496](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5995496)

[6] [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6165309](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6165309)

[7][https://www.researchgate.net/publication/229045898_Detecting_Human_Actions_in_Surveillance_Videos](https://www.researchgate.net/publication/229045898_Detecting_Human_Actions_in_Surveillance_Videos)

[8] [https://ieeexplore.ieee.org/document/6165309](https://ieeexplore.ieee.org/document/6165309)

[9] [https://link.springer.com/content/pdf/10.1007/978-3-540-72393-6_85.pdf](https://link.springer.com/content/pdf/10.1007/978-3-540-72393-6_85.pdf)

[10] [https://link.springer.com/content/pdf/10.1007%2F978-3-642-25446-8_4.pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-642-25446-8_4.pdf)

[11] [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[12] [https://ieeexplore.ieee.org/document/5995496](https://ieeexplore.ieee.org/document/5995496)

[13] [http://yann.lecun.com/exdb/publis/pdf/taylor-eccv-10.pdf](http://yann.lecun.com/exdb/publis/pdf/taylor-eccv-10.pdf)

------

# Deep Learning on Video (Part Two): The Rise of Two-Stream Architectures

### How two-stream network architectures revolutionized deep learning on video.

[](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Jan 29, 2022

[

![](https://substackcdn.com/image/fetch/$s_!9zoo!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F71383b8a-9fc7-4773-b0d0-65256c130d84_800x432.png)



](https://substackcdn.com/image/fetch/$s_!9zoo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F71383b8a-9fc7-4773-b0d0-65256c130d84_800x432.png)

Two-Stream Architecture for Video Understanding (Image by Author)

This post is the second in a series of blog posts exploring the topic of deep learning on video data. The goal of this series of blog posts is to both overview the history of deep learning on video and provide relevant context for researchers or practitioners looking to become involved in the field. In the [first post](https://towardsdatascience.com/deep-learning-on-video-part-one-the-early-days-8a3632ed47d4) of the series, I overviewed the earliest publications on the topic that used 3D convolutions to extract learnable features from video.

Within this post, I will overview the next major phase of video deep learning: the introduction and popularization of two-stream network architectures. Two-stream architectures for video recognition are composed of two separate convolutional neural networks (CNNs) — one to handle spatial features and one to handle temporal/motion features. These separate CNNs are typically referred to as the “spatial” and “temporal” networks within the two-stream architecture, and the output of these separate network components can be combined together to form a spatiotemporal video representation. Two-stream architectures yielded massively-improved performance in video action recognition, making them a standard approach to video deep learning for some time.

The post will begin by overviewing relevant preliminary information, such as the definition/formulation of two-stream architectures and the limitations of previous work. I will then overview the literature on two-stream network architectures, including the papers that originally proposed the architecture and later, more complex variants. At the end of the post, I will discuss other, related approaches to video understanding that were proposed during this time period and outline the limitations of two-stream networks, thus motivating improvements that were yet to come.

### Preliminaries

Several preliminary concepts must be discussed prior to outlining two-stream approaches to video deep learning. For all details regarding the formulation of 2D/3D convolutions, the structure of video data, and existing approaches to video deep learning prior to two-stream architectures, I refer the reader to the [first post](https://towardsdatascience.com/deep-learning-on-video-part-one-the-early-days-8a3632ed47d4) in the series. Given an understanding of these concepts, however, I try to overview relevant information in a way that is understandable even with minimal background knowledge.

#### What problem are we trying to solve?

[

![](https://substackcdn.com/image/fetch/$s_!jFFU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9e93eaba-43da-4ac5-8f0a-756718821e27_800x254.png)



](https://substackcdn.com/image/fetch/$s_!jFFU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9e93eaba-43da-4ac5-8f0a-756718821e27_800x254.png)

Human Action Recognition Problem Depiction (Image by Author)

The majority of methodologies overviewed within this post study the problem of video-based [human action recognition](https://en.wikipedia.org/wiki/Activity_recognition) (HAR). HAR datasets contain a number of variable-length videos that are each associated with a semantic label, corresponding to the action being performed in that video. Typically, the videos within the dataset are focused upon a single entity that is performing the action in question, and the video does not extend far before or after the action is performed. Thus, the goal of the underlying model is to predict the semantic action label given the video as input. HAR was, by far, the most commonly-studied video understanding problem at the time of two-stream network architectures.

#### Why do we need something new?

In the first post of the series, we overviewed several possible approaches for video deep learning. These approaches typically adopt 3D CNN models (i.e., several consecutive layers of 3D convolutions separated by non-linear activation layers), pass either raw video or hand-crafted video features (e.g., [optical flow](https://en.wikipedia.org/wiki/Optical_flow) or [directional gradients](https://en.wikipedia.org/wiki/Image_gradient)) as input to these models, and perform supervised training via back propagation based on the semantic label assigned to each video. Given that these methodologies exist, one may begin to wonder why a new approach to video deep learning is needed.

_The answer to this question is quite simple — existing models just performed poorly._ In fact, earlier deep learning-based approaches to HAR were often outperformed by hand-crafted, heuristic methods and performed comparably to deep learning models that use individual frames as input (i.e., completely ignoring the temporal aspect of video) [1, 2]. Such poor performance was shocking given the massive success of deep learning in the image recognition domain [3]. As such, the research community was left wondering how deep learning could be made more useful for video.

The initially poor performance of 3D CNN architectures was mostly attributed to the lack of large, supervised datasets for video understanding [1]. For example, [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), the most commonly-used datasets for HAR at the time, each contain only 13,320 and 7,000 labeled video clips, respectively. In comparison, ImageNet — a widely-used benchmark for image classification — contains ~1.3 million training examples. Although larger datasets (e.g., Sports1M [1]) were proposed for HAR, they were usually collected automatically and quite noisy, leading smaller, curated datasets to be used more often. Because 3D CNNs contain a larger number of parameters in comparison to their 2D counterparts, a lot of data is required for them to learn meaningful, discriminative representations. Thus, the small-scale datasets of the time were not sufficient — _either more data or a different learning paradigm was needed to enable better performance_.

#### The Two-Stream Network Architecture

The two-stream architecture took the first step towards deep learning-based approaches surpassing the performance of heuristic and single-frame methods for HAR, catalyzing the onset of a new era in video understanding. Put simply, this architecture enabled high-performance video understanding despite a lack of sufficient supervised data by encoding motion information directly into the network’s input. We will now overview the basic ins and outs of the two-stream architecture to provide context for the relevant research overviewed throughout the rest of the post.

The two-stream network architecture [2] is motivated by the two-stream hypothesis for the human visual cortex in biology [4], which states that the brain has separate pathways for recognizing objects and motion. Attempting to mimic this this structure, the two-stream network architecture for video understanding utilizes two separate network components that are dedicated to processing spatial and motion information, respectively. Thus, the two-stream architecture delegates the tasks of object recognition and motion understanding to separate network components, forming different pathways for spatial and temporal cues.

The input to the two stream architecture is typically centered around a single frame within the input video, which is directly passed as input into the network’s spatial stream (i.e., no adjacent frames are considered). As input to the temporal stream, `L` consecutive frames (centered around the frame passed as input to the spatial stream) are selected. The horizontal and vertical [optical flow](https://en.wikipedia.org/wiki/Optical_flow) fields are then computed for each of the adjacent frames in this group, forming an input of size `H x W x 2L` (i.e., `H` and `W` are just the height and width of the original image). Then, this stack of optical flow fields is passed as a fixed-size input into the network’s temporal stream.

From here, the spatial and temporal streams process the frame and optical flow input using separate convolutional networks with similar structure — the only difference between the respective networks is that the temporal stream is adapted to accept input with a larger number of channels (i.e., `2L` channels instead of 3). Once the output of each stream is computed, stream representations are fused together to make a single, spatiotemporal representation that is used for prediction. Later two-stream architectures make use of more sophisticated fusion strategies between the two streams, but for now we assume this simple, “late” fusion strategy. This formulation of the two-stream architecture is illustrated below.

[

![](https://substackcdn.com/image/fetch/$s_!0f7w!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc8336062-bc9a-44d1-b4af-ea3a7f8e1f38_800x250.png)



](https://substackcdn.com/image/fetch/$s_!0f7w!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc8336062-bc9a-44d1-b4af-ea3a7f8e1f38_800x250.png)

Illustration of a Basic, Two-Stream Network Architecture (Image by Author)

As shown above, the input to the two-stream architecture only contains a single frame for the spatial stream and a fixed-size group of optical flow maps for the temporal stream. Although one may argue this approach is limited because it only looks at a fixed-size, incomplete portion of the video, this issue can be mitigated by sampling several of these fixed-size clips from the underlying video and averaging their output to yield a final prediction. Furthermore, the clips used as input to the two-stream architecture could be sampled with a stride (i.e., instead of sampling adjacent frames, sample those those at consecutive intervals of two, three, four, etc. frames) so that the network considers a larger temporal extent within the underlying video.

#### Why does this work?

After providing the formulation of a basic two-stream network architecture, one may begin to wonder why such an architecture would be superior to something like a 3D CNN. After all, 3D CNNs have very high representational capacity (i.e., lots of parameters), so they should be able to learn good spatial and temporal features, right?

Recall, however, that the amount of supervised data for video understanding was limited at the time two-stream architectures were proposed. As such, the two-stream approach provides a few major benefits that enable them to exceed the performance of 3D CNNs. First, the fact that optical flow is passed directly as input to the temporal stream allows motion-based features to be learned more easily, as information relevant to motion (i.e., the optical flow) is passed directly as input instead of being learned. Additionally, because the spatial stream is just a 2D CNN operating on a single frame, it can be pre-trained on large image-classification datasets (e.g., ImageNet), which provides a massive performance benefit. These points of differentiation between two-stream architectures and 3D CNNs are depicted below.

[

![](https://substackcdn.com/image/fetch/$s_!vIjW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F89f7633b-f46f-4724-9fa7-a1515d78050a_800x604.png)



](https://substackcdn.com/image/fetch/$s_!vIjW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F89f7633b-f46f-4724-9fa7-a1515d78050a_800x604.png)

Depiction of the Differences between Two-Stream and 3D CNN Networks (Image by Author)

While 3D CNNs represent space and time as equivalent dimensions (i.e., this contradicts the two-stream hypothesis in biology) [1], two-stream architectures enable better performance because _i)_ motion information is encoded directly in the input (i.e., no longer any need to learn this from data) and _ii)_ large amounts of image classification data can be leveraged to train the spatial network. In the low-data regime, this basic two-stream architecture took a large step towards surpassing the performance of the best hand-crafted, heuristic methods for video understanding.

### Evolution of the Two-Stream Architecture

Now that the two-stream architecture and other preliminary information has been introduced, I will explore some of the literature behind the proposal and development of two-stream architectures for video understanding. I will begin with early papers that explored the topic, followed by more advanced architectural variants — still following the two-stream approach — that later emerged.

#### Early Approaches

**Context and Fovea Streams [1].** The concept of two-stream architectures, although popularized more formally by a later paper [2], was loosely explored in [1]. Within this paper, the authors created two separate processing streams for input data: context and fovea streams. Each of these separate stream share the same network architecture and take the same number of frames as input.

To improve computational efficiency, frames are reduced to 50% of their original area prior to being provided as input to each of the streams, but the context and fovea stream adopt different approaches for reducing the size of the input. Namely, frames within the context stream are just resized, while frames in the fovea stream are center cropped. Put simply, such an approach ensures that context and fovea streams receive low and high-resolution input, respectively — one network sees full frames at low resolution, while the other sees only the center of each frame but at full resolution.

Notice that, unlike the original description provided for two-stream architectures, this approach does not explicitly try to separate motion and spatial recognition into separate processing streams. Instead, each stream is given the same group of raw frames — as opposed to a single frame and optical flow stack — as input (i.e., just resized/cropped differently) and passes such frames through identical, but separate network architectures. Then, the outputs of the two streams are combined prior to prediction. See the figure below for a depiction.

[

![](https://substackcdn.com/image/fetch/$s_!0PbP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F80c91994-6434-470e-b454-9f761978785e_800x276.png)



](https://substackcdn.com/image/fetch/$s_!0PbP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F80c91994-6434-470e-b454-9f761978785e_800x276.png)

Context and Fovea Streams for Video Understanding (Image by Author)

Given that both streams are responsible for detecting both spatial and temporal features, one must determine how to best incorporate temporal information within the streams. We cannot just adopt 2D CNNs within each stream, as this will never consider relationships between adjacent frames. To determine how to best fuse spatial and temporal information, the authors test several possibilities for the CNN architectures of each stream:

- _Early Fusion:_ change the first convolutional layer of each stream to a 3D convolution.
    
- _Late Fusion:_ use 2D CNNs for each stream, compute their output on two frames that are 15 frames apart, then merge the final output for both frames in each stream.
    
- _Slow Fusion:_ change all convolutional layers within each stream to be 3D convolutions with a smaller temporal extent (i.e., kernel size is smaller in time) in comparison to early fusion.
    

The authors find that slow fusion consistently performs best. As a result, the final network architecture adopts a two-stream approach (loosely), where each stream is a 3D CNN that takes a group of frames as input. The only distinction between these streams is their input — frames are resized within the context stream (i.e., lower resolution) and center cropped within the fovea stream (i.e., higher resolution). Although this approach is efficient in comparison to previous 3D CNNs (i.e., due to reducing the dimensionality of input images) and performs comparably, it only performs slightly better than single-frame, 2D CNNs on HAR and is often outperformed by hand-crafted, heuristic methods. Thus, this approach had to be extended and improved upon.

**The Original Two-Stream Architectures [2].** The two-stream architecture described within the preliminaries section was proposed shortly after the architecture described in the previous section [1]. This architecture was the first to adopt the approach of handling spatial and temporal features separately within each stream by passing, as input, a frame or a stack of optical flow maps to the spatial and temporal streams, respectively. As a result, the two-stream architecture was one of the first to make an explicit effort at capturing motion information within the underlying video. By adopting late fusion as described within the preliminaries, the spatial and temporal network output could be combined to form highly robust spatiotemporal features.

The two-stream architecture, as originally proposed, was the first deep learning-based methodologies to achieve consistently improved performance in comparison to single-frame and heuristic baseline methodologies on HAR benchmarks. Thus, it became a standard for video understanding that was heavily studied, utilized, and extended in later work. The dependence of the two-stream architecture upon hand-crafted optical flow features as input (as well as several other previously-discussed aspects of the architecture’s design) enabled better performance in the face of limited data, but this dependence upon hand crafted-features (i.e., optical flow) as input was eventually seen as a limitation to the architecture’s design.

**Best Practices [5].** In addition to the main papers that originally proposed and explored the two-stream network architecture, following work adopted this architecture and explored the best practices for achieving optimal performance. In particular, [5] explored deeper variants of the original two-stream architecture, finding that using CNN backbones with more layers (e.g., VGG [6] and inception-style networks [7]) within each stream of the architecture can yield significant performance benefits if trained properly. The authors claim that the improvement in performance from deeper two-stream architectures comes from the increased representational capacity of the underlying network, which is beneficial for complex tasks like HAR.

To yield the best possible performance, the spatial and temporal streams are _**both**_ pre-trained (i.e., as opposed to just pre-training the spatial stream) using image classification and optical flow data (i.e., generated from an image recognition dataset), respectively. Then, the model is trained with a low learning rate with high levels of data augmentation and regularization, yielding final performance that exceeds that of previous implementations of two-stream architectures on HAR.

#### Advanced Variants

Following the initial proposal of the two-stream architecture, several followup works proposed variants of the architecture (with slight modifications) that yielded massively-improved performance. These more advanced variants maintained the same general network architecture and input schema, but added supplemental modules or connections within the network to improve the representation of temporal information. These modification were motivated by the fact that original two-stream networks relied heavily upon spatial information and did not represent temporal data well.

**Improved Fusion for Two-Stream Architectures [8].** Initial criticisms to the two-stream architecture claimed that the existing formulation did not properly synthesize spatial and temporal information. Namely, because temporal and spatial features were only fused at the output layer of the two streams, the model does not learn to utilize temporal information properly and relies mostly upon spatial information to generate a correct classification. Additionally, the temporal “scale” of the two-stream architecture was limited, as it only considered a fixed-size subset of frames as input to the temporal stream (i.e., as opposed to the full video).

To improve the fusion of temporal information within the two-stream architecture, the authors of [8] explored numerous methods of fusion between feature representations of spatial and temporal streams within the two-stream architecture. As recommended by previous work [5], deeper VGG networks are adopted as the backbone for each stream. Then, the authors consider the following fusion types: sum, max, concatenate, convolutional (i.e., concatenate feature maps then convolve with a bank of 1x1 filters), and bilinear (i.e., compute matrix outer product over features at each pixel and sum over pixel locations to output a single vector). After testing each of these fusion types at different layers within the two-stream network, the authors find that adopting convolutional-style fusion combined with a temporal pooling operation after the last convolutional layer (i.e., before ReLU) of the VGG networks yields the best performance.

Beyond developing a better fusion methodology within the two-stream network, the authors also propose a sampling methodology that allows the underlying network to consider frames across the entire video. Namely, several different “clips” are sampled throughout the video, each of which have a different temporal stride; see the image below. By sampling clips at numerous different locations in the video and utilizing both small and large strides, the final methodology is able to consider a large number of frames within the underlying video despite the fixed-size input to the two-stream architecture. When this sampling methodology is combined with previously-proposed architectural improvements, the authors are able to set a new state-of-the-art performance in HAR benchmarks.

[

![](https://substackcdn.com/image/fetch/$s_!bb4m!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe046db6-1416-40c6-b2e9-7103fd180c71_800x444.png)



](https://substackcdn.com/image/fetch/$s_!bb4m!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe046db6-1416-40c6-b2e9-7103fd180c71_800x444.png)

Clip Sampling from an Underlying Video with Varying Temporal Stride (Image by Author)

**Residual Two-Stream Architectures [9].** Shortly after the proposal of improved fusion techniques, the two-stream architecture was adapted to utilize a ResNet-style CNN architecture within each of its streams. Such a modification was motivated by the incredible success of the ResNet family of CNN architectures for image recognition [10], which remain a widely-used family of architectures to this day. Beyond the popularity of the ResNet architecture, however, many best practices for achieving optimal performance with CNN architectures for image recognition had emerged (e.g., batch normalization [11], maximizing the receptive field, avoiding information bottlenecks, etc.) that were not yet used within video deep learning. Thus, authors of [9] attempted to introduce these numerous improvements, including the ResNet architecture, to two-stream networks.

The ResNet architectures used for the spatial and temporal streams within [9] are modified slightly from their original formulation. Namely, supplemental residual connections are added _i)_ in between the spatial and temporal streams (i.e., the authors claim that this connection aids the fusion of spatial and temporal information) and _ii)_ between adjacent frames being processed in the temporal stream (i.e., this is implemented by converting numerous convolutional layers within the network streams to utilize 3D convolutions). As a result of these supplemental residual connections, the resulting architecture has a large spatiotemporal receptive field (i.e., the full extent of the video is considered in both space and time).

Interestingly, the parameters of both network streams are initialized using pre-trained weights from ImageNet. Then, 3D convolutions are initialized in a way that uses the same weights as the original corresponding 2D convolution for the center frame, but forms residual connections through time (i.e., just an identity operation) to each of the adjacent frames within the 3D convolution’s receptive field. This residual, two-stream architecture (i.e., including the supplemental residual connections between streams and through time) is shown to learn and extract features that better represent the evolution of spatial concepts through time, thus further improving upon the performance of previous approaches for HAR.

### What else did people try?

Although the two-stream architecture was a very popular choice for deep learning on video, not all research on video understanding during this time leveraged such an approach. In fact, many other interesting algorithms were developed in parallel to the two-stream architecture that were able to achieve impressive performance on HAR (or other video understanding benchmarks) despite using a completely different approach.

**New 3D CNN Variants [12].** Another popular architecture was the C3D network [12], which adopts a convolutional architecture that is fully-composed of 3D convolutions (i.e., `3x3x3` kernels in every layer). In particular, this network takes a fixed-length set of frames as input and passes them through a sequence of convolutional and pooling layers, followed by two fully-connected layers at the end of the network. To train this network, authors use the massive Sports1M dataset for HAR [1], thus enabling the network to learn discriminative features over a large dataset (i.e., recall that previous attempts at 3D CNNs performed poorly due to a lack of sufficient supervised data). Nonetheless, this architecture was outperformed by more advanced two-stream variants and criticized for only considering a limited temporal window within the video. As a result, C3D gained less popularity than two-stream architectural variants.

**Siamese Networks [13].** In a different vein, concurrent work on video recognition explored the use of siamese networks to model actions in video [13]. More specifically, this work claimed that _any action within a video could be defined by the resulting change that action brings upon the environment_. Inspired by this idea, authors developed a method for splitting the underlying video into “pre-condition” and “effect” states, representing portions of the video before and after the action takes places. Then, these groups of frames are passed through separate CNN architectures to extract feature representations for each. From here, an action is modeled as a linear transformation (i.e., a matrix multiplication) that transforms the pre-condition features into the effect features (i.e., this can be measured with the cosine distance between predicted and actual effect feature vectors). Interestingly, this entire siamese network architecture (including the action transformations) could be trained using an [expectation maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) procedure to achieve competitive performance on HAR benchmarks.

**Other stuff…** Some work on video understanding studied more efficient representations of 3D convolutions, finding that robust spatiotemporal relations could be learned by factoring 3D convolutions into separate 2D spatial and 1D temporal convolutions that are applied in sequence [14]. The resulting architecture contains significantly fewer parameters than corresponding 3D CNN architectures and, thus, can achieve better performance in the limited-data regime. Additionally, concurrent work went beyond the HAR problem domain and considered the problem of action detection [15], where actions must be both identified/classified and localized within the underlying video. By adopting a region proposal and feature extraction approach that utilizes early, two-stream architecture variants [1], impressive performance could be achieved on action detection benchmarks.

### Conclusions and Future Directions…

Though many approaches to video understanding were explored concurrently, the impressive performance of the two-stream approach led to popularization of the technique. Nonetheless, the two-stream architecture was still — at its core — dependent upon hand-crafted features that are extracted from the underlying video. In particular, it relied upon optical flow maps that were extracted from the underlying video and passed as input to the temporal stream. Although such features make minimal assumptions about the underlying video (i.e., just smoothness and continuity assumptions), this reliance upon hand-crafted optical flow features would be criticized by later work, leading to the development of more sophisticated architectural variants that will be discussed in the next post of this series.

Thank you so much for reading this post! I hope you found it helpful. If you have any feedback or concerns, feel free to comment on the post or reach out to me via [twitter](https://twitter.com/cwolferesearch). If you’d like to follow my future work, you can follow me on Medium or check out the content on my [personal website](https://wolfecameron.github.io/). This series of posts was completed as part of my background research as a research scientist at [Alegion](https://www.alegion.com/). If you enjoy this post, feel free to check out the company and any relevant, open positions — we are always looking to discuss with or hire motivated individuals that have an interest in deep learning-related topics!

_Bibliography_

[1][https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)

[2] [https://arxiv.org/abs/1406.2199](https://arxiv.org/abs/1406.2199)

[3] [https://arxiv.org/abs/1803.01164](https://arxiv.org/abs/1803.01164)

[4] [https://pubmed.ncbi.nlm.nih.gov/1374953/](https://pubmed.ncbi.nlm.nih.gov/1374953/)

[5] [https://arxiv.org/abs/1507.02159](https://arxiv.org/abs/1507.02159)

[6] [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

[7] [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

[8] [https://arxiv.org/abs/1604.06573](https://arxiv.org/abs/1604.06573)

[9] [https://arxiv.org/abs/1611.02155](https://arxiv.org/abs/1611.02155)

[10] [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

[11] [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

[12] [https://arxiv.org/abs/1412.0767](https://arxiv.org/abs/1412.0767)

[13] [https://arxiv.org/abs/1512.00795](https://arxiv.org/abs/1512.00795)

[14] [https://arxiv.org/abs/1510.00562](https://arxiv.org/abs/1510.00562)

[15] [https://arxiv.org/abs/1411.6031](https://arxiv.org/abs/1411.6031)