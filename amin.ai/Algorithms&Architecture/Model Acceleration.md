

amatically improves the efficiency of autoregressive decoding. It stores intermediate attention computations from previous decoding steps—specifically, the key and value tensors produced within the self-attention mechanism—so they do not need to be recomputed at every new generation step. This reduces both inference time and redundant memory access, making long-form generation feasible for LLMs.

##### The Problem: Quadratic Recomputation in Naive Generation

- During autoregressive generation, each new token depends on all previously generated tokens. In a **naive transformer implementation**, the model recomputes the key K and value V representations for **all tokens** in the sequence at every decoding step and across all layers. This quickly becomes computationally expensive, since the total cost per predicted token for a single attention head is:
    
    O(l⋅n⋅d2)
    
    - where:
        
        - n = number of tokens seen so far (sequence length)
        - l = number of layers (depth)
        - d = model (embedding) dimension
- Without caching, predicting each new token involves:
    
    1. Computing the key and value matrices for all past tokens and for every layer.
    2. Performing matrix multiplications of the form:
        
        K=XWK,V=XWV
        
        - where X is the layer input, and WK, WV are fixed weight matrices.

##### Why Naive Generation Fails

- KV caching fundamentally **solves the quadratic recomputation problem** that arises from this naive approach.
- Without a KV cache, generating even a 100-token response leads to massive redundant computation:
    
    - **Token 1:** compute attention over 1000 context tokens
    - **Token 2:** recompute attention over all 1001 tokens
    - **Token 100:** recompute attention over all 1099 tokens
- The total number of attention computations can be derived from the arithmetic sum of all attention lengths per decoding step:

∑t=1100(1000+t−1)=1000×100+100×992=55,000

- Here’s what each term means:
    
    - The **1000** represents the fixed-length context available before generation begins (e.g., the prompt).
    - The **(t − 1)** accounts for the number of **previously generated tokens** already added before generating token t. At step t, the model has already generated t−1 new tokens on top of the initial context, so it must now attend to 1000+(t−1) total tokens.
    - The summation over all 100 decoding steps gives the total number of attention operations across the full generation.
- Thus, to generate 100 tokens, the model performs approximately **55,000 redundant attention computations** — most of which are recomputations of previously calculated keys and values.
    
- The inefficiency is striking:
    
    > Without KV cache: 100 output tokens = ~55,000 attention operations With KV cache: 100 output tokens = 100 attention operations (≈550× reduction)
    
- This highlights the key trade-off: **KV caching exchanges memory usage for compute savings**. By storing previously computed keys and values, the model avoids redoing work it has already completed—unlocking massive gains in speed and scalability.
    

##### The Solution: Reusing Cached Representations

- The KV cache optimization addresses this problem by **reusing previously computed K and V** representations for all past tokens. Instead of recalculating them every time a new token is generated, the model simply:
    
    - Reuses cached K1:(n−1) and V1:(n−1),
    - Computes only the new kt and vt for the current token,
    - And appends them to the cache.
- This approach effectively removes redundant computation, changing the per-step cost from O(l⋅n⋅d2)toO(l⋅d2—an **n-times speedup** in the sequence dimension.
    
- The following figure ([source](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)) illustrates a typical self-attentioblock in transformers:
    

![Self-Attention Block](https://aman.ai/primers/ai/assets/model-acceleration/SA.jpg)

##### Why This Matters

- This improvement is especially critical for long sequences, where n can reach thousands or even millions of tokens. Without caching, latency would scale quadratically with sequence length, quickly becoming impractical. With KV caching, inference scales linearly, enabling efficient streaming and low-latency text generation for modern LLMs.

#### Structure and Size of the KV Cache

- The KV cache stores the key and value tensors for each transformer layer, attention head, the sample indices within each batch, and the token prefix length (i.e., the number of tokens already processed, including the prompt and any previously generated tokens, not just the immediate past token).
- Assuming a transformer with:
    
    - Sequence length so far: n
    - Number of layers: l
    - Number of attention heads per layer: h
    - Head dimension: dk
    - Batch size: b
- The KV cache for the above setup would consist of two main tensors per layer:
    
    1. A **key tensor** of shape (b,h,n,dk), which stores the projected keys K for all past tokens.
        
    2. A **value tensor** of shape (b,h,n,dk) which stores the projected values V for all past tokens.
        
- Since each layer requires its own copy of both (K and V) tensors, the total number of stored elements is:

Total elements=2⋅l⋅b⋅h⋅n⋅dk

- If we assume each element is stored in 16-bit floating point precision (`float16`), then the total KV cache size in bytes is:
    
    Size (bytes)=2⋅l⋅b⋅h⋅n⋅dk⋅2
    
    - where the final factor of 2 accounts for the 2 bytes per `float16` element.
- **Example:**
    
    - For a model with l=32 layers, h=32 heads, dk=128, b=1, and =1000:
    
    Size=2⋅32⋅1⋅32⋅1000⋅128⋅2=524,288,000 bytes (≈500 MB)
    
- This shows that the KV cache can become a significant memory consumer for long sequences, which is why optimizations such as quantization or chunked attention are often used in large language model inference.
    

#### Caching Self-Attention Values

- KV caching exploits two key properties:
    
    1. The model weights (WK and WV) are fixed during inference.
    2. The K and V representations for a given token depend on that token and all prior tokens (via its hidden state), but they do not depend on or change with any future tokens (i.e., they are **immutable** for for all subsequent decoding steps).
- Therefore, once we compute the K and V representations for a given (token, layer, head) tuple, we can store them and reuse them in all subsequent decoding steps.
    
- At decoding step t:
    
    - **Without caching**: recompute K1:n and V1:n from scratch for all t tokens.
    - **With caching**: reuse K1:(n−1) and V1:(n−1) compute only the new kt and vt and append them to the cache.
- The following figure ([source](https://huggingface.co/blog/not-lain/kv-caching)) illustrates the KV caching process, showing how only the new token’s K and V are computed while the rest are reused:
    

![KV Caching Illustration](https://aman.ai/primers/ai/assets/model-acceleration/KV.png)

- This optimization changes the cost from O(l⋅n⋅d2) to O(l⋅d2) per decoding step — an **n-times speedup** in the sequence dimension.
    
- The improvement is especially significant for long sequences, where n can reach thousands or even millions of tokens. By eliminating redundant attention computations, KV caching enables efficient, low-latency generation at scale.
    

###### Why Not Cache Prior Queries?

- Only the most recent query qt is used in the self-attention operation (which is recomputed at every step because it depends on the most recent token’s embedding), so caching prior queries (q1:(n−1)) offers no benefit. Put simply, only the query corresponding to the latest token in the sequence is needed for decoding.

#### Autoregressive Decoding Process with Caching

1. **Initial Sequence (Prefill Phase)**:
    
    - Given a prompt sequence S=[x1,x2,…,xn] the model computes K and V tensors for all prompt tokens in all layers and stores them in the KV cache.
        
    - This step still incurs the full cost O(l⋅n⋅d2) because we have no cached values yet.
        
    - After this prefill step, the model transitions to the _decode phase_, where we process one token per step.
        
2. **Predict Next Token (Decode Phase)**:
    
    - At decoding step n+1:
        
        - Compute the query vector qn+1=xn+1WQ for the new token.
            
        - Retrieve all previous keys and values from the cache:
            
        
        K1:n,V1:n
        
        - Compute the attention output for the new token using:
        
        Attention(qn+1,K1:n,V1:n)
        
3. **Update Cache**:
    
    - Compute the new key and value vectors for the current token:
        
        kn+1=xn+1WK,vn+1=xn+1WV
        
    - Append these to the KV cache so they can be reused in future decoding steps:
        
        Kcache←[K1:n,kn+1]
        
        Vcache←[V1:n,vn+1]
        
4. **Repeat**:
    
    - Continue until the end-of-sequence (EOS) token is generated or the maximum token limit is reached.

#### Implementation Details

##### Cache Tensor Shape

- Assuming:
    
    - Batch size B
    - Max sequence length n
    - Number of heads H
    - Head dimension dk
    - Number of layers l
- The KV cache is structured to store K and V for each (token,layer,head) tuple. For a **given layer** and **head**, the cache tensor shapes and sizes are:
    
    - **Key cache (per layer, per head):**
        
        K(l,h)cache∈ℝB×n×dk
        
        - **Key cache size (in bytes):**
        
        S(l,h)K=B×n×dk×sizeof(dtype)
        
    - **Value cache (per layer, per head):**
        
        V(l,h)cache∈ℝB×n×dk
        
        - **Value cache size (in bytes):**
        
        S(l,h)V=B×n×dk×sizeof(dtype)
        
    - **Total memory size per layer and head (in bytes):**
        
        Slayer, head=2×B×n×dk×sizeof(dtype)
        
        - The factor of 2 accounts for both the key and value tensors.
- When combining all layers and attention heads, the total KV cache represents the complete memory footprint required to store all key and value tensors across the entire model. The following equations describe the full model-wide KV Cache cache dimensions and their corresponding memory requirements:
    
    - **Key cache (all layers, all attention heads):**
        
        K(total)cache∈ℝB×l×H×n×dk
        
        - **Key cache size (in bytes):**
        
        S(total)K=B×l×H×n×dk×sizeof(dtype)
        
    - **Value cache (all layers, all attention heads):**
        
        V(total)cache∈ℝB×l×H×n×dk
        
        - **Value cache size (in bytes):**
        
        S(total)V=B×l×H×n×dk×sizeof(dtype)
        
    - **Total memory size across the entire model (in bytes):**
        
        Stotal=2×B×l×H×n×dk×sizeof(dtype)
        
- **Notes:**
    
    - The cache size scales linearly with B, l, H, n, and dk. During autoregressive generation, the model processes one token per step in the decode phase, only **one** new key and one new value vector are appended at each step for every layer and head. Consequently, as decoding progresses, the total cache grows linearly with the number of generated tokens—reflecting the incremental accumulation of key-value pairs over time.
    - In practice:
        - sizeof(dtype)=2 bytes for FP16/BF16 caches.
        - sizeof(dtype)=1 byte for INT8 caches.
    - Example: For a model with l=32, H=64, dk=128, B=8, and n=4096, the KV cache can easily consume tens of gigabytes of GPU memory. Efficient cache management (e.g., truncation, quantization, or offloading) is thus essential for real-world deployment.
    - Efficient memory layout is crucial — contiguous buffers enable fast appends and reduce memory copy overhead.

##### Prefill Phase

- When the prompt is first processed:
    
    - The model computes K and V for **all** prompt tokens in every layer, filling the cache.
    - This initial step has the same cost as the naive approach:
    
    O(l⋅n⋅d2)
    
- After this, we move into the decode phase, where caching delivers the performance benefits.
    

##### Updates to the KV Cache

- During autoregressive decoding, the K and V projections are cached for every processed token, across all layers and heads.
- Each time a new token is generated:
    
    1. The model computes kt and vt for that token in each layer.
    2. These vectors are appended to the existing Kcache and Vcache.
    3. The updated cache is then used to compute the attention output for the next token.

#### Latency Optimization/Savings

##### Projection Cost

- **Without caching**:
    
    - For a single head, at decoding step with sequence length n, the self-attention module recomputes K and V for all n tokens across all l layers. Put simply, each new generated token waits for full attention recomputation.
    - Computational cost per predicted token:
        
        O(l⋅n⋅d2)
        
- **With caching**:
    
    - Only the key and value for the **new** token are computed, while the rest are reused from the cache. Put simply, each token only computes new attention scores.
    - Computational cost per predicted token:
        
        O(l⋅d2)
        
- This represents an n-times speedup in the sequence dimension. For large n (e.g., thousands or millions of tokens), the cost reduction is dramatic.
    

##### Attention Score Computation

- **Without caching**:
    
    - At sequence length n, computing the attention scores requires multiplying the query for the new token with all n keys. This is done for every layer, so the attention score computation cost per predicted token is:
        
        O(l⋅n⋅d)
        
    - Because n increases with each generated token, the latency for this step grows linearly in n per token generation, but overall decoding (projection + attention) without caching still has quadratic growth in n.
        
- **With caching**:
    
    - Keys from all previous tokens are already stored. At sequence length n, we only compute the dot products between the new query and the cached keys:
        
        O(l⋅n⋅d)
        
    - The cost per token still grows linearly with n, but caching removes the quadratic growth that comes from recomputing keys and values for older tokens.
        

##### Total Complexity

- KV caching transforms overall decoding latency from quadratic in n to approximately linear in n, a major improvement for long-sequence generation. Specifically, KV caching changes the dominant scaling term from O(n2⋅d2) to O(n2⋅d) which, for typical transformer sizes, is a substantial improvement in long-sequence latency. This is mathematically represented below.
    
- **Without caching**:
    
    - Total cost per predicted token = projection cost + attention score computation:
        
        O(l⋅n⋅d2)+O(l⋅n⋅d)≈O(l⋅n⋅d2)
        
    - Over an entire sequence of length **n**, the total decoding cost is:
        
        O(l⋅n2⋅d2)
        
- **With caching**:
    
    - Total cost per predicted token = projection cost + attention score computation:
        
        O(l⋅d2)+O(l⋅n⋅d)≈O(l⋅n⋅d)
        
    - Over an entire sequence of length **n**, the total decoding cost is:
        
        O(l⋅n2⋅d)
        

#### Practical Deployment Considerations

#### Practical Deployment Considerations

##### Memory Management

- Managing KV caches efficiently is one of the main engineering challenges in large-scale transformer deployment. The cache for each sequence grows linearly with the number of processed tokens, since for every new token the model must store its key and value representations for each layer and attention head. Consequently, even moderate increases in context length can result in exponential GPU memory pressure when serving multiple concurrent requests.
    
- To mitigate this, systems adopt several strategies:
    
    - **Sliding Window Caching**: Instead of maintaining the entire attention history, servers often retain only the most recent N tokens per request. This sliding window allows for long-running conversations without exceeding memory budgets, at the cost of slightly reduced long-range recall. For instance, if the model supports 32k tokens but memory is constrained, the cache may only keep the last 8k–16k tokens.
        
    - **Cache Truncation and Compression**: For extreme cases, caches can be truncated or quantized. Truncation drops the oldest segments of the context when the memory budget is reached, while compression methods—like storing keys and values in lower precision (e.g., FP16 or INT8)—trade off a small amount of accuracy for substantial memory savings.
        
    - **Layer-Aware Cache Allocation**: Not all layers contribute equally to performance. Some deployment systems dynamically allocate higher precision or longer cache retention to the most attention-sensitive layers while reducing resource usage for others.
        
    - **Offloading to Host Memory**: For very long contexts or multi-turn conversations, GPU memory may not suffice. Systems can offload part of the cache to CPU memory or even NVMe-based memory pools, fetching it back as needed. However, this introduces latency trade-offs and requires careful memory pinning to minimize data transfer overheads.
        

##### Dynamic Batching

Dynamic batching is essential for maximizing GPU utilization in real-time inference scenarios. Since users issue requests of varying lengths and progress asynchronously through token generation, each request maintains an independent KV cache that grows at its own rate. A well-designed system must:

- Efficiently **group requests with similar decoding steps** to form micro-batches without breaking sequence dependencies.
- Maintain **per-request cache isolation**, ensuring that the correct KV history is retrieved during each attention computation.
- Implement **fast lookup and append mechanisms**, typically backed by memory pools or custom allocators, allowing concurrent cache updates without heavy synchronization locks.
- Use **streaming attention scheduling**: at each decoding step, the system identifies which requests are ready to decode and merges them temporarily into a batch. Once the next token is produced, each request’s cache is updated independently.

Systems such as vLLM and TensorRT-LLM provide specialized runtime schedulers that dynamically manage per-request caches while achieving near-optimal GPU occupancy. In such architectures, the ability to reuse KV states and batch across requests determines overall throughput.

##### Cache Parallelism

- In large-scale multi-GPU or distributed serving environments, KV caches themselves become distributed data structures. When the model’s layers or attention heads are sharded across devices, the corresponding keys and values must follow the same partitioning strategy. Typical configurations include:
    
    - **Tensor Parallelism**: Each GPU holds only a subset of the attention heads. During cross-device attention computations, the K and V tensors are exchanged or synchronized across GPUs via collective operations (e.g., all-gather). Efficient implementations overlap communication and computation to minimize latency.
        
    - **Pipeline Parallelism**: Layers are distributed across GPUs. Each GPU must maintain the KV cache for only the layers it owns. However, during forward passes, intermediate activations are streamed between devices. The system must ensure that caches align temporally across pipeline stages to preserve attention correctness.
        
    - **Model Parallel + Data Parallel Hybridization**: In highly scalable deployments, KV caches are both sharded (for model parallelism) and replicated (for data parallelism). Systems must handle synchronization and memory consistency between replicas, often through NCCL-based communication backends.
        
    - **Cross-Node Caching**: When models run across multiple nodes, caches may be stored in distributed shared memory or remote memory access (RDMA)-capable hardware, allowing direct GPU-to-GPU cache retrieval without CPU intervention.
        

##### Why You Can’t Always Cache Everything

- Memory growth in KV caching is linear in sequence length and proportional to the number of layers, heads, and hidden dimensions:
    
    - Memory scaling: For large models (tens of billions of parameters), a single sequence of 1000 tokens may consume roughly 1 GB of KV cache.
    - Batch size impact: The more concurrent sequences are cached, the fewer requests can fit into GPU memory, directly impacting throughput.
    - Context length: With ultra-long contexts (e.g., 100k tokens), a naive full cache could exceed 100 GB—far beyond the capacity of even high-end GPUs.

#### Multi-Head Attention and KV Cache

- In practice, self-attention is implemented with multiple attention heads, each operating in a subspace of the embedding dimension. For head h in {1,…,H}, we have:
    
    Q(h)=XWQ(h),K(h)=XWK(h),V(h)=XWV(h)
    
- The attention outputs from each head are concatenated:
    
    Q=concat(Q(1),Q(2),…,Q(H))
    
    - and similarly for K and V.
- **Caching in multi-head attention**:
    
    - The KV cache stores keys and values for every head and every layer.
    - Shape for the key and value cache:
    
    Kcache∈ℝB×H×n×dk
    
    Vcache∈ℝB×H×n×dk
    
    - **where**
        
        - B = batch size (number of sequences processed in parallel)
        - H = number of attention heads
        - n = sequence length (number of tokens stored in the cache)
        - dk = dimension of the key (and value) vectors per head
- **Performance implications**:
    
    - Since each head’s KV cache is independent, the caching logic operates head-wise, but the storage is typically implemented as a unified tensor for efficiency.
    - This unified tensor is arranged to be friendly to GPU tensor cores, enabling very fast read and write operations during decoding.
- While KV caching greatly reduces the sequence dimension cost, the **depth dimension** (number of layers l) is still a significant contributor to compute. This leads to the _KV Sharing_ idea, covered in detail in the section on [KV Sharing](https://aman.ai/primers/ai/model-acceleration/#kv-sharing) — reusing K and V representations across the last half (or fraction) of layers to further cut computation. KV sharing builds on KV caching, but attacks the problem from the layer/depth dimension rather than the token dimension.
    

#### Summary of KV Cache Benefits

- **Reduces repeated computation** by storing and reusing K, V tensors instead of recomputing them at every step.
- **Enables efficient decoding** in autoregressive generation by cutting per-step cost from O(l⋅n⋅d2) to O(l⋅d2) — an **n-times speedup** in the sequence dimension.
- **Optimized for hardware acceleration** via unified tensor layouts that are friendly to GPU tensor cores.
- **Scales well** to large models and long contexts, with latency growing linearly rather than quadratically with sequence length.
- **Maintains accuracy** because cached K and V are identical to recomputed values, given fixed weights.

#### KV Sharing

- KV caching, introduced in [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254) by Sun et al. (2024), optimizes the **sequence dimension** (n) cost, but the **depth dimension** (l) — the number of layers — still incurs full computation for each layer’s K and V.
- **KV Sharing** addresses this by reducing the cost of computing K and V along the depth dimension.
- The intuition behind why this can work comes from studies such as [Do Language Models Use Their Depth Efficiently?](https://arxiv.org/abs/2505.13898) by Csordás et al., which show empirically that in a deep transformer-like model, the last layers are correlated with each other. This means the final few layers are not necessarily adding much new information, but rather tweaking the output produced so far. This redundancy can potentially be exploited to save computation without significantly degrading model quality.

##### How KV Sharing Works

- The core idea: share **actual K and V representations** (not just weight matrices) across the last fraction of layers.
    
- For example, if we share across the last half of the layers (l2 layers):
    
    1. The final layer before the shared region computes K and V normally.
    2. All subsequent layers in the shared region reuse these K and V without recomputation, regardless of their inputs.
    3. Other parameters (e.g., WQ, MLP weights) remain distinct per layer.
- Mathematically:
    
    - Let Lshare be the index of the first shared layer.
    - For any layer j≥Lshare:
    
    K(j)=K(Lshare),V(j)=V(Lshare)
    
- The following figure ([source](https://arxiv.org/abs/2405.05254)) illustrates KV Sharing across the last half of the layers, showing how a single computed K and V set is reused instead of recalculated:
    

![KV Sharing Illustration](https://aman.ai/primers/ai/assets/model-acceleration/KVS.jpg)

##### FLOP Savings

- If the last lk layers share K and V, we avoid computing them in lk layers entirely.
- FLOP reduction: Savings=lkl=1k fraction of the total keys and values computation.
- Combined with KV caching:
    
    - KV caching cuts cost in n (sequence) dimension.
    - KV sharing cuts cost in l (layer) dimension.

##### Why KV Sharing Can Work

- Empirical studies referenced in the paper show that in deep transformer models, the last few layers often produce correlated outputs.
- This suggests that later layers are mostly fine-tuning rather than introducing fundamentally new information.
- Reusing K and V in these layers therefore has minimal impact on output quality while significantly reducing compute and memory usage.

##### Memory Benefits

- **No need to store keys and values** for the shared layers at all.
- Reduces memory footprint in both inference and training.
- Particularly valuable when serving long sequences, where cache size is dominated by B×H×n×dk×l scaling.

##### Deployment Notes

- KV sharing must be considered at **training time** for best results, since models not trained with this constraint may suffer quality drops if sharing is applied post hoc.
- Works alongside KV caching since KV sharing tackles **depth**, while KV caching tackles **sequence length**.


-------

### Model Quantization

- Model quantization is a technique used to reduce the precision of numerical values (typically weights and activations) in a neural network from high-precision formats like 32-bit floating point (`float32`) to lower-precision formats such as `int8`, `float8`, or even `int4`. This allows for faster inference, reduced memory usage, and lower power consumption, particularly on hardware that supports low-precision arithmetic.
- A detailed discourse on this topic is available in our [Model Compression](https://aman.ai/primers/ai/model-compression) primer.

#### Why Quantize?

- Quantization can lead to significant improvements in efficiency:
    
    - **Reduced Memory Footprint**: An `int8` model consumes 75% less memory than its `float32` counterpart.
    - **Faster Arithmetic**: Lower-precision operations (like `int8` or `int4` matmuls) are natively supported and highly optimized on modern accelerators (e.g., NVIDIA Tensor Cores, Intel AVX-512 VNNI).
    - **Lower Latency**: With less data to move and faster compute kernels, quantized models offer reduced end-to-end inference time.

#### Types of Quantization

##### Post-Training Quantization (PTQ)

- PTQ involves converting a pre-trained `float32` model to a lower-precision model without retraining. It works by calibrating the ranges of tensors using a small sample of data.
    
- **Key steps in PTQ:**
    
    - **Range Calibration**: Identify the min/max values of weights and activations from a calibration dataset.
        
    - **Scale and Zero-Point Calculation**: For each quantized tensor, calculate:
        
        q=round(rs)+z
        
        - where:
            
            - r is the real-valued number
            - s is the scale (i.e., step size)
            - z is the zero-point to preserve zero mapping in the quantized domain
            - q is the quantized value (e.g., 8-bit integer)
- **Weight and Activation Clipping**: Clip values to fit within the representable range of the target bit-width (e.g., [-128, 127] for signed `int8`).
    

##### Quantization-Aware Training (QAT)

- QAT simulates quantization during training. Fake quantization layers are added to mimic low-precision computation while maintaining gradients in high precision.
    
- **Advantages:**
    
    - More accurate than PTQ for sensitive models (e.g., GPT, BERT).
    - Allows the model to adapt to quantization errors during fine-tuning.
- **Implementation Details:**
    
    - Frameworks like PyTorch and TensorFlow include fake quantization modules (e.g., `torch.quantization.FakeQuantize`).
    - Quant-dequant pairs are inserted in the model graph to simulate the behavior of actual quantized operations.

#### Static vs. Dynamic Quantization

- **Static Quantization**: Activations are quantized ahead of time using calibration. Requires representative input data and is more performant but less flexible.
- **Dynamic Quantization**: Weights are quantized ahead of time, but activations are quantized at runtime based on actual values. More flexible and easier to integrate but slightly slower.

#### Quantization in Transformers

- In transformer models like GPT or BERT, quantization is applied to:
    
    - **Linear layers**: Including query, key, value, and output projections in attention layers.
    - **GEMM-heavy blocks**: MLP (feed-forward) layers.
    - **Embedding layers**: Often quantized with special handling to preserve lookup efficiency.
- **Special Considerations**:
    
    - LayerNorm and Softmax are sensitive to quantization and often kept in `float32`.
    - Attention scores may require FP16 or `float32` to avoid instability.
    - Mixed-precision quantization (e.g., `float8` weights with `int8` activations) is sometimes used.

#### Tooling and Frameworks

- **NVIDIA TensorRT / FasterTransformer**
- **Intel Neural Compressor (INC)**
- **PyTorch Quantization Toolkit**
- **ONNX Runtime Quantization**
- **BitsAndBytes (for 8-bit and 4-bit LLMs)**
    
- These tools offer end-to-end pipelines for quantizing, validating, and deploying models.


------

### Operator Fusion

- Operator fusion is an inference optimization technique that combines multiple adjacent operations in a neural network computation graph into a single composite operation. This is done to reduce overhead from memory reads/writes, kernel launches, and inter-operation communication, especially on GPU- or TPU-based systems.
    
- Fusion reduces latency and increases compute efficiency by keeping data in faster registers or shared memory, rather than flushing it out to slower global memory between every small operation.
    

#### Motivation

- Modern deep learning workloads often involve many small operations executed sequentially—e.g., matrix multiplications followed by bias addition, normalization, and non-linear activations:

x→Linear→AddBias→LayerNorm→ReLU

- Each of these operations might otherwise be implemented as a separate kernel. This leads to:
    
    - Increased kernel launch overhead.
    - Inefficient use of GPU parallelism.
    - Repeated memory access and latency.
    - Limited optimization opportunities for compilers.
- By fusing them, the computation becomes more compact, minimizing overhead and maximizing performance.
    

#### Common Fusion Patterns

- Some of the most commonly fused sequences in transformer inference include:
    
    - **GEMM + Bias Add + Activation**
        
        - Example: Y=ReLU(X@W+b)
        - Typically fused in MLP layers.
    - **Residual Add + LayerNorm + Dropout**
        
        - Used in transformer blocks.
    - **Query/Key/Value Linear Projections**
        
        - Three `Linear` ops fused into a single matmul followed by splitting heads.
    - **Softmax + Masking**
        
        - In attention, softmax is often fused with masking logic to avoid branch divergence on GPUs.

#### Fusion in Transformers

- In transformer architectures, operator fusion is especially valuable in:
    
    - **Multi-Head Attention Blocks**:
        
        - Combine Q/K/V projections and reshape + transpose logic into a single kernel.
        - Fuse attention score computation, masking, and softmax into one efficient operation.
    - **Feed-Forward Networks (FFNs)**:
        
        - Fuse two linear layers with intermediate activation (e.g., GELU or ReLU).

#### Implementation Details

- Fusion can be implemented in several ways:

##### Graph-Level Fusion (Ahead-of-Time)

- High-level compilers like XLA (for TensorFlow) or TorchScript (for PyTorch) can analyze the computational graph and fuse operations during compilation.
    
- Example in PyTorch:
    

![](https://aman.ai/images/copy.png)

`@torch.jit.script def fused_layer(x, w1, b1, w2, b2):     return F.relu(F.linear(x, w1, b1)) @ w2.T + b2`

- TorchScript may fuse `linear + relu` into a single kernel.

##### Kernel-Level Fusion (Runtime)

- Frameworks like NVIDIA’s TensorRT and FasterTransformer include hand-written CUDA kernels that combine multiple operations (e.g., QKV projection + transpose + scale + matmul) in one pass.
    
- Example: A fused transformer kernel might compute:
    

![](https://aman.ai/images/copy.png)

`qkv = fused_linear_bias_act(x);  // one call q, k, v = split_heads(qkv);      // internal fused transpose and reshape`

- This reduces global memory traffic and utilizes registers/shared memory for intermediate results.

##### 3. Custom Kernel Generation

- Libraries like TVM or Triton enable defining custom fused kernels in a hardware-optimized DSL. These can be compiled just-in-time for maximum throughput.
    
- Example in Triton:
    

![](https://aman.ai/images/copy.png)

`@triton.jit def fused_gemm_relu(...):     # Define fused matmul + bias + relu logic using GPU thread blocks`

#### Performance Impact

Operator fusion can lead to:

- **30–50% improvement in latency** for attention blocks.
- **Higher hardware utilization**, especially on GPUs with tensor cores or vectorized ALUs.
- **Reduced memory bandwidth pressure**, which is often the bottleneck in LLM inference.

#### Tooling and Ecosystem

- **TensorRT**: Extensive fusion for transformer blocks.
- **FasterTransformer**: Fused QKV and FFN kernels.
- **ONNX Runtime with Graph Optimizer**: Automatic fusion passes.
- **TorchScript + FBGEMM**: Fusion of linear + activation ops.
- **TVM / Triton**: Customizable and tunable fusion kernels.


-------

### Speculative Decoding

- Speculative decoding is an inference-time optimization technique designed to reduce the latency of autoregressive sequence generation in large language models (LLMs). Instead of generating one token at a time using the full, expensive model, speculative decoding uses a smaller, faster “draft” model to guess multiple tokens in parallel, then validates these guesses with the full “target” model. If the guesses are correct, they are accepted as part of the output. Otherwise, they are partially or fully discarded and recomputed.
    
- This method maintains the output quality of the original model while significantly improving throughput.
    

#### Motivation

- Autoregressive decoding is inherently sequential. In a naive setup, the model generates one token, then feeds it back as input to generate the next. This sequential loop introduces latency and becomes a bottleneck during long-form generation.
    
- Let:
    
    - f be the full model (large, accurate but slow)
    - g be the draft model (smaller, less accurate but fast)
- Naively, generation requires T forward passes of f for a sequence of T tokens. Speculative decoding aims to reduce the number of times f is called.
    

#### Basic Algorithm

1. **Initialize Context**: Use a prompt or previous tokens x.
2. **Draft Generation**: Use the draft model g to generate a sequence of k speculative tokens:
    
    y1,y2,...,yk=g(x)
    
3. **Validation**: Use the full model f to compute the log-probabilities pf(yt‖x,y1,...,yn−1).
4. **Accept or Reject Tokens**:
    
    - Accept as many tokens as f agrees with (within a confidence threshold or by matching top-1 outputs).
    - Rewind to the last agreed-upon token and resume with the draft model from there.

#### Pseudocode

![](https://aman.ai/images/copy.png)

`x = initial_prompt while not done:     draft_tokens = g.generate_next_k(x)     probs_f = f.get_probs(x + draft_tokens)     accepted_prefix = match(draft_tokens, probs_f)     x = x + accepted_prefix`

#### Key Parameters

- **Draft Model Quality**: Must be fast enough to justify speculative overhead but good enough to match the full model reasonably often.
- **Block Size k**: Number of speculative tokens generated per iteration. Larger blocks = fewer full model calls, but higher risk of rejection.
- **Matching Strategy**: Usually uses top-1 match or a log-prob threshold.

#### Mathematical View

- Let the probability of accepting each token be α. Then the expected number of full-model calls is:

𝔼[full passes]≈Tk⋅α

- If α≈0.7 and k=4, we reduce full-model calls by nearly 3×.

#### Implementation Details

- **Parallel Calls**: f can validate all k tokens in one forward pass by using cached KV states and batched logits.
- **KV Cache Management**: Efficient speculative decoding updates the cache only after validation.
- **Multimodel Serving**: Systems like NVIDIA’s FasterTransformer or Hugging Face’s `transformers` can host both f and g concurrently with shared memory or GPU residency.

#### Notable Variants

- **Medusa** (Meta): Uses a tree-structured decoder to validate multiple candidates at once.
- **FastRAG**: Combines speculative decoding with retrieval-based models.
- **Draft & Verify** (Google): A formalized framework for plug-and-play speculative decoding with checkpointing.

#### Benefits

- **Latency Reduction**: 2×–4× speedup in decoding for long sequences.
- **Full-Model Accuracy**: Final output matches the output of the full model f, so there’s no accuracy loss.
- **Compatibility**: Can be layered on top of existing decoding strategies (e.g., greedy, top-k, nucleus).

#### Limitations

- Requires additional memory and compute for the draft model.
- Effectiveness depends on alignment between the draft and full model distributions.
- Complex cache management and integration overhead.


---------

### FlashAttention and Efficient Attention Kernels

- In transformer models, self-attention is a core operation that enables the model to learn relationships between tokens. However, traditional attention implementations scale poorly with sequence length due to quadratic memory and compute complexity. **FlashAttention** and other efficient attention kernels address these bottlenecks by optimizing the attention computation to reduce memory overhead and improve performance.

#### Motivation

- The standard attention computation involves the following operations for a sequence of length L and hidden dimension d:
    
    Attention(Q,K,V)=softmax(QKTd‾‾√)V
    
- This requires:
    
    - Computing a full L×L attention matrix (expensive for long sequences).
    - Storing intermediate results like logits and softmax scores in global memory.
    - Limited reuse of on-chip memory (registers, shared memory), resulting in bandwidth-bound performance.
- FlashAttention addresses these inefficiencies by restructuring the attention algorithm to use memory-efficient block-wise computation.
    

#### FlashAttention: Key Concepts

- Originally proposed in Dao et al., 2022 in [FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness](https://arxiv.org/abs/2205.14135), FlashAttention is a fused, tiled implementation of scaled dot-product attention that:
    
    - **Eliminates materialization of the full attention matrix**: Avoids creating and storing the entire L×L attention score matrix in GPU memory. Instead, computes small blocks of logits on-chip, applies masking and softmax immediately, and discards them, drastically reducing memory usage for long sequences.
        
    - **Uses tiling to partition queries, keys, and values into small blocks that fit in GPU shared memory**: Splits Q, K, and V into manageable tiles (e.g., 64×64) that can be loaded into fast on-chip shared memory or registers. This improves memory locality, reduces global memory reads/writes, and allows the GPU to reuse loaded data for multiple computations within the block.
        
    - **Fuses softmax, scaling, masking, and matmul into a single kernel**: Combines these operations into one GPU kernel to avoid storing intermediate results in memory. By performing scaling, masking, softmax computation, and the weighted sum with V in a single pass, FlashAttention reduces memory bandwidth usage and improves computational efficiency.
        

##### High-Level Algorithm

1. Load a block of queries Qi and keys Kj into shared memory.
2. Compute attention logits QiKTjd√ for the block.
3. Apply mask and softmax **in-place**, updating the running sum of exponents and maximums for numerical stability.
4. Accumulate partial outputs Ai,j=softmax(QiKTj/d‾‾√)Vj without storing intermediate attention matrices.
5. Repeat across blocks until the full result is computed.

##### Numerical Stability

- To avoid numerical overflow when computing softmax in a block-by-block fashion, FlashAttention keeps running statistics for each query row:
    
    - mi=maxjzij — the maximum logit value seen so far for that row, used to shift logits and prevent large exponentials.
    - si=∑jexp(zij−mi) — the running sum of the shifted exponentials, which forms the softmax denominator.
- As new blocks are processed, these values are updated using associative operations that merge current and previous block statistics without loss of precision. This ensures the final softmax is mathematically equivalent to computing it on the full L×L matrix, but without ever storing that matrix.
    

#### Implementation Details

- Written as a custom CUDA kernel.
- Uses **shared memory** to hold Q/K/V tiles and compute locally.
- Optimized to run in **mixed precision** (e.g., FP16 or BF16) for speed and memory efficiency.
- Compatible with dropout, masking, and rotary embeddings.

#### FlashAttention-2 Improvements

- Adds support for **non-causal attention**, **variable-length sequences**, and better **warp-level parallelism**.
- Removes redundant memory loads through more aggressive caching and loop unrolling.
- Enables **backward pass efficiency**, making it useful not only for inference but also for training.

#### Other Efficient Kernels

- **xFormers** (Meta): Modular attention implementations that support Flash, sparse, and memory-efficient variants.
- **Triton-based Attention**: Enables easy definition of fused attention kernels using Triton’s GPU DSL.
- **PagedAttention (vLLM)**: Optimizes KV cache access for batch inference, reducing memory fragmentation and improving latency.

#### Performance Gains

- FlashAttention reduces attention memory complexity from:
    
    - **(L2)** to **(L)** for memory consumption.
    - Achieves **1.7–2.7× speedup** on A100 GPUs for long sequence lengths (> 1k tokens).
    - Maintains exact attention output (within floating-point precision), unlike approximate methods.

#### Use in Inference

- FlashAttention is especially beneficial for:
    
    - Long-context models (e.g., 4k to 128k tokens).
    - Multi-head attention, where per-head memory use adds up quickly.
    - Deployment on GPUs with large shared memory (e.g., NVIDIA A100, H100).

#### Integration

- Supported in:
    
    - **Hugging Face Transformers** via `use_flash_attention_2=True`
    - **PyTorch** through custom CUDA extensions or Triton kernels
    - **DeepSpeed**, **FasterTransformer**, and **xFormers**


----------

### Batching, Sequence Packing, and Prefilling

- **Batching** and **prefilling** are inference-time optimization techniques that improve efficiency and throughput by better utilizing hardware and avoiding redundant computations. These are especially critical when serving LLMs in real-time or at high concurrency.

#### Batching

- Batching refers to the process of grouping multiple inference requests into a single forward pass through the model. This increases hardware utilization, amortizes overhead, and reduces latency per request (on average), particularly on GPUs that are optimized for matrix-heavy workloads.

##### Motivation

- Without batching, each request results in an under-utilized forward pass:
    
    - Small input tensor → Poor occupancy/utilization of GPU cores
    - High overhead per kernel launch
    - Wasted memory bandwidth
- Batching solves this by aligning multiple requests into a tensor of shape:
    
    Batch Tensor: (B,L,d)
    
    - where:
        - B is batch size
        - L is sequence length
        - d is hidden dimension

##### Types of Batching

1. **Static Batching**: Requests are grouped together at fixed time intervals. Simple but less flexible.
2. **Dynamic Batching**: Requests are buffered and grouped at runtime based on heuristics like request arrival time, sequence length, or prompt similarity.
3. **Token-Level Batching**: Pioneered by vLLM, this groups sequences by shared decoding step instead of sequence. Supports long-running generation jobs without blocking new ones.
4. **Asynchronous Batching**: Uses request queues and a scheduler to decide when to batch based on hardware load.

##### Padding and Masking

- Since sequences may vary in length, shorter ones are padded and masked accordingly. Padding increases memory cost but enables unified matrix operations.
    
- Example:
    
    - Sequence A: `[Hello, how, are, you]` → length 4
    - Sequence B: `[Hi]` → length 1
    - Batched input: `[[Hello, how, are, you], [Hi, PAD, PAD, PAD]]`

##### Performance Benefits

- Higher throughput: GPUs can process large matrices in parallel.
- Lower kernel launch overhead.
- Amortized use of KV cache and memory bandwidth.

#### Sequence Packing

- **Sequence packing** is an optimization that reduces padding overhead when batching variable-length sequences. Instead of padding all sequences in a batch to the maximum length, multiple shorter sequences are concatenated into a single continuous sequence within the same batch element.
    
- This approach stores and processes only actual tokens, using an **attention mask** to ensure tokens from different original sequences do not attend to each other.
    

##### Example

- Without packing:
    
    ![](https://aman.ai/images/copy.png)
    
    `[Hello, how, are, you, PAD, PAD, PAD] [Hi, there, PAD, PAD, PAD, PAD, PAD]`
    
    - **Memory usage:** proportional to 7 tokens per sequence (including pads).
- With packing:
    
    ![](https://aman.ai/images/copy.png)
    
    `[Hello, how, are, you, Hi, there]`
    
    - Plus a mask to block attention between `you` and `Hi`.

##### Benefits

- **Reduced memory footprint** — fewer padding tokens stored and processed.
- **Better hardware utilization** — higher effective sequence density in each batch.
- **Lower latency for mixed-length workloads** — especially beneficial when many short sequences are served alongside long ones.

##### Trade-offs

- Slight overhead in constructing and applying more complex attention masks.
- May require specialized batching logic and kernel support for optimal performance.

#### Prefilling

- Prefilling refers to the one-time computation of model activations (primarily KV cache) for the prompt or context tokens before autoregressive decoding begins.

##### Motivation

- Transformer inference separates the process into:
    
    1. **Prompt Phase (Prefill)**: Process entire prompt to initialize the KV cache.
    2. **Generation Phase (Decode)**: Generate one token at a time using cached keys and values.
- The prompt phase is significantly more expensive because it processes multiple tokens without caching, while the decode phase uses KV caching for each new token.
    

##### Prefilling Logic

- Given a prompt of n tokens:
    
    - The model performs a full forward pass to compute attention outputs for all n positions.
    - During this, it initializes the KV cache tensors:
        
        K1:n,V1:n
        
    - These are used in all subsequent generation steps to avoid recomputation.

##### Optimizations

- **Fused Prefill Kernels**: Libraries like FasterTransformer use specialized kernels to batch and prefill KV caches in a single efficient pass.
- **Prompt Sharing**: If multiple requests use the same prompt (e.g., “You are a helpful assistant…”), cache the prefilled results and reuse them across requests.
- **Layer-Wise Streaming**: Some implementations stream KV cache population layer-by-layer to overlap computation and memory operations.

##### Real-World Use

- In production systems:
    
    - Prompt prefill is often the **dominant source of latency**, especially with long prompts (e.g., 1k+ tokens).
    - Prefilling is **not cacheable** unless the prompt is reused. That’s where [prompt caching](https://aman.ai/primers/ai/model-acceleration/#prompt-caching) comes in.
    - Systems may delay decoding until all requests in a batch complete their prefill phase.

##### Performance Benefits

- Avoids redundant computation across decoding steps.
- Enables efficient reuse of memory and attention context.
- Critical for long-context inference and multi-user serving.


-----


### Prompt Caching

- Prompt caching is an inference-time optimization that reuses the computed key-value (KV) attention states for frequently occurring or repeated prompt tokens. It eliminates the need to recompute the prefill phase of autoregressive decoding, which is typically the most computationally expensive part of the inference pipeline for long prompts.
    
- This technique is especially effective in systems with repeated system messages, user templates, or static few-shot examples.
    

#### Motivation

- During autoregressive generation, transformer models process the prompt (or context) once to initialize the attention cache. For a prompt of length n, this involves a full forward pass through all transformer layers to compute the KV tensors:

K(l)1:n,V(l)1:nfor all layers l

- This prefill step is expensive and must be repeated for every new request — even if the prompt is the same.
    
- **Observation**: Many applications use identical or highly similar prompts repeatedly. For example:
    
    - Instructional prompts like: “You are a helpful assistant.”
    - Few-shot templates in customer support bots.
    - System prompts in chat APIs.
- Prompt caching avoids repeated prefill for these common contexts.
    

#### Basic Mechanism

1. **Cache Initialization**:
    
    - Compute and store KV tensors for a given prompt:
        
        KVprompt=f(prompt)
        
    - Store in memory or disk with a unique key (e.g., hash of token IDs).
        
2. **Cache Lookup**:
    
    - For each incoming request, compute a cache key from its prompt.
    - If a match is found, retrieve KV tensors instead of recomputing them.
3. **Continue Decoding**:
    
    - Begin token-by-token generation using the cached KV state:
        
        Generate(xn+1∣KVprompt)
        

#### Implementation Details

##### Cache Granularity

- **Full Prompt Cache**: Caches the entire KV cache of a prompt. Simple and effective but can use a lot of memory.
- **Prefix Sharing**: If prompts differ by suffix (e.g., `Prompt A + User 1` and `Prompt A + User 2`), share the KV prefix and compute only the delta.
- **Subgraph Caching**: In more advanced systems, only the first few layers or tokens may be cached.

##### Cache Storage

- **In-Memory KV Cache**: For maximum performance, use GPU or CPU memory with LRU eviction.
- **On-Disk Cache**: Slower but scalable for cold-start scenarios.
- **Keyed by Hash**: Tokenized input is hashed using SHA or CRC to form a cache key. Some systems normalize prompts before hashing.

##### Integration with Serving Systems

- Requires cache-aware batch scheduling.
- Works best when integrated with dynamic batching and token-level schedulers (e.g., vLLM).
- May include cache warming: preloading common prompts at system startup.

#### Performance Impact

- Let:
    
    - Tp = time to prefill prompt
    - Td = time per token for decode
- For long prompts (e.g., 1000+ tokens), Tp≫Td, so caching the prefill can save **80–95%** of per-request compute for repeated prompts.
    

#### Applications

- **Chat APIs**: System messages or few-shot exemplars remain fixed across turns.
- **Agent Frameworks**: Tools like LangChain often replay the same template structure.
- **Batch Inference**: Multi-user prompts often share context headers (e.g., “Summarize the following…”).

#### Limitations

- Prompt cache is only useful for **identical** or **prefix-matching** prompts.
- Memory usage scales with prompt length and cache size.
- May add overhead for hash computation or miss rate handling.
- Not helpful for fully dynamic, unique user inputs.

----

### Early Exit and Token Pruning

- **Early exit** and **token pruning** are inference-time optimizations designed to reduce computation in large transformer models by selectively skipping or trimming parts of the computation graph that have diminishing contribution to the final output. These methods exploit redundancy in token representations and layer-wise stability in transformer models.
    
- Both techniques aim to speed up inference without significantly affecting model output quality, making them valuable in latency-sensitive or resource-constrained applications.
    

#### Early Exit

- **Early exit** allows the model to stop processing certain tokens or even entire sequences at intermediate layers if the model’s confidence in the prediction is already high.

##### Motivation

- Transformer models use a fixed number of layers (e.g., 24 or 96), but not all tokens require the full depth to make a confident prediction. For example, easily classifiable tokens (like punctuation or common stopwords) may converge earlier than rare or ambiguous tokens.

##### Mechanism

- At each transformer layer l, evaluate a confidence metric based on the current token representation:

1. **Entropy-Based Confidence**:
    
    - Compute the softmax output p(l) from the current logits.
    - Compute entropy:
        
        H(p(l))=−∑ip(l)ilogp(l)i
        
    - If entropy < threshold, consider the prediction confident enough to exit.
2. **Cosine Similarity to Previous Layer**:
    
    - If representation at layer l is similar to layer l−1, the token may have converged.
3. **Learned Gates**:
    
    - Add a small classification head to each layer to learn exit decisions during training (as in **BranchyNet** or **LayerDrop** approaches).

##### Implementation

- Models like **BERT with Early Exit** (DEEPL) implement classifier heads at multiple depths.
- Hugging Face `transformers` has prototype support for early exit in sequence classification.
- Requires threshold tuning to balance accuracy and latency.

##### Benefits

- Reduces average inference depth (e.g., from 24 layers to 12–16 for many tokens).
- Saves computation for simpler or high-confidence examples.
- Ideal for classification or QA tasks where tokenwise prediction is not necessary.

##### Limitations

- Adds overhead from confidence computation at intermediate layers.
- Not widely adopted in generation tasks due to sequential dependencies between tokens.

#### Token Pruning

- **Token pruning** reduces the number of tokens that are propagated through the deeper layers of a transformer by identifying and removing tokens with low contextual importance.

##### Motivation

- In many attention-based computations, some tokens contribute very little to the output. For example, padding tokens or tokens with low attention weights to the rest of the sequence.
    
- Pruning these tokens saves compute in later layers, especially in long-context models or batch scenarios.
    

##### Mechanism

1. **Attention-Based Pruning**:
    
    - Compute the **attention score variance** or **total attention mass** a token receives:
        
        αi=∑jAttention(xi,xj)
        
    - Prune tokens with low total attention received or given.
        
2. **Top-k Token Selection**:
    
    - Keep only the top-k most important tokens per head or per sequence based on learned importance scores.
3. **Dynamic Thresholding**:
    
    - Use learned or rule-based thresholds to drop tokens whose impact is below a tunable cutoff.
4. **Progressive Pruning**:
    
    - Start with full tokens, and prune more aggressively as layers go deeper.

##### Implementation

- Typically done at attention module boundaries.
- Can be combined with sparse attention mechanisms.
- Token indices need to be tracked to reconstruct output or map back to the original sequence.

##### Benefits

- Reduces computation in deeper layers, especially for long sequences.
- Improves throughput with minimal impact on quality in summarization, QA, and retrieval tasks.
- Can be applied during training for alignment with inference.

##### Limitations

- May degrade quality if pruning is too aggressive or incorrectly calibrated.
- Requires complex index tracking and masking logic.
- Harder to apply in autoregressive settings where all tokens are sequentially dependent.

#### Tools and Research

- **DeLighT**, **LayerDrop**, and **EarlyBERT** for early exit variants.
- **SparseFormer**, **Synthesizer**, and **Longformer** introduce related token reduction ideas.
- Hugging Face and NVIDIA’s Megatron support token pruning hooks in research branches.


-------

### Hardware-Aware Scheduling

- **Hardware-aware scheduling** refers to a set of optimization strategies that tailor the execution of neural network inference to the specific architecture and performance characteristics of the underlying hardware—such as GPUs, TPUs, or specialized accelerators. These optimizations aim to improve compute throughput, memory utilization, and latency by orchestrating how and when operations are executed.
    
- This is especially important for transformer inference, where workloads are large, heterogeneous (e.g., KV cache lookups, matrix multiplies, normalization), and sensitive to memory bandwidth and parallelism.
    

#### Motivation

- Transformer inference involves many stages of computation and memory access:
    
    - Matrix multiplications (GEMMs) in attention and feed-forward blocks.
    - Data movement between layers and devices.
    - KV cache management and resizing.
    - Softmax, activation, and normalization operations.
- Without careful scheduling, bottlenecks can emerge due to:
    
    - Underutilized compute units (e.g., Tensor Cores).
    - Memory stalls and cache thrashing.
    - Synchronization overhead between layers or streams.
- Hardware-aware scheduling optimizes these execution flows to keep the pipeline full and latency low.
    

#### Core Techniques

##### Stream Parallelism

- Modern GPUs support multiple concurrent execution streams (e.g., via CUDA). In transformer inference:
    
    - Use separate CUDA streams for different model stages (e.g., one for KV cache update, one for GEMM).
    - Overlap memory copies (e.g., `cudaMemcpyAsync`) with compute to hide latency.
- **Example**:
    
    ![](https://aman.ai/images/copy.png)
    
     `cudaMemcpyAsync(..., stream1);  cublasGemmEx(..., stream2);  // runs concurrently with stream1`
    

##### Tensor Core Utilization

- Tensor cores are specialized units in NVIDIA GPUs for low-precision matrix ops (e.g., `float16`, `bfloat16`, `int8`). To maximize their usage:
    
    - Ensure all matrix multiplications are aligned to multiple-of-8 dimensions.
    - Use fused kernels to eliminate intermediate `float32` conversions.
    - Prefer mixed-precision pipelines (AMP / `float16`) for higher throughput.
- Libraries like **cuBLAS**, **FlashAttention**, and **TensorRT** handle these optimizations automatically when configured correctly.
    

##### Operator Placement and Reordering

- Efficient inference scheduling may involve reordering or co-locating operations based on:
    
    - **Memory locality:** Fuse or group operations that share data.
    - **Execution time:** Prioritize long-running ops earlier in the pipeline.
    - **Device affinity:** Keep frequently accessed data on the same GPU or chip.
- **Example**: Run attention blocks first in multi-layer transformer if they dominate compute time, allowing FFNs to be prefetched concurrently.
    

##### KV Cache Management

- Efficient KV cache handling is essential in decoder models:
    
    - **Paged KV Cache**: Used in systems like vLLM, stores KV in contiguous memory pages and allows random-access updates.
    - **Memory Pools**: Preallocate KV buffers for each request and reuse them to avoid memory fragmentation.
    - **Lazy Allocation**: Delay cache instantiation until first generation step to save memory for short prompts.

##### Pipeline and Model Parallelism

- In large-model deployments:
    
    - **Pipeline Parallelism**: Distribute transformer layers across devices. Stage execution overlaps compute and communication.
    - **Tensor Parallelism**: Split individual tensor dimensions (e.g., weights) across devices for large GEMMs.
- Combined, these allow serving models with billions of parameters across multiple GPUs efficiently.
    

##### Custom Kernel Scheduling

- Frameworks like Triton and TVM allow defining and tuning custom kernels:
    
    - Auto-tune tiling sizes and shared memory usage.
    - Schedule GPU threads based on warp/block-level parallelism.
    - Implement custom token-wise or layer-wise scheduling logic.

##### Cache and Memory Prefetching

- Use `__prefetch` instructions or async loads to bring data into shared memory before it is needed.
- Overlap KV fetches with matmul execution to hide memory latency.

#### Deployment-Aware Strategies

- **Load Balancing**: Use dynamic batching queues with GPU-aware request routing (e.g., based on latency or memory pressure).
- **Thread Affinity**: Bind computation to specific CPU cores or NUMA zones in CPU-bound systems.
- **Execution Profiling**: Use profilers like NVIDIA Nsight Systems or PyTorch Profiler to tune for bottlenecks.

#### Ecosystem Support

- **NVIDIA TensorRT** and **FasterTransformer**: Hardware-aware fused kernels and scheduling policies.
- **ONNX Runtime (ORT)**: Execution providers tuned for different hardware (CUDA, DirectML, TensorRT).
- **DeepSpeed**, **vLLM**, **Triton**, and **TVM**: Offer fine-grained control over scheduling and memory layout.

#### Performance Impact

- Hardware-aware scheduling can yield:
    
    - **1.5×–4× speedup** over naive scheduling for long sequences or large batches.
    - Better **multi-GPU scaling** for high-throughput inference.
    - Lower **latency variability** in real-time serving environments.

------

### Comparative Analysis

|**Technique**|**Purpose**|**Key Benefits**|**Primary Use Cases**|**Implementation Notes**|
|---|---|---|---|---|
|KV Caching|Reuse attention keys/values from previous tokens|Reduces per-token latency after first step|Autoregressive decoding (GPT, LLaMA)|Requires careful cache management; starts from second token onward|
|Model Quantization|Use lower-precision weights/activations|Reduces memory and compute cost|Edge inference, high-throughput serving|`int8`/PTQ for speed; QAT for better accuracy; needs hardware with quantization support|
|Operator Fusion|Combine adjacent ops into single kernel|Reduces memory access and kernel launch overhead|Attention blocks, FFNs, LayerNorm + activation|Use graph compilers (XLA, TorchScript), or fused CUDA kernels (TensorRT, FasterTransformer)|
|Speculative Decoding|Use draft model to guess multiple tokens|Reduces number of full-model forward passes|Long-form generation, chatbots|Needs a lightweight auxiliary model; uses top-1 match or log-prob threshold for validation|
|FlashAttention & Kernels|Memory-efficient attention computation|Reduces memory usage and boosts speed|Long-sequence LLMs, multi-head attention|Implemented with CUDA (FlashAttention), or Triton/xFormers; avoids storing full attention matrix|
|Batching|Process multiple requests together|Increases throughput and GPU utilization|High-concurrency inference (API servers, batch jobs)|Dynamic and token-level batching supported in vLLM, DeepSpeed, TensorRT|
|Prefilling|Precompute KV cache from prompt tokens|Avoids recomputation in autoregressive models|Chat and generation tasks with long prompts|Often paired with batching; prompt KV cache initialized before decoding begins|
|Prompt Caching|Cache KV states of repeated prompts|Saves time and compute on repeated static contexts|Chat APIs, few-shot prompt templates|Requires hashing/tokenizing prompt and storing cache; memory usage grows with cache diversity|
|Early Exit|Stop processing tokens/layers early based on confidence|Reduces per-token compute in deep models|Classification, QA tasks|Needs entropy or learned gating logic; difficult to apply in token-dependent generation|
|Token Pruning|Discard low-importance tokens during inference|Reduces sequence length in deeper layers|Long-sequence summarization, QA|Attention-based importance scoring; careful masking and index tracking required|
|Hardware-Aware Scheduling|Optimize kernel execution for specific hardware|Maximizes throughput and minimizes latency|All transformer-based workloads|Includes stream parallelism, memory prefetch, cache layout, tensor core tuning, and multi-GPU distribution|

## References

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)
- [Gaurav’s Blog – Efficient AI: KV Caching and KV Sharing](https://blog.gaurav.ai/2025/08/05/kv-caching-kv-sharing/)
- [Let’s build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)


--------


# Primers • FlashAttention

## Motivation & Background

- Introduced in [FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness](https://arxiv.org/abs/2205.14135) by Dao et al. (2022), FlashAttention aims to dramatically speed up Transformer-style attention on GPUs while simultaneously reducing memory usage.
    
- Standard self-attention scales poorly with sequence length N, because it computes the full N×N attention matrix and performs O(N2⋅d) operations and memory storage. Especially on long-context models, both compute and memory usage balloon. Existing approximate methods (e.g., Linformer, Performer) often sacrifice accuracy or fail to deliver wall-clock improvements in practice due to GPU inefficiencies.
    

> FlashAttention’s core insight is **IO-awareness**: recognizing that the primary performance bottleneck on GPUs is not floating-point operations (FLOPs), but data movement between high-bandwidth memory (HBM) and the on-chip cache (SRAM/registers). Rather than optimizing just the computation, FlashAttention **restructures the memory access pattern**. It tiles the attention computation into blocks that fit entirely in SRAM and processes them sequentially, drastically reducing expensive off-chip memory traffic. Crucially, it recomputes certain intermediate values (like softmax normalization constants) rather than storing them, which is cheaper than reading from HBM.

- The following figure ([source](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)) shows a comparison between standard attention and FlashAttention, showing how FlashAttention reduces memory reads/writes by fusing operations into fewer memory transactions.

![](https://aman.ai/primers/ai/assets/flashattention/flash-attn.jpg)

- This design leads to **exact attention results** (unlike approximations) while achieving **linear memory growth** in N, thanks to never materializing the full attention matrix. The implementation uses **kernel fusion** to combine operations like QKᵀ matmul, masking, softmax, and dropout into a single CUDA kernel, minimizing inter-kernel launch overhead and avoiding unnecessary memory round trips.
    
- Key benefits documented:
    - Up to 3× speedup on GPT‑2 (seq = 1K),
    - 15% end‑to‑end speedup on BERT‑large (seq=512) compared to MLPerf baselines,
    - Ability to handle much longer contexts (1K–64K) with viable accuracy gains.
- Mathematically, given queries Q∈ℝN×d, keys K and values V, FlashAttention splits into tile blocks of size B and loops:

O=softmax(QKTd‾‾√)V

- … but never materializes the full N×N logits matrix. Instead it processes blocks of K,V, accumulates partial results, and recomputes necessary maxima for numerically stable softmax. The I/O complexity is shown to be optimal for typical on‑chip cache sizes within constant factors.
- This background establishes the rationale: attention is memory‑bound; FlashAttention removes the bound by reordering computation; yields real speed and memory improvements with no approximation.


-----------


