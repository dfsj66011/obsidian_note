
发表时间：2025.02.19
建议阅读时长：2-4 天
作者：Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf

[5D parallelism in a nutshell](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#5d_parallelism_in_a_nutshell)

[Finding the Best Training Configuration](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#finding_the_best_training_configuration)

- [Step 1: Fitting a Training Step in Memory](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#step_1:_fitting_a_training_step_in_memory)
- [Step 2: Achieving Target Global Batch Size](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#step_2:_achieving_target_global_batch_size_)
- [Step 3: Optimizing Training Throughput](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#step_3:_optimizing_training_throughput)
- [Benchmarking thousands of configurations](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#benchmarking_thousands_of_configurations)
- [Lessons learned on benchmarking](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#lessons_learned_on_benchmarking)

[Diving in the GPUs – fusing, threading, mixing](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#diving_in_the_gpus_%E2%80%93_fusing,_threading,_mixing)

- [A primer on GPU](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a_primer_on_gpu)
- [How to improve performance with Kernels ?](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#how_to_improve_performance_with_kernels_?)

- [Memory Coalescing](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#memory_coalescing)
- [Tiling](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#tiling)
- [Thread Coarsening](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#thread_coarsening)
- [Minimizing Control Divergence](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#minimizing_control_divergence)

- [Fused Kernels](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#fused_kernels)
- [Flash Attention 1-3](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#flash_attention_1-3)
- [Mixed Precision Training](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#mixed_precision_training)

- [FP16 and BF16 training](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#fp16_and_bf16_training)
- [FP8 pretraining](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#fp8_pretraining)

[Conclusion](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#conclusion)

- [So, what’s next?](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#so,_what%E2%80%99s_next?)
- [Acknowledgements](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#acknowledgements)
- [Discussion page](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#discussion_page)

[References](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#references)

- [Landmark LLM Scaling Papers](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#landmark_llm_scaling_papers)
- [Training Frameworks](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#training_frameworks)
- [Debugging](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#debugging)
- [Distribution Techniques](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#distribution_techniques)
- [Hardware](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#hardware)
- [Others](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#others)

[Appendix](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#appendix)

- [A0: Parallel Programming Crash Course](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a0:_parallel_programming_crash_course)

- [Broadcast](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#broadcast)
- [Reduce & AllReduce](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#reduce_&_allreduce)
- [Gather & AllGather](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#gather_&_allgather_)
- [Scatter & ReduceScatter](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#scatter_&_reducescatter)
- [A quick focus on Ring AllReduce](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a_quick_focus_on_ring_allreduce)
- [Barrier](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#barrier)
- [NCCL: NVIDIA Collective Communications Library](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#nccl:_nvidia_collective_communications_library)

- [A1: Distributed Training Profiling](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a1:_distributed_training_profiling)

- [Kernels](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#kernels)
- [CPP extension](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#cpp_extension)

- [A2: Typical Scales in LLM Training](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a2:_typical_scales_in_llm_training)
- [A3: Math for Compute/Communication Overlap](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#a3:_math_for_compute/communication_overlap)

- [Data Parallelism Communication Analysis](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#data_parallelism_communication_analysis)
- [ZeRO-3 (FSDP) Communication Analysis](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#zero-3_\(fsdp\)_communication_analysis)
- [TP Communication Analysis](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#tp_communication_analysis)
- [PP Communication Analysis](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#pp_communication_analysis)


## 三、数据并行（DP）

数据并行（DP）背后的理念是在多个 GPU 上复制模型（我们将副本称为“模型实例”），并针对每个 GPU 并行地对不同的微批次数据进行前向传播和反向传播，因此得名数据并行。你可能已经在简单的训练示例中见过数据并行，但正如你很快会看到的，在本节中我们将深入探讨这一内容，所以即使你已经了解一般方法，也请继续关注。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_diagram.png)

（如果你不熟悉 broadcast、gather 或 all-reduce 等分布式通信模式，我们在 A0：并行编程速成课程中准备了一个小型速成课程。）

每个 GPU 使用不同的微批次意味着每个 GPU 中会有不同的梯度，因此为了使不同 GPU 上的模型实例保持同步，将使用一种称为 “all-reduce” 的操作对来自模型实例的梯度进行平均处理，该操作在反向传播期间、优化器步骤之前进行。

这涉及我们的第一个“分布式通信”原语：***all-reduce***，它处理 GPU 实例和节点之间的同步和通信。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap1.svg)

一个简单的分布式数据并行（DP）实现方式是等待反向传播完成，这样我们就有了所有梯度，然后触发所有分布式数据并行 ranks 之间的一次 all-reduce 操作来同步这些梯度。但这种先计算后通信的顺序步骤是***大忌***！因为我们不希望像上图那样，在进行通信时我们的 GPU 处于闲置状态。

相反，我们应该尽可能地让通信和计算重叠，使它们尽可能同时发生。

让我们来看看三种优化方法，它们能让我们比最初的简单实现做得更好！

### 3.1 三种优化方法

#### 3.1 方案一：将梯度同步与反向传播重叠

我们刚刚描述的朴素 DP 方法的主要缺点是，在反向传播（*计算*）之后，我们必须等待梯度同步（*通信*）才能更新参数。我们能否将此通信与我们的计算重叠？答案是肯定的！

如下图所示，在计算前面层的梯度之前，就可以收集并求和某一层的梯度。例如，一旦最后一层的反向传播完成，这些梯度就可以在为前面的层继续进行反向计算的同时被收集和求和。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap2.svg)

这可以在 PyTorch 中通过每个参数上附加一个 *all-reduce 钩子函数* 实现 。一旦该参数的梯度准备好，就会触发 all-reduce 操作，而其他参数的梯度仍在计算中。这种方法将大部分 all-reduce 操作与梯度计算重叠，从而提高效率。以下是一个用于附加钩子的简单函数：

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that 
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

计算和通信的重叠减少了等待整个模型梯度同步的时间。梯度同步可以（至少部分地）与反向传播并行进行，显著加快数据并行速度。以下是具有同步重叠的朴素数据并行（DP）的完整实现：

👉 Picotron 中存在重叠的朴素动态规划实现（点击展开）

```python
class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting point to understand how data parallelism works.
    It implements a simple all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """
    def __init__(self, module):
        """
        Initializes the DataParallel wrapper for a given module.

        Args:
            module (nn.Module): The model to be wrapped for data parallelism.
            process_group (torch.distributed.ProcessGroup): The process group used for gradient synchronization. 
                                                            It could be a data parallel or context parallel group.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require gradients.    
        """
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """
        Performs an all-reduce operation to synchronize gradients across multiple processes.    
        """
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization. 
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing 
        gradients in between.
        """
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
```


> [!important]
> [all-reduce 和 ring-reduce 在数据同步上的示意图](https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model)


这是我们第一个 “*计算与通信重叠*” 的例子，在本文中我们将多次讨论它，这是实现最大扩展效率的一项关键技术。但我们可以进一步提高效率！

#### 3.2 方案二：梯度分桶

GPU 操作在处理大张量时通常比在多个小张量上运行许多操作更高效。通信操作也是如此。因此，我们可以将梯度有利地分组到桶中，并对同一桶内的所有梯度启动单个 all-reduce，而不是对每个梯度执行独立的 all-reduce。通常看起来如下：

![dp_overlap3.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap3.svg)

这就像在装运前将物品装入箱子一样。发送几个大箱子比发送许多小箱子更高效。通过对每个桶执行单个 all-reduce 操作，我们可以显著减少通信开销并加快通信操作。

以下是采用分桶方式的代码实现：

👉 Bucket DP 在 Picotron 中的实现（点击展开）

```python
class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        """
        Initialize the DataParallelBucket module.
        
        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either 
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes. 
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 
```

#### 3.3 方案三：与梯度累积的相互作用

最后，正如我们之前看到的，梯度累积通过在用 `optimizer.step()` 更新参数之前执行多次前向和后向传播来工作。当将梯度累积与数据并行性结合时，我们希望在同步梯度时要小心。

在一个简单版本中，在累积过程中每次反向传播后都会自动触发 all-reduce 操作，这是次优的，因为在最后一步之后进行单次 reduce 将产生相同的效果，同时减少开销。

在 PyTorch 中，通常的解决方法是在不需要进行 reduce 的后向传播过程中添加一个 [`model.no_sync()`](https://github.com/pytorch/pytorch/blob/5ea67778619c31b13644914deef709199052ee55/torch/nn/parallel/distributed.py#L1408-L1435)装饰器，该装饰器可以禁用梯度同步。

> [!NOTE]
> 在执行通信操作时，张量在内存中必须是连续的，以避免多余的内存拷贝。为了以最优方式实现这一点，我们通常会预先分配大小与激活值或模型参数相匹配的连续缓冲区，专门用于通信。虽然这加快了通信速度，但在一定程度上也导致了训练期间的峰值内存使用量增加。

现在让我们看看这对全局批量大小意味着什么。

### 3.2 重新审视全局批量大小

我们可以使用新添加的数据并行和梯度累积参数来更新我们的批量大小公式：$$\text{bs} = \text{gbs} = \text{mbs} \times \text{grad\_acc} \times \text{dp}$$这里 $\text{grad\_acc}$ 是梯度累积步数，$\text{dp}$ 是用于数据并行的并行实例数量。

给定一个目标全局批量大小，我们因此可以通过梯度累积步骤来换取数据并行进程，从而加速训练。

在实际应用中，由于数据并行本质上是并行的，而梯度累积具有顺序性，人们倾向于尽可能多地增加数据并行节点（DP）而非采用梯度累积。当仅扩展数据并行性在 GPU 用完之前不足以达到目标全局批量大小时，就在数据并行的基础上添加梯度累积。

(关于数据并行性进一步阅读的一个好的资源是 https://siboehm.com/articles/22/data-parallel-training)

能够将训练分布到不同的样本上，为我们提供了第一个并行化的维度，因此这被称为 1D 并行（我们后续将逐步介绍另外四个维度）。

### 3.3 到目前为止我们的旅程

让我们快速总结一下如何设置我们的第一个 1D 并行训练，并为最佳数据并行设置提供一个草案配方：

1. 我们首先应通过查阅文献或开展测量模型收敛情况的实验来确定最佳的（全局）批量大小（以 tokens 为单位，`GBST`）。
2. 然后我们选择一个用于训练的序列长度，同样可以通过查阅文献或开展实验来确定。一般来说，对于我们目前的评估工作，2-8k 个 tokens 能可靠地发挥良好效果（我们在此不深入探讨训练方法，不过各团队通常会在训练结束时增加序列长度，混入一些更长上下文的数据样本，以达到如今的更长上下文尺寸）。
3. 现在我们已经知道了批量大小（`GBS`）。我们可以通过逐渐增加本地批量大小，直至耗尽内存，从而找出单个 GPU 上的最大本地批量大小（`MBS`）。
4. 最后，我们确定目标 DP 可用的 GPU 数量。GBS 与 DP 的比值能让我们得出实现所需 GBS 还需要的梯度累积步数。

(例如，DeepSeek 和 Llama 模型在主要预训练阶段是以 4k tokens 的序列长度进行训练的。)

(2-8k 在预训练中效果很好的原因是，网络上非常长的文档极为罕见。有关详细分析，请参阅 [Harm 的博客文章](https://www.harmdevries.com/post/context-length/)。)

如果梯度累积比率小于 1，也就是说我们有太多的 GPU（称为 GPU 丰富🤑），我们可以选择不使用所有的 GPU，探索更大的全局批量大小，或者测试较小的 MBS（每个 GPU 的批量大小）是否会加速训练。在后一种情况下，我们会优先考虑整体吞吐量而不是单个 GPU 的计算效率，使用比可能的更小的 MBS 来加快训练速度。

现在是时候举一个具体的例子了：假设我们想要训练一个最近提出的模型，该模型的全局批量大小（GBS）为 4M tokens，序列长度为 4k。因此，我们的批量大小将是 1024 个样本（我们选择最接近的 2 的幂次方）。假设我们观察到单个 GPU 在内存中只能容纳微批量大小 MBS=2，并且有 128 个 GPU 可用于训练。这意味着通过 4 个梯度累积步骤，我们将实现每个训练步骤 1024 个样本或 4M tokens 的目标。现在，如果我们突然有 512 个 GPU 可用呢？我们可以通过保持 MBS=2 并将梯度累积步骤设置为 1 来实现相同的 GBS，从而实现相同的训练，并获得更快的训练速度！

> [!NOTE]
> 请记住，在 512 个及以上 GPU 的规模下，根据所使用的网络，通信操作将开始受*环形延迟*（信号沿环形传输一圈所需的时间）的限制，这意味着我们无法再完全重叠数据并行（DP）通信。这将降低我们的计算效率并影响吞吐量。在这种情况下，我们应该开始探索其他并行维度。

虽然数据并行性能够很好地将  all-reduce 梯度同步与反向计算重叠以节省时间，但这种优势在大规模情况下开始崩溃。为什么呢？因为随着我们添加越来越多的 GPU（数百个或数千个），协调它们之间的开销显著增长，并且网络需求对于所获得的收益来说变得过大。结果，我们每向系统中添加一个额外的GPU，我们的设置将变得越来越低效。

让我们通过一些基准测试来看看这在实践中是如何实现的：

[交互图]

我们发现，在超过某个限制后，我们的吞吐量开始显著下降，而每个 GPU 的内存使用量保持不变，并且不会因为增加更多的 DP ranks 而受到影响。

*数据并行是我们首个（简单）的策略，用于将训练扩展到更多的 GPU 上。这种技术类似于梯度累积，但它对微批次的前向传播和反向传播进行并行处理，从而提高吞吐量！*

然而，敏锐的读者可能已经注意到，这是假设我们至少能将一个输入样本的前向传播（mbs=1）装入我们的 GPU 内存。但并非总是如此！我们可以看到，即使启用了激活重新计算，较大的模型也无法装入单个 GPU 中：

> [!tip]
> 提示：你可以通过将模型参数数量乘以 2 来快速估算模型参数所需的最小内存，例如 70B → 140GB（=133GiB）

[交互图]

我们还发现，在达到一定的扩展水平后，数据并行开始出现一些限制性的通信开销。对于这些更大的模型或大批量大小，我们还有其他选择吗？幸运的是，我们确实有一些解决方案。它们要么涉及将一些张量移动到 CPU，要么将权重/梯度/优化器状态张量拆分到 GPU 设备上！让我们开始深入了解它们。

有两种主要的拆分方法：并行性（张量并行、上下文并行或流水线并行）和共享（DeepSpeed Zero 或 PyTorch FSDP）。这两种方法在某种程度上是正交的，实际上可以结合起来！

共享范式与 DP 密切相关，因此我们将首先通过研究 ZeRO 方法来对其进行了解！

### 3.4 ZeRO (**Ze**ro **R**edundancy **O**ptimizer)

在本节中，我们将介绍 DeepSpeed ZeRO（零冗余优化器），这是一种内存优化技术，旨在减少大型语言模型训练中的内存冗余。

虽然数据并行是一种有效的扩展训练的方式，但在每个 DP rank 上简单复制优化器状态、梯度和参数会引入显著的内存冗余。ZeRO 通过将优化器状态、梯度和参数在数据并行维度上进行划分来消除内存冗余，同时仍然允许使用完整的参数集进行计算。这有时需要在 DP rank 之间进行更多的通信，这些通信是否能够完全重叠，我们接下来将会看到！

在本博客中，我们将重点关注 ZeRO-1 到 ZeRO-3，因为这应该能让我们全面了解它如何帮助减少内存占用，同时展示需要考虑的权衡。你可以在 [DeepSpeed 文档](https://www.deepspeed.ai/tutorials/zero/) 中找到更多 ZeRO 的相关内容。

这种方法分为 ZeRO 的三个可能的优化阶段：

- ZeRO-1：优化器状态分区
- ZeRO-2：优化器状态+梯度分区
- ZeRO-3（也称为 FSDP，即“完全分片数据并行”）：优化器状态+梯度+参数分区

（当我们说分区时，是指沿着 DP 轴进行分区，因为 ZeRO 是数据并行的一部分。稍后我们会看到，我们还可以沿着其他轴进行分区。）

你可能忽略了我们在可进行分片处理的事物中的激活操作。由于模型的每个 DP 副本接收不同的微批次，因此每个 DP rank 上的激活操作也各不相同，所以它们不会被复制，也就无法进行分片！

让我们更仔细地看看通过对每个 ZeRO 阶段进行分区，我们能节省多少！

#### 3.4.1 内存使用情况再探

你可能还记得我们在前面的章节中提到的标准训练期间优化器状态、梯度和参数的内存使用情况。我们把模型参数的数量记为 $Ψ$（之前用 $N$ 表示，但这里我们使用原始 ZeRO 论文的符号表示法）。在使用 Adam 优化器的混合精度训练中（更多细节见后面的章节），我们需要存储的每一项的内存使用量为：

- 模型的参数（半精度，即 bf16/fp16）：$2Ψ$
- 模型的梯度（半精度，即 bf16/fp16）：$2Ψ$
- 模型的 fp32 参数和优化器状态：$4Ψ+(4Ψ+4Ψ)$
- 模型的 fp32 梯度：$4Ψ$（可选，仅在我们要以 fp32 累积梯度时计算）

如果我们不在 fp32 中累积梯度，那么总的内存消耗为 $2Ψ+2Ψ+12Ψ$；如果我们进行累积，那么将是$2Ψ+6Ψ+12Ψ$。为简单起见，我们现在先关注不进行 fp32 梯度累积的情况，不过你可以将受 ZeRO-2 和 ZeRO-3 影响的梯度项的额外字节数加上去。

ZeRO 的理念是将这些对象分片到 DP 各个 rank 中，每个节点仅存储这些项的一个切片，当且仅当需要时才对这些项进行重构，从而将内存使用量按数据并行度 $N_d$​ 进行划分 。

![zero_memory.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/zero_memory.svg)
这里 $Ψ$ 表示参数数量，$k$ 表示优化器状态的内存乘数（如我们刚刚看到的，对于 Adam，$k=12$），$N_d$ 表示 DP 度。

让我们通过探究每个 ZeRO 阶段的工作原理来解释这张图及其数值。我们将从 ZeRO-1 开始。

#### 3.4.2 ZeRO-1: 分区优化器状态

在普通 DP 中，所有进程在后向传播后收集相同的梯度，并同时执行相同的优化器步骤。这看起来像是很多重复的工作。我们能否避免这种情况，同时减少内存使用呢？

在 ZeRO-1 中，优化器状态被划分为 $N_d$ 个相等部分，其中 $N_d$ 是数据并行（DP）度。这意味着分布在每个 DP rank 上的每个模型副本仅跟踪 $1/N_d$ 的优化器状态。在优化步骤中，只有 $1/N_d$ 的 float32 权重被更新。

然而，在前向传播过程中，每个副本都需要所有参数，因此我们需要在优化器步骤之后添加一个额外的 ***all-gather*** 操作（这是我们遇到的第二种通信原语！），以便每个模型副本都有完整的更新后的权重集。

这解释了我们在上图中看到的内存占用公式 $2Ψ+2Ψ+kΨ/N_d$，以下是单个训练步骤的操作顺序总结：

- 在每个副本上使用相同的完整 bf16 参数集进行前向传播，但不同副本处理不同的微批次。
- 在每个副本上使用相同的完整梯度集进行反向传播，但不同副本处理不同的微批次。
- 对梯度执行 reduce-scatter 操作（我们将在下图中解释 reduce-scatter 原语）。
- 每个副本在其本地优化器上执行一步优化器操作（仅有 $1/N_d$ 优化器状态），以获得更新的 $1/N_d$ fp32 参数，然后将其转换为完整 bf16 参数集的 $1/N_d$。
- 在 bf16 参数之间执行 all-gather 操作，将缺失的切片发送回每个副本。这是 ZeRO 中的新操作，在普通的数据并行（DP）中未使用。

> [!NOTE]
> 注意：reduce-scatter 比 all-reduce 快 2 倍！_耶，第三种通信原语！_
> 

你可能会想知道这个 “reduce-scatter” 操作是什么，以及这一切看起来是怎样的，所以让我们借助下面的图示让这一切更加直观。我们将详细讲解前向/反向传播周期的所有步骤：

![dp_zero1.gif|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero1.gif)

在实际通信方面，与普通 DP 相比，Zero-1 将我们的 “all-reduce” 梯度通信更改为 “reduce-scatte” 操作，并在优化器步骤之后添加一个针对所有参数的 “all-gather” 操作。其过程如下：

![dp_zero1_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero1_overlap.svg)

如果你一直关注，会从普通 DP 中回想起，我们可以在反向传播计算过程中重叠进行 all-reduce 梯度通信。在 ZeRO-1 中，我们还可以研究如何高效地重叠新添加的 bf16 参数 all-gather 操作。主要有两种策略：

- 在优化器步骤期间：我们可以在优化器更新部分参数后立即启动 all-gather 操作。这使得通信有可能与其他参数的更新重叠。
- 在前向传播期间：我们可以将每层参数的 all-gather 操作与前向传播过程重叠起来。

> [!NOTE]
> 不幸的是，这些技术并不容易实现，并且需要巧妙地使用钩子/分桶。在实际应用中，我们可以直接使用 PyTorch 原生的 ZeRO-3/FSDP 实现，并将 FSDPUnit 设置为整个模型，关于这个的更多细节稍后会介绍。

在 ZeRO-1 中，优化器状态已被分区，这意味着每个副本仅更新 $1/N_d$ 的优化器状态。敏锐的读者肯定已经注意到，其实一开始并不需要所有 DP ranks 上都有所有梯度，因为优化步骤只需要其中一部分梯度。这就引出了 ZeRO-2！

#### 3.4.3 ZeRO-2: 添加梯度分割

由于我们只需要在每个副本上拥有与优化器状态分片相对应的梯度分片，因此将梯度也类似地分片是有意义的。在反向传播过程中，我们不是对梯度执行 all-reduce 操作，而是只执行 reduce-scatter 操作！我们只在内存中传播所需的 $1/N_d$ 梯度，从而比 ZeRO-1 节省更多内存。

在 FP32 梯度累积的情况下，我们只需要保留 $1/N_d$ fp32_grads，用于累积来自 reduce-scatter 的 bf16 梯度。在优化器步骤中，我们使用这 $1/N_d$ fp32_grads。

![dp_zero2.gif|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero2.gif)

现在很容易看出，对梯度进行分片会导致 $2Ψ+\frac{2Ψ+kΨ}{N_d}$，并且随着 $N_d$​ 的增加，与基线相比，我们可以节省多达 8 倍的内存。在通信方面，与 ZeRO-1 的过程相同，唯一的区别是我们即时进行通信并释放。总的来说，就通信而言，ZeRO-2 也因此等同于普通的 DP 训练。

在通信方面，ZeRO-2 与 ZeRO-1 相似，它们都需要对梯度进行 reduce-scatter 操作，并对所有参数进行 all-gather 操作。
![dp_zero2_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero2_overlap.svg)

> [!tip]
> 注意：您可能会注意到，与 ZeRO-1 相比，使用 ZeRO-2 并没有真正的额外开销，实际上 ZeRO-2 通常是最佳选择。


现在我们已经对梯度进行了分片处理，那么我们是否已经完成了任务，还是可以继续这样做呢？嗯，差不多。接下来就是 ZeRO-3！

#### 3.4.4 ZeRO-3: 添加参数分区

对于第 3 阶段，我们将上述在数据并行（DP）副本上对优化器状态和梯度进行分片的方法扩展到对模型的参数进行分片。

> [!NOTE]
> 这个阶段在 PyTorch 原生实现中也被称为 FSDP（完全共享数据并行）。在本文中，我们仅使用 ZeRO-3 这个术语，但无论何时看到它，你都可以将其理解为 FSDP 。
> 

那么，如果模型的所有部分都是分布式存储的，我们在实践中如何进行前向传播或反向传播呢？很简单，我们在需要时按需收集它们。在前向传播中，过程如下：

![dp_zero3_fwd.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_fwd.svg)

因此，在进行前向传播并依次通过各层时，我们会按需检索必要的参数，并在不再需要这些参数时立即将它们从内存中清除。反向传播的工作方式相同，只是流程相反，我们会生成梯度分片：

![dp_zero3_bwd.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_bwd.svg)

另一个问题是，在前向传播和反向传播步骤中，我们需要持续执行这些全规约操作。与 Zero-2 相比，在一个训练步骤中，这相当于额外增加了 $2⋅\text{num\_layers}−1$ 次 all-gathers 操作，而且正如我们在下图中看到的，每次操作都会带来一定的基础延迟开销 。

![dp_zero3_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_overlap.svg)

在前向传播过程中，当我们需要参数时，我们会对它们执行 all-gather 操作，因此会产生 $Ψ$ 的通信开销。由于在前向传播中一旦用到参数就会立即丢弃，所以在反向传播过程中我们还需要再进行一次 all-gather 操作，这又产生了 $Ψ$ 的通信开销。最后，和 ZeRO-2 一样，我们对梯度也需要进行相同的 ***reduce-scatter*** 操作，这在通信方面同样需要 $Ψ$ 的开销。综上，总的通信开销为 $3Ψ$，而 ZeRO-2 的通信开销为 $2Ψ$。

这听起来可能像是会有大量的通信开销，但实际上情况还挺好的，因为我们可以采用所谓的预取（prefetching）技术，将下一层参数的通信与当前层的前向传播过程重叠起来。通过预取，在进行前向传播时计算当前层（第 $n$ 层）的前向过程的同时，我们会 “all-gather” 第 $n+1$ 层的权重；同样地，在计算第 $n$ 层的反向传播过程时，我们会 “all-gather” 第 $n-1$ 层的权重。当然，只有当我们对数据并行（DP）的扩展程度不太大时，这种重叠才是有效的。（经验法则：数据并行的规模不应超过 512）

在内存方面，我们可以看到我们的方程现在达到了其最终形式 $\frac{2Ψ+2Ψ+kΨ}{N_d}$，这意味着如果我们能够增加 DP ranks，至少对于模型相关参数而言，我们可以无限降低内存使用量。注意，这对中间激活值并无帮助，对于中间激活值，正如我们在前面章节中所看到的，我们可以使用激活值检查点和梯度累积的方法。

*让我们总结一下迄今为止在分布式数据并行（DP）和 ZeRO 方面的探索历程：我们已经看到，通过简单地增加模型副本，利用分布式数据并行（DP）可以显著提高训练的吞吐量。而借助 ZeRO，我们甚至能够训练那些通常无法放入单个 GPU 的模型，方法是将参数、梯度和优化器状态在分布式数据并行（DP）中进行分片处理，不过这会带来一定的通信开销。*

如果你想了解更多关于 FSDP1、FSDP2 以及它们周围一些实现复杂性的内容，你应该花些时间仔细阅读[这篇不错的博客](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/)。

然而，这里存在一个限制，即 DP 仅在模型的一个层能适配单个 GPU 时才有效，而 ZeRO 只能对参数、梯度和优化器状态进行分区，却无法对激活内存进行分区！我们从激活内存的讨论中回忆一下，这部分内存随着序列长度和批量大小而扩展。自然地，我们可以简单地限制这些因素，但在实践中，我们并不希望由于硬件的限制而只能使用短序列长度进行训练。

[交互图]

为了克服这些问题，是时候探索一种新的、正交的并行性轴——张量并行性（TP）了。与依赖大量参数通信的 ZeRO3 不同，TP 提出在设备间对参数、梯度、优化器状态以及激活进行分片，而不需要在GPU 之间进行模型参数的通信。

什么？这怎么可能？！让我们一起探索这种看似神奇的方法吧！ 🙂

## 四、张量并行（TP）

所以我们已经使用 ZeRO 对模型的参数、梯度和优化器状态进行了分片，但是一旦激活内存超过我们的内存预算，我们就遇到了一个限制。欢迎张量并行（TP），这是一种对权重、梯度、优化器状态以及激活进行分片的方法，并且不需要在计算之前将它们全部收集起来。这听起来像是一个梦想！让我们首先看看张量并行是如何通过简单的矩阵乘法来工作的。

张量并行利用了矩阵乘法 $A\times B$ 的数学特性。要理解其工作原理，让我们来看看使这种并行化成为可能的两个基本方程式：$$\begin{align*}
1. \quad & A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots \end{bmatrix} \\[1.2ex]
2. \quad & A \cdot B = \begin{bmatrix} A_1 & A_2 & \cdots \end{bmatrix} \begin{bmatrix} B_1 \\ B_2 \\ \vdots \end{bmatrix} = \sum_{i=1}^{n} A_i B_i
\end{align*}$$
这意味着我们可以通过以下两种方式来计算矩阵乘积：1）分别乘以 $B$ 的每一列；或者2）分别乘以每一行并将结果组合起来。在神经网络中，矩阵乘法通常以以下格式表示：$X \times W$，其中：

* $X$ 表示输入或激活值  
* $W$ 表示 `nn.Linear` 的权重

在实际操作中，该操作的一个小示例是这样的：

![TP diagram|240](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram.svg)

让我们看看如何对这个操作进行并行化处理！在张量并行中，张量将沿着特定维度被分割成 $N$ 个分片，并分布在 $N$ 个 GPU 上。矩阵可以在列部分或行部分进行分割，从而实现行并行和列并行。接下来我们会看到，选择行分片还是列分片将需要不同的通信原语。

我们的第一个选择是使用按列分片（也称为***列线性*** ）：我们将把完整的输入矩阵复制到每个工作节点，这需要一个称为***广播*** 的操作，并将权重矩阵分割成列。然后将输入与部分权重矩阵相乘，最后使用 ***all-gather*** 操作合并结果。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram2.png)

以下是按列进行张量并行的代码实现：

👉 Picotron 中的列并行张量并行实现（点击展开）

```python
class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.output_size_per_partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        if self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias) 
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias) 
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)
        return output
```

第二个选项称为按行分片（也称为***行线性***分片）：细心的读者可能会猜到，行线性分片意味着我们将权重矩阵分割成行块。然而，这也要求我们对输入进行分割，这需要一个 ***scatter*** 操作，而不是像列线性分片中使用的广播。每个工作器上的结果已经是正确的形状，但需要求和以得到最终结果，因此在这种情况下需要一个 all-reduce 操作。

我们在这里看到了我们的第四个分布式原语：**_scatter_**！

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram3.png)

以下是按行进行张量并行的实现方式：

👉 Picotron 中的行并行张量并行实现（点击展开）

```python
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # X_i * W_i^T + b
        output_parallel = F.linear(x, self.weight)
        # All-reduce across all the partitions.
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output if self.bias is None else output + self.bias
```

既然我们已经了解了 Transformer 的基本构建模块，现在让我们看看如何在 Transformer 层中有效地组合它们！

### 4.1 Transformer 块中的张量并行性

为了提出一个可遵循的策略，让我们从一个简单的示例过渡到一个真实的模型构建模块。Transformer模型由两个主要的构建模块组成：前馈层（MLP）和多头注意力（MHA）。我们可以对这两者都应用张量并行性。

前馈部分可以通过先进行“列线性”操作，再进行“行线性”操作来实现并行化，这相当于在前向传播中进行广播以复制输入并进行 all-reduce 操作。请注意，在实际训练中不需要广播，因为我们可以确保输入已经在 TP rank 之间同步。这种设置比先进行“行线性”操作，再进行“列线性”操作更高效，因为我们可以跳过两个拆分操作之间的中间 all-reduce 操作。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram4.png)

既然我们已经找到了 Transformer 前馈部分的一个高效模式，那么让我们来看看多头注意力块（MHA）。

我们通常可以采用类似的方法，其中 Q、K 和 V 矩阵以列并行的方式拆分，输出投影沿行维度拆分。对于多头注意力机制，列并行方法有一个非常自然的解释：每个工作器计算单个头或一组头的注意力。这种方法同样适用于多查询（MQA）或分组查询注意力（GQA），在这些机制中，键和值在查询之间是共享的。

然而，值得注意的是，张量并行度不应超过查询/键/值（Q/K/V）头的数量，因为每个张量并行（TP）等级都需要完整的头（否则我们就无法在每个 GPU 上独立计算注意力，并且将需要额外的通信操作）。如果我们使用分组查询注意力（GQA），张量并行度实际上应小于键/值（K/V）头的数量。例如，LLaMA-3 8B 有 8 个键/值头，所以张量并行度最好不超过 8。如果我们对这个模型使用 TP=16，我们需要在每个 GPU 上复制键/值头，并确保它们保持同步。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_full_diagram.png)

最后请注意，张量并行性仍然不是训练的万能良方。我们在模型的计算路径中直接添加了几种分布式通信原语，因此很难像我们在 ZeRO 中所做的那样将其与计算完全隐藏/重叠，我们的最终性能将是计算和内存增益与增加的通信开销之间权衡的结果。让我们来说明这一点：

![Forward pass in Tensor Parallelism|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_overlap.svg)

（通过对分块矩阵乘法并结合异步通信/计算，可以部分隐藏这种通信。）

观察张量并行多层感知机（MLP）（注意力机制同样适用）的操作时间线，我们能更好地理解其中涉及的权衡。在每个解码器层的前向传播中，我们会遇到一个同步点，即AllReduce操作，该操作无法与计算重叠。这种 *显露出来的通信开销* 对于在最终应用层归一化（LayerNorm）之前合并张量并行 ranks 之间的部分结果是必要的。

（例如，Megatron-LM/Nanotron 实现了 all-gather 与 FC1 计算的部分重叠，其中矩阵乘法结果的一部分将在另一部分仍在计算时开始发送到其他 GPU。）

张量并行确实有助于减少矩阵乘法的激活内存，因为中间激活被分片到多个 GPU 上。然而，对于像 LayerNorm 这样的操作，我们仍然需要收集完整的激活，这意味着我们没有获得我们本可以获得的全部内存优势。此外，张量并行引入了显著的通信需求，这在很大程度上取决于网络基础设施。无法将这种特定的 AllReduce 完全隐藏在计算背后意味着它会直接增加前向传播的关键路径。

（这一研究领域仍然是一个活跃的研究领域，近期的研究工作如 Domino[4] 探索了最大化这种重叠的新技术。）

让我们更仔细地看看在扩展 TP（张量并行）度时所涉及的权衡：

[交互图]

增加训练进程数（TP）会导致每个 GPU 的吞吐量降低（左图），但它能够处理更大的批量大小（右图），这说明了分布式训练中计算效率和内存可用性之间的权衡。

实际上，正如我们在左图中看到的那样，当扩展到 8 个 GPU 以上时，张量并行的通信开销变得尤为明显。虽然在单个节点内可以利用快速的 NVLink 互连实现张量并行，但跨节点则需要较慢的网络连接。我们观察到从 TP=8 增加到 TP=16 时出现了显著的下降，并且从 TP=16 增加到 TP=32 时下降更为陡峭。在更高的并行度下，通信开销变得如此之高，以至于迅速占据了计算时间。

话虽如此，张量并行通过将模型参数、梯度、优化器状态以及（在一定程度上）激活分布到多个 GPU 上，为内存使用提供了重要优势。让我们看看这对一个 70B 参数模型产生的影响：

[交互图]

提高张量并行度可减少每个 GPU 上模型参数、梯度和优化器状态所需的内存，从而让我们能够开始在单个 8 GPU 节点上拟合大型模型。

有没有办法从这种技术中获得更多的好处呢？我们已经看到，层归一化和 dropout 仍然需要在每个 GPU 上收集全部激活值，这在一定程度上抵消了内存节省的效果。我们可以通过寻找并行化这些剩余操作的方法来做得更好。

> [!NOTE]
> 关于张量并行训练中层归一化的一个有趣说明——由于每个张量并行（TP）ranks 在 all-gather 后看到相同的激活值，因此在反向传播后，层归一化权重实际上不需要 all-reduce 来同步它们的梯度。它们自然会在各等级之间保持同步。然而，对于 dropout 操作，我们必须确保跨 TP ranks 同步随机种子，以维持确定性行为 。

接下来让我们探讨一下张量并行的一种小型且自然的扩展方式，即*序列并行*，它所做的事情正是如此。

### 4.2 序列并行

序列并行（SP）涉及对张量并行（TP）未处理的模型部分（如 Dropout 和 LayerNorm）的激活和计算进行拆分，但沿输入序列维度而非隐藏维度进行拆分。

> [!NOTE]
> “序列并行”（Sequence Parallelism）这个术语有些含义过载：本节中的序列并行与张量并行紧密相关，并且适用于dropout（随机失活）和层归一化（layer norm）操作。然而，当我们将处理更长的序列时，注意力计算将成为瓶颈，这就需要诸如环形注意力（Ring-Attention）之类的技术，这些技术有时也被称为“序列并行”，但为了区分这两种方法，我们将把它们称为“上下文并行”（Context Parallelism）。所以，每次看到“序列并行”时，要记住它是与张量并行一起使用的（与上下文并行相对，上下文并行可以独立使用）。

这是因为这些操作需要访问完整的隐藏维度才能正确计算。例如，LayerNorm 需要完整的隐藏维度来计算均值和方差：$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
其中，$\mu = \text{mean}(x)$ 和 $\sigma^2 = \text{var}(x)$ 是在隐藏维度 $h$ 上计算得到的。

因此，即使这些操作在计算上成本较低，但由于需要完整的隐藏维度，它们仍然需要大量的激活内存。序列并行（SP）允许我们通过沿序列维度进行拆分，在多个 GPU 之间分担这一内存负担。

在实际操作中，我们将从左图过渡到右图：
![|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_diagram.png)
该图表展示了我们如何使用不同的集合操作（分别标记为 “f” 和 “g”）在张量并行区域和序列并行区域之间进行转换。关键挑战在于高效地管理这些转换，同时保持较低的内存使用量并确保正确性。

在前向传播中：

* "f" 是一个空操作（无操作），因为激活已在各 ranks 之间进行了复制  
* "f*" 是一个 all-reduce 操作，用于同步激活并确保正确性

在反向传播中：

* "f*" 是无操作（no-op），因为梯度已在各 ranks 中被复制  
* "f"是 all-reduce 操作，用于同步梯度

这些运算 “f” 和 “f*” 被称为*共轭* 对，因为它们相互补充——当一个在正向中是无操作（no-op）时，另一个在反向中就是 all-reduce，反之亦然 。

对于序列并行（SP），我们使用标记为 “g” 和 “g*” 的不同操作。具体而言，我们在 SP 区域中避免使用 all-reduce 操作，因为这需要收集全部激活值，从而增加我们的峰值内存使用量，违背了 SP 的初衷。

那么这里到底发生了什么？正如一个著名的 LLM 所说，让我们一步一步来：

* **初始层归一化（SP 区域）**:
	* 输入张量 $X_1$ 和 $X_2$ $(b,s/2,h)$ 进入层归一化，已在序列维度上拆分
	- 每个 GPU 独立地在其序列块上计算层归一化，并给出 $Y_1$ 和 $Y_2$

* **第一次转换（SP → TP）:**
	- “g” 操作（all-gather）将 $Y_1$ 和 $Y_2$ 重新组合为完整序列长度
	- 恢复 $Y$ $(b,s,h)$，因为列线性需要完整的隐藏维度 $h$

* **第一次线性（TP 区域）:**
	* $A_1$ 是列线性的，因此它沿隐藏维度分割 $Y$  
	* GeLU 在每个 GPU 上独立应用
	* $Z_1$ 是 $(b,s,h/2)$

* **第二次线性（TP 区域）：**  
	* $B_1$ 是行线性的，因此它恢复隐藏维度  
	* $W_1$ 是 $(b,s,h)$ 

* **最终转换（TP → SP）：** 
	* “g*” 操作（reduce-scatter），在沿序列维度分散的同时确保前一行线性的正确性  
	* $W_1$ 是 $(b,s/2,h)$

![image.png|240](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_diagram_zoomed.png)

序列并行性的一个关键优势在于它减小了我们需要存储的最大激活大小。在仅使用张量并行的情况下，我们不得不在不同位置存储形状为 $(b,s,h)$ 的激活值。然而，有了序列并行性，由于我们总是在序列维度或隐藏维度上进行拆分，最大激活大小减小为 $\frac{b⋅s⋅h}{tp}$。

跟踪在 TP 和 TP/SP 中以不同方式进行分片的所有部分有点困难——相信我们，我们也觉得很难进行映射，所以我们制作了这个小表格来总结在前向传播过程中，激活（即 `hidden_states`）的形状在隐藏维度 $h$ 和序列维度 $s$ 上是如何变化的：

|Region|TP only|TP with SP|
|---|---|---|
|Enter TP (Column Linear)|h: sharded (weight_out is sharded)  <br>s: full|h: sharded (weight_out is sharded)  <br>s: **all-gather** to full|
|TP Region|h: sharded  <br>s: full|h: sharded  <br>s: full|
|Exit TP (Row Linear)|h: full (weight_out is full + **all-reduce** for correctness)  <br>s: full|h: full (weight_out is full + **reduce-scatter** for correctness)  <br>s: **reduce-scatter** to sharded|
|SP Region|h: full  <br>s: full|h: full  <br>s: sharded|

并且对于嵌入层：

|Region|Vanilla TP|TP with SP|
|---|---|---|
|Embedding Layer (Row Linear sharded on vocab)|h: full (weight_out is full + **all-reduce** for correctness)  <br>s: full|h: full (weight_out is full + **reduce-scatter** for correctness)  <br>s: **reduce-scatter** to sharded|

通过使用序列并行性，我们可以实现更大的激活内存节省，从而使我们能够将批量大小和序列长度推得比仅使用张量并行性时更远。让我们看看这对我们之前的 70B 模型示例意味着什么：

[交互图]

正如我们所见，我们再次大幅降低了每个 GPU 的最大内存使用量，使我们能够在 TP/SP=16 的情况下处理 16k tokens 的序列长度，这比普通 TP 情况有所改进！（如前一节所述，TP=16 仍然有点大，但我们将在下一节中看到如何改进这一点）。

你可能会问自己的一个问题是，使用 TP+SP 是否比普通 TP 产生更多的通信量？嗯，答案是有时是，有时不是。在普通 TP 的前向传播中，每个 Transformer 块有两个 all-reduce 操作，而在 SP 中，每个 Transformer 块有两个 all-gather 和两个 reduce-scatter 操作。所以 SP 的通信操作数量是 TP 的两倍。但由于一个 all-reduce 操作可以分解为一个 all-gather + reduce-scatter（见附录中的“快速关注 Ring AllReduce”部分），它们在通信方面实际上是等效的。后向传播的推理也相同，因为我们只是使用每个操作的共轭（无操作 ↔ allreduce 和 allgather ↔ reducescatter）。

如果你一直密切关注，就会注意到我们正在讨论每层的 4 个通信操作（注意力机制的 2 个和多层感知机的 2 个）。使用张量+序列并行时，多层感知机的分析情况如下：

![tp_sp_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_overlap.svg)

就像普通的张量并行（TP）一样，张量并行（TP）+流水线并行（SP）也不能轻易地与计算重叠，这使得吞吐量严重依赖于通信带宽。同样，在这里，就像普通的张量并行（TO）一样，张量并行（TP）+流水线并行（SP）通常也只在单个节点内进行（使张量并行度保持在每个节点的GPU数量之下，例如张量并行度 ≤ 8 ）。

我们可以衡量随着张量并行性扩展，这种通信开销变得日益棘手的情况。让我们在针对序列长度为 4096 的 3B 参数模型，将张量并行（TP）与流水线并行（SP）一起扩展时，测量吞吐量和内存利用率：

[交互图]

在这里，计算效率（左）和内存容量（右）之间又存在一种权衡。虽然更高的并行度通过减少激活内存能够处理显著更大的批量大小，但它们也降低了每个 GPU 的吞吐量，特别是在对应于每个节点的 GPU 数量的阈值以上。

让我们总结一下我们的观察结果：

- 对于这两种方法，我们注意到当从 TP=8 切换到 TP=16 时，性能下降最为明显，因为此时是从仅在单个节点内通信（NVLink），转变为节点间通信（EFA）。
- 在使用 TP 与 SP 时，激活内存的节省使我们能够处理比仅使用 TP 时大得多的批次。

*我们已经看到 TP 如何通过在隐藏维度上拆分注意力和前馈操作来帮助我们在多个 GPU 上分片激活，以及 SP 如何通过在序列维度上拆分来自然地补充其余操作。*

> [!NOTE]
> 由于 SP 区域中的层归一化（LayerNorms）对序列的不同部分进行操作，因此它们在张量并行（TP）ranks 上的梯度会有所不同。为了确保权重保持同步，我们需要在后向传播期间对它们的梯度进行 all-reduce 操作，类似于数据并行（DP）确保权重保持同步的方式。然而，这是一种较小的通信开销，因为层归一化的参数相对较少。

然而，TP 和 SP 有两个限制：1）如果我们扩展序列长度，激活内存仍然会在 TP 区域爆炸式增长；2）如果模型太大而无法适应 TP=8，那么由于节点间连接问题，我们将看到巨大的减速。

我们可以用上下文并行性来解决问题 1)，用流水线并行性来解决问题 2)。让我们先来看看上下文并行性！

## 五、上下文并行（CP）

通过张量并行和序列并行，我们可以显著降低每个 GPU 的内存需求，因为模型权重和激活都分布在多个 GPU 上。然而，当在越来越长的序列上训练模型时（例如，当扩展到每个序列 128k 或更多 tokens 时），由于在 TP 区域内我们仍然需要处理完整的序列长度，我们可能仍然会超出单个节点上的可用内存。

此外，即使我们使用完全重新计算激活值（这会带来约 30% 的巨大计算开销），我们仍然需要在层边界处保留一些与序列长度线性相关的激活值在内存中。让我们来看一下上下文并行性如何帮助我们：

[交互图]

上下文并行性的核心思想是将类似的思路应用于序列并行性方法（即沿序列长度进行拆分），但要应用于我们已经应用张量并行性的模块。因此，我们将沿两个维度对这些模块进行拆分，从而减少序列长度的影响。在我们已经讨论过的内容之后，你会发现这种方法相当直观……但这其中有个技巧，所以要保持清醒！

为了实现上下文并行性；就像序列并行性一样，我们将沿着序列维度拆分输入，但现在我们将这种拆分应用于整个模型，而不仅仅是我们之前在张量+序列并行性中所做的模型的序列并行区域。

拆分序列不会影响大多数模块，如 MLP 和 LayerNorm，其中每个标记都是独立处理的。它也不需要像TP那样进行昂贵的通信，因为只拆分了输入而不是权重矩阵。就像数据并行一样，在计算梯度后，会启动一个 all-reduce 操作来同步上下文并行组中的梯度。

不过有一个重要的例外情况，那就是我们需要特别关注注意力块（哈哈，双关语😀）。在注意力模块中，每个标记都需要访问所有其他序列标记的键/值对；在因果注意力的情况下，至少要对每个前面的标记予以关注。

由于上下文并行性会沿序列维度将输入拆分到多个 GPU 上，因此注意力模块将需要在 GPU 之间进行完整的通信，以交换必要的键/值数据。

“如果我们天真地去做这件事，那听起来成本非常高。有没有一种既能高效又能快速完成的方法呢！值得庆幸的是，有这样的方法：一种用于高效处理键值对通信的核心技术叫做*环形注意力（Ring Attention）*。”

> [!NOTE]
> 上下文并行性与 Flash Attention（稍后详述）在概念上有一些相似之处——这两种技术都依赖于在线 softmax 计算来减少内存使用。虽然 Flash Attention 专注于优化单个 GPU 上的注意力计算本身，但上下文并行性通过将序列分布到多个 GPU 上来实现内存减少。


### 5.1 发现环形注意力机制

在这种注意力机制的实现中，每个 GPU 首先发起一个异步通信操作，将其键/值对发送到其他 GPU。在等待其他 GPU 的数据时，它会计算其内存中已有数据部分的注意力分数。理想情况下，在本次计算完成之前，会从另一个 GPU 接收到下一个键/值对，从而使 GPU 在完成第一轮计算后能够立即开始下一轮计算。

让我们来说明一下。假设我们有 4 个 GPU 和 4 个输入标记。最初，输入序列会沿着序列维度均匀拆分，因此每个 GPU 将仅有一个标记以及其对应的 Q/K/V 值。假设 Q1、K1 和 V1 分别代表位于第 1 个 GPU 上的第一个标记的查询、键和值。注意力计算将需要 4 个时间步来完成。在每个时间步，每个 GPU 执行这三个连续的操作：

1. 以非阻塞方式将“当前的键和值”发送到下一台机器，但在最后一个时间步除外，这样我们就可以在这一步完成之前开始下一步。
2. 在本地计算其已有的“当前的键和值”上的注意力分数，这通常涉及执行 $\text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \times V$。
3. 等待接收来自前一个 GPU 的键和值，然后循环回到步骤 1。此时“当前的键和值”现在是刚刚从前一个 GPU 接收到的键/值。

我们执行这 3 个步骤四次以完成注意力计算。

以下动画展示了使用 4 个 GPU 的整个过程：
![ring-attention.gif|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ring-attention.gif)
在这个动画中，作者选择将这种方法称为环形注意力机制（Ring Attention），这一点对您来说可能显而易见。

不过有一个大问题是，环形注意力（Ring Attention）的简单实现会由于因果注意力矩阵的形状导致 GPU 之间出现严重的不平衡。让我们通过考虑带有因果注意力掩码的注意力得分矩阵来看看 SoftMax 计算。
![cp_attnmask.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_attnmask.svg)

SoftMax 是按行计算的，这意味着只要 GPU 接收到一行的所有 tokens，就可以进行计算。我们看到 GPU1 可以立即计算它，因为它从 token 1 - 4开始，并且 GPU1 实际上不需要从任何其他 GPU 接收任何信息。然而，GPU2 将需要等待第二轮才能也接收到 1-4，从而拥有 tokens 1-8 的所有值。此外，GPU1 似乎比所有其他 GPU 执行的工作量要少得多。

让我们看看能否更好地平衡我们的计算。

### 5.2 锯齿（之字形）环注意力——一种平衡的计算实现

我们需要一种更好的方法来分配输入序列。这可以通过不纯粹按顺序将 tokens 分配给 GPU，而是稍微混合一下顺序来实现，这样每个 GPU 上都有早期和晚期 tokens 的良好混合。这种方法称为锯齿形（Z 字形）注意力，在这种新的排列中，注意力掩码将显示计算量的均匀分布，但如果你数一数有颜色的方块数量，就会发现计算量现在在所有 GPU 之间是平衡的。

![cp_zigzagmask.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_zigzagmask.svg)

与此同时，我们还会看到，为了完成所有行，每个 GPU 都需要来自其他所有 GPU 的信息。

我们有两种通用的方法来重叠计算和通信，要么通过执行常规的 all-gather 操作，同时在每个 GPU 上重新分组所有的键值对（以 Zero-3 类型的方式），要么根据需要从每个 GPU 逐个收集到其他 GPU。

![cp_overlap_allgather.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_overlap_allgather.svg)

![cp_overlap_all2all.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_overlap_all2all.svg)

这两种实现方式的关键区别在于它们的通信模式和内存使用情况：

1. **AllGather实现**：
    - 所有 GPU 同时从所有其他 GPU 收集完整的键/值对
    - 由于每个 GPU 需要一次性存储完整的 KV 对，因此需要更多的临时内存
    - 通信在一步内完成，但具有更大的内存开销
2. **全互联（环形）实现**：
    - GPU 以环形模式一次交换一个数据块的 KV 对
    - 更节省内存，因为每个 GPU 只需要临时存储一个额外的数据块
    - 通信分散并与计算重叠，尽管由于多个通信步骤而有一些额外的基本延迟开销

全互联（All-to-All）方法通常以稍微复杂的通信模式为代价提供更好的内存效率，而全收集（AllGather）方法更简单，但在注意力计算期间需要更多的临时内存。

我们现在看到了如何通过在一个节点上使用张量并行（TP）来拆分模型以应对大型模型，以及如何使用序列并行（CP）来应对长序列中的激活爆炸问题。

然而，我们仍然知道TP 在跨节点扩展方面表现不佳，那么如果模型权重不容易放在 1 个节点上，我们该怎么办呢？另一种并行度，我们的第四种，*并行流水线（Pipeline Parallelism）* 来拯救了！

## 六、流水线并行（PP）

在“张量并行”部分，我们看到，尝试将张量并行扩展到单个节点的 GPU 数量（通常为 4 或 8）以上时，会遇到一种带宽较低的网络，称为“节点间连接”，这可能会严重损害我们的性能。例如，当我们在跨多个节点（每个节点有 8 个 GPU）的集群上对其进行基准测试时，我们可以清楚地看到这一点，比如 all-reduce 操作。

[交互图]

不同节点数量下节点间通信带宽的测量结果，展示了AllReduce、AllGather 和 ReduceScatter 操作的中位数（线条）以及第 5 至第 95 百分位范围（阴影区域）。

序列并行和上下文并行对于长序列可能有帮助，但如果序列长度并非我们内存问题的根本原因，而是模型本身的大小导致的，那么这两种并行方式作用就不大了。对于大型模型（70B 参数及以上），仅权重的大小就可能超过单个节点上 4 到 8 块 GPU 的极限。我们可以通过引入第四个（也是最后一个）并行维度——“流水线并行”来解决这个问题 。

流水线并行是一种简单但强大的技术 —— 我们将模型的层分布到多个 GPU 上！例如，如果我们有 8 个 GPU，我们可以将第 1-4 层放在 GPU 1 上，第 5-8 层放在 GPU 2 上，依此类推。这样，每个 GPU 只需要存储和处理模型层的一部分，显著减少了每个 GPU 的内存需求。让我们看看流水线并行在 8B 模型的内存使用上的效果：

（这种技术可能会让你想起我们在讨论 ZeRO-3 时的情况，当时我们将模型参数分割到多个 GPU 上。在后面的 “5D 并行简介” 部分，我们会详细对比这两种技术。）

[交互图]

观察上图，我们注意到一个有趣的现象：虽然模型参数在各个 GPU 上分配得很好，但每个 GPU 上的激活内存却保持不变！这是因为每个 GPU 仍然需要处理完整的数据批次，只是处理不同的层。一个 GPU 层的激活将被发送到下一个 GPU 以继续前向传播。

这引入了一种新的通信模式：我们现在不是像在数据并行中使用 ZeRO-3 那样通信参数，而是在 GPU 之间按顺序“流水线式”传递激活张量。虽然概念上很简单，但高效实现这种技术相当棘手。让我们直接深入细节！

### 6.1 在不同节点上拆分层 - 全部前向，全部后向

所以，假设我们只是将各层分布到几个设备上，例如，第一个 GPU 将处理前几层，第二个 GPU 将处理模型的第二部分，依此类推。现在，通过我们模型的前向传播简单地涉及按顺序将数据批次沿模型传递，从而连续使用每个计算设备。

我们有一个直接的首要优势：所需的互连带宽保持相当低，因为我们在模型深度的少数位置仅发送中等大小的激活。与例如张量并行中的通信相比，这可以产生巨大差异，后者在每层内会发生数次。

但也许你开始感觉到一些即将到来的麻烦： **“sequentially”** 和 **“successively”**？！？在并行计算的世界里，这听起来并不高效，特别是在我们讨论了计算和通信重叠之后。

确实，读者朋友们！流水线并行中的主要挑战在于如何有效地规避流水线并行的顺序性，使我们的 GPU 始终保持忙碌状态，避免出现一个 GPU 在计算而其他 GPU 在等待的情况。下面展示的是我们在对模型进行简单直接的向前和向后传播时 GPU 的利用率情况（此处数字表示模型的各层）：

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_afab.svg)

*一个在 4 个 GPU 上分布有 16 层的模型的流水线并行示例。数字对应层 ID。*

剩余的空闲时间以灰色表示，通常称为“气泡”，在花费了这么多时间优化吞吐量之后，看到这个可能会让你心碎。

我们可以通过观察因“气泡”而损失的时间来量化流水线设置的效率。设 $t_f$ 和 $t_b$ 分别为前向和后向传播的时间（针对一个微批次和流水线的一个阶段测量）。一个简单的假设是 $t_b \approx 2 \times t_f$，如上图所示。如果我们能够完美并行化，理想的总时间将是 $t_{\text{id}} = t_f + t_b$。然而，由于流水线气泡，额外的时间为 $t_{\text{pb}} = (p-1) \times (t_f + t_b)$，其中 $p$ 是流水线并行度，即上图中的 GPU 数量。这表示每个 GPU 在其他 GPU 计算时的等待时间。

我们可以计算额外气泡时间相对于理想时间的比率：$$r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{t_f + t_b} = p - 1$$
随着我们增加更多阶段，气泡时间因此增加，利用率下降。正如我们所看到的，在简单实现中，气泡可能会非常大！

值得庆幸的是，人们已经设计出了各种流水线并行方案来*减小气泡规模*。

让我们从工具箱中取出第一个工具，思考将我们的批次分割成更小的、可以并行或几乎并行处理的小块（部分），就像我们之前在数据并行中所做的那样。现在，当第二个 GPU 忙于处理微批次1时，第一个 GPU 已经可以开始处理微批次 2 了。以下是使用 8 个微批次的调度安排：

![pp_afab2.svg|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_afab2.svg)

（在之前的图表中，数字表示的是层，但从现在起（包括本图）的所有流水线并行图中，数字表示的是微批次。你可以将这里的每个方格看作包含若干层，就像前一幅图中所展示的那样。）

上述调度方式被称为***全前向全后向（AFAB）*** 调度，因为我们首先进行所有前向传播，然后再只进行所有后向传播。其优势在于前向步骤和后向步骤总体上仍然是顺序执行的，因此我们保留了模型训练代码的一般组织结构。这使得这种流水线并行（PP）实现成为最容易实现的方式之一 。

你可以在 picotron 中找到 AFAB pipeline 的完整实现。

👉 AFAB PP 在 Picotron 中的 PP 实现（点击展开）

```python
def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1

    for _ in range(data_loader.grad_acc_steps): # All forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for ith_microbatch in range(data_loader.grad_acc_steps): # All backward passes
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss
```

让我们在这个例子中估算一下气泡。与我们的第一个例子不同的是，现在处理 $m$ 个小批量的理想时间是 $t_{id}=m \times (t_f+t_b)$：$$r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{m}$$
正如我们所见，通过增加更多的微批次，我们可以将气泡的大小缩小 $m$ 倍，从而解决流水线阶段的一些低效率问题。

然而，与气泡同样烦人的是存储所有激活所需的存储空间。在达到反向阶段之前，我们需要将所有激活保存在内存中，这导致这些 PP 实现中的内存迅速爆炸。我们能做得更好，避免这种内存爆炸吗？

由于内存爆炸是由我们为反向传播存储的激活触发的，让我们尝试看看是否可以在我们仍在进行计算的前向部分时就开始执行反向传播。这将使我们能够尽快丢弃一些用于反向传播所需的激活。

### 6.2 “向前一步-向后一步”及 LLama 3.1 方案

这种调度被称为*一前一后（1F1B）*，因为中间/稳定状态涉及交替进行一次前向传播和一次反向传播。其总体思路是尽早开始进行反向传播。调度过程如下：
![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b.svg)

如果你仔细计算就会发现气泡大小仍然相同，因此我们的训练效率并没有显著提高。然而，我们只需存储 $p$ 个微批次的激活值（其中 $p$ 是流水线并行度），而不是 $m$ 个（其中 $m$ 是微批次的数量），这样可以减少 AFAB（假设为某种调度方式，具体需结合上下文确定准确含义）调度中出现的激活值内存爆炸问题。因此，我们可以增加更多微批次，而这实际上会减少气泡。

这种设置的复杂性（如上图所示）在于，前向传播和反向传播不再是清晰的顺序执行，而是在设备间并行执行并交错进行。这意味着我们将不得不在每个设备上独立地安排从前向传播到反向传播的切换，而不是像往常一样在一个简单且通用的中央训练循环中进行。

这就是实施流水线并行通常需要对训练代码以及建模代码进行相当广泛的修改的原因之一。

你也可以在 picotron 中找到 1F1B 的完整实现：

👉 1F1B PP 在 Picotron 中的实现（点击展开）

```python
def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):    
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    
    def _forward_step(input_tensor):
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    for _ in range(num_warmup_microbatches): # Warmup forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    for ith_microbatch in range(num_microbatches_remaining):  # 1F1B steady state
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    for ith_warmup_microbatches in range(num_warmup_microbatches): # Cooldown backward passes
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss
```

让我们通过在我们的集群上进行的一些基准测试，来看一下 1F1B 流水线并行调度在实践中是如何扩展的：

![Throughput scaling of Pipeline Parallelism with varying microbatch sizes|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b_scaling.png)

在左侧，微批处理数量等于或小于流水线并行（PP）度减一（$m = p - 1$）时，我们可以看到流水线气泡的危害有多大——性能很低，而且随着流水线并行度的增加甚至还会下降。右侧图表显示，使用远多于流水线并行度的微批处理数量（$m = 32 \gg p - 1$）有助于改善低流水线并行度下的性能，但在非常大的流水线并行度下仍然受限。实际上，由于最终受到目标全局批量大小的限制，我们无法随意增加微批处理数量以保持 $m \gg p - 1$ 的比例。随着流水线并行度的增加，当微批处理数量达到可能的最大值时，我们最终必须根据 $r_\text{bubble}=\frac{p−1}{m}$ 来增加气泡大小 。

有趣的是，在微批次数量较少时，从单个节点（$p = 8$）扩展到两个节点（$p = 16$），性能仅下降了 14% ——这比张量并行性在类似的跨节点场景中通常出现的约 43% 的性能退化要好得多。当遇到节点间低带宽网络时，这种行为使得流水线并行性在跨多个节点的分布式训练中特别有吸引力。

虽然 1F1B 显著减少了我们的激活内存占用，但从这最后一张图中我们可以看到，流水线气泡仍然是一个主要的效率瓶颈。由于气泡大小仍与流水线阶段数成正比，我们让宝贵的 GPU 计算能力处于闲置状态。我们能否设计出一种更巧妙的调度方案来尽量减少这种浪费的计算时间呢？

### 6.3 交错阶段

1F1B 调度让我们改善了内存使用情况，但对空闲包的大小改善不大。无论如何，我们还能推进这一边界吗？

原来，如果我们愿意引入一些额外的通信操作，这是可行的。现在是时候谈谈*交错阶段*了。

到目前为止，我们一直沿着模型深度维度天真地对模型进行切片处理，例如将第 1-4 层放在第一个 GPU 上，将第 5-8 层放在第二个 GPU 上。但是，我们还可以考虑其他对层进行切片的方式，比如将奇数层（第 1、3、5、7 层）放在第一个 GPU 上，将偶数层（第 2、4、6、8 层）放在第二个 GPU 上。

这通常可以被看作是一种“环形管道”，微批次在通过模型的前向传播时会从一个 GPU 循环移动到下一个 GPU。让我们通过图形来看看这是如何工作的：

![pp_1f1b_interleaved.svg|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b_interleaved.svg)

*一个在 4 个 GPU 上分布各层的模型的交错流水线并行的示例。编号仍对应微批处理 ID，但为了清晰起见，我们对模型的第一层和最后一层进行了不同着色，以说明各层是如何分布在 GPU 上的。*

因此，我们看到由于模型针对同一计算在每个 GPU 上要经过多次（而此前只需一次），从而出现了额外的通信情况。不过，每次前向传播和反向传播都被一个因子 $v$ 所分摊，其中 $v$ 是阶段数或者每个 GPU 的模型块数，因为我们现在能够更好地交错进行前向传播和反向传播。$$\begin{align}
t_{pb} &= \frac{(p - 1) \times (t_f + t_b)}{v} \\[1.2ex]
r_{\text{bubble}} &= \frac{1}{v} \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{v \times m}\end{align}$$
因此，我们现在可以通过添加微批次和交错阶段来减小气泡，但请注意，从数量上来说，通信量也会因 $v$ 而增加，所以这是一种权衡。在下面的图中，你可以看到 $p = 8$ 的 PP 设置的几种配置，其中特殊情况 $m = 1, v = 1$ 对应于简单的流水线并行，$v = 1$ 的配置是 AFAB 或 1F1B 设置，而 $v ≠ 1$ 是交错配置。

[交互图]

调度在这里也变得更加复杂，因为我们必须在给定的 GPU 上和给定的时刻决定我们是优先处理先到达的微批次数据，让它们通过后续层（即尽快完成前向和反向传播循环，也就是所谓的“深度优先”，即优先让批次数据尽快离开模型），还是优先处理后续的微批次数据，让它们通过前面的层（即所谓的“广度优先”，即优先尽可能填满流水线）。这种选择在精彩的“广度优先流水线”论文[^6]中有详细解释。

你现在拥有了理解 Llama 3.1 中流水线并行方法的所有要素，该方法采用一次前向传播一次反向传播的设置，各阶段交错进行，并且优先级设置可以在深度优先和广度优先之间进行调整。

![pp_llama3.1_schedule.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_llama3.1_schedule.png)

然而，我们尚未到达可能的流水线调度方案的尽头，最近已经有一些方法被提出来将气泡减少到几乎为零！例如，这些技术已在 DeepSeek V3/R1 实现[^7]中使用。勾起你的好奇心了吗？在我们离开流水线并行世界之前，让我们最后快速看一眼这些神奇的调度方案！

### 6.4 零气泡和双管

最近提出了更为复杂的方法来减少气泡，这些方法几乎达到了“零气泡”状态。这里的秘诀是在更细粒度的层面上拆分相关操作，以最有效的方式将它们交错执行。例如，DeepSeek V3/R1 中的管道实现方法，称为 DualPipe，几乎达到了零气泡状态。

（DeepSeek V3 技术报告[^7]中的终极“灵活配置”，作者在报告中指出他们的设置“实现了近乎为零的全互联通信开销”。）

让我们简要地通过总结作为 DualPipe 前身的 ZeroBubble[^8] 的工作来看看这是如何运作的。 ZeroBubble 的基本观察结果是，矩阵乘法的反向传播实际上涉及两个分离的操作：输入（B）的反向操作和权重（W）的反向操作。

虽然输入的反向传播（即 B 的输出）对于执行较低层的反向传播是必要的，但权重的反向传播（即 W 的反向传播）对于其余的反向传播过程并非必要，并且通常只需在优化器步骤之前执行。我们可以在下图中看到这一点：

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_compgraph.png)

这意味着 W 可以灵活地安排在同一阶段的相应 B 之后的任何位置。这使得可以策略性地放置 W 以填充流水线气泡。右上角的 ZB-H2 时间表是利用这种细粒度分解实现零气泡的（理论）时间表示例。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_ppschedule.png)

*在顶部（ZeroBubble 论文中的图 2）：经典的 1F1B 调度，交错进行前向和后向传播，但保持粗粒度的后向传播。在下面的两个图中（ZeroBubble 论文中的图 3），ZeroBubble 调度的两个变体，将后向操作拆分为更细粒度的 “B” 和 “W” 操作。最后一个调度，称为 “ZB-H2”，是一个（理论上）利用这种细粒度分解实现零气泡的调度示例。*

DeepSeek 的 DualPipe 在其 V3 技术报告[^7]中介绍了这种分解方法的扩展，即针对从 PP 维度的两端传播的两个流的情况，这些流交错排列，以进一步减少 GPU 中的空闲时间。该调度方案显示在下面的调度图中，比之前的方案更为复杂。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_dualpipe.png)

一般来说，要完全优化这种复杂的调度安排，需要仔细测量各种细粒度操作的持续时间，并求解一个整数线性规划（ILP）问题以使最终的气泡时间最小化。例如，可参考 ZeroBubble 论文[^8]中对执行此类调度所采用的启发式算法和算法的讨论。因此，ZeroBubble 和 DualPipe 的调度安排过于复杂，在此无法给出代码片段，但你应该开始对所涉及的概念有一个大致的了解。

这就结束了我们对管道调度和气泡世界的参观。希望您喜欢这次导览之旅！

现在是时候探讨我们将详细介绍的最后一种并行方法了，即我们可以用它来高效训练大型模型的方法：*专家并行*。

## 七、专家并行（EP）

这是我们要讨论的最后一个并行方法。在探讨它之前，如果您对专家混合模型（Mixture-of-Experts）没有任何了解，欢迎阅读我们之前发布的一篇较短的博客文章，这篇文章应该能帮助您更好地理解专家混合模型（MoE）架构。

专家混合模型近期因 GPT - 4、Mixtral[^9] ，以及更近期的 DeepSeek-V3/R1 等模型而受到关注并崭露头角。其基本思想是，我们可以在每一层设置多个并行模块，而非单个前馈模块，并将令牌通过其中一个或另一个模块进行路由，以进行不同的处理。

![ep_moe.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ep_moe.png)
*取自 Switch Transformers 论文[^10]的 MoE 层示意图*

MoE 层的设计实际上使得跨专家维度实现并行性变得非常容易，我们将其称为专家并行性（EP）。由于前馈层是完全独立的，我们可以简单地将每个专家的前馈层放在不同的工作器上。与 TP 相比，它要轻量得多，因为我们不需要拆分矩阵乘法，只需要将 tokens 的隐藏状态路由到正确的专家即可。

在实际应用中，专家并行（EP）通常会与其他并行形式（例如数据并行）结合使用。这是因为专家并行仅影响混合专家（MoE）层，并且不会对输入标记进行分片（不像上下文并行那样沿序列长度维度对标记进行分片）。这意味着，如果我们仅使用专家并行，我们的图形处理单元（GPU）将对所有非 MoE 块执行冗余计算。通过将专家并行与数据并行相结合，我们可以像下面的简化示意图中所示，有效地在 GPU 之间对专家和输入批次进行分片 。

![ep_schema.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ep_schema.png)
*来源：《专家混合模型综述》[^11]*

但先别高兴得太早——在接下来的部分中，我们将具体探讨不同并行策略之间的所有交互，所以如果你还没理解这最后一张图，也别担心。

在实际应用中，要使专家并行（EP）高效运行有一些技巧，并且这些技巧与模型设计密切相关。例如，DeepSeek-V3 在路由器中施加了一种约束，确保每个 token 最多被发送到 M 个节点（在他们的案例中，M 为 4），以将 token 保留在单个节点上并减少通信开销。虽然专家并行已经存在一段时间了[^12]，但随着混合专家（MoE）架构越来越受到关注，它现在才重新受到重视 。

我们计划在 picotron/nanotron 中尽快添加一个更完整的 EP 示例，敬请关注更多内容！

## 八、5D 并行性概述

恭喜读者，您现在已经了解了可用于扩展模型训练的所有 5 种并行策略：

1. DP ——沿批次维度
2. TP ——沿隐藏维度
3. SP/CP ——沿序列维度
4. PP ——沿模型层
5. EP ——沿模型专家

以及 3 种 ZeRO 策略，这些策略可与数据并行性相结合以减少内存占用：

1. ZeRO-1 – 在 DP 副本间对优化器状态进行分片
2. ZeRO-2 – 在 DP 副本间对优化器状态和梯度进行分片
3. ZeRO-3 – 在 DP 副本间对优化器状态、梯度和参数进行分片

At this stage, one aspect you are probably curious about is how all these parallelism and ZeRO strategies compare to, and interact with, each other. In other words, which ones should we use and efficiently combine together, and which ones should we rather keep separated?

Let’s take a look at the similarities and interplay. We'll start by comparing Pipeline parallelism are ZeRO-3 side-by-side as they have some very close similarities but also important differences.

**Pipeline parallelism vs. ZeRO-3 -** Both PP and ZeRO-3 are ways to partition the model weights over several GPUs and perform communication/computation along the model depth axis (for example in ZeRO-3, we prefetch the next layer while computing). This means in both cases full layer operations are computed on each device, as opposed to TP or EP for instance in which computation are performed on sub-layer units.

In the following we say “a layer” to simplify what should be in general called “a set of layer” (as the basis sharding unit of the model).

However, there are a few major differences between PP and ZeRO-3 approaches:

||**ZeRO-3**|**Pipeline Parallelism**|
|---|---|---|
|Each compute unit stores|only a fraction of a layer|a full layer|
|Communication is used to transfer|weights|activations|
|Orchestration|model agnostic|model agnostic|
|Implementation challenges|Complex to handle model partitioning and communications|Complex to handle efficient PP schedules|
|Scaling considerations|Prefers large mbsmbs and seq_lenseq_len to hide comms|Prefers large grad_accgrad_acc to hide bubble|

As you can see, ZeRO-3 and PP solve the same challenge but involve different approaches and the choice between both will depend whether you decide to focus communication either on weights or on activations. While they can be combined, it's not often done in practice as doing so requires increasing the global batch size significantly to amortize the communication costs, creating a tradeoff between global batch size, model size, network bandwidth, and training efficiency. If you decide to combine them, ZeRO-3 should be configured to keep the weights in memory during the series of PP micro-batches to minimize as much as possible un-necessary communication overhead.

On the other hand, ZeRO-1 and ZeRO-2, which focus on optimizer states and gradients, can be easily combined with Pipeline Parallelism and are complementary to it. Combining them don't raise any particular new challenge. For instance, the training of DeepSeek-v3 used PP combined with ZeRO-1 (sic).

**Tensor Parallelism** (with Sequence Parallelism) is naturally complementary and can be combined with both Pipeline Parallelism and ZeRO-3 as it relies on the distributive property of matrix multiplications which allows weights and activations to be sharded and computed independently before being combined.

![TP & SP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_tp_sp.svg)

The main reason we don't want to use TP only for parallelism is that, in practice, TP has two limitations we've discussed in the previous sections: First, since its communication operations are part of the critical path of computation, it's difficult to scale well beyond a certain point at which communication overhead begins to dominate. Second, unlike ZeRO and PP which are model-agnostic, TP requires careful handling of activation sharding - sometimes along the hidden dimension (in the TP region) and sometimes along the sequence dimension (in the SP region) - making it more cumbersome to implement correctly and requiring model-specific knowledge to ensure proper sharding patterns throughout.

As a consequence, when combining parallelism strategies, TP will typically be kept for high-speed intra-node communications while ZeRO-3 or PP can be used for parallelism groups spanning lower speed inter-node communications as their communication patterns require less bandwidth (for PP) or can be more easily overlapped with computation (for ZeRO-3). The main consideration when combining these techniques is to organize the GPU efficiently in groups for each parallelism dimension to maximize throughput and minimize communication overhead, while being mindful of TP's scaling limitations. For instance, the groups of GPUs communicating for TP should be kept inside nodes.

**Context Parallelism** and **Expert Parallelism** also help us shard activations, and can be seen as complimentary to TP. The first one handles long sequences while the second enables distributed Mixture of Experts training and they can be combined together without any particular issue.

**Context Parallelism (CP)** specifically targets the challenge of training with very long sequences by sharding activations along the sequence dimension across GPUs. While most operations like MLPs and LayerNorm can process these sharded sequences independently, attention layers require communication since each token needs access to keys/values from the full sequence. As we saw in [CP section](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#context_parallelism), this is handled efficiently through ring attention patterns that overlap computation and communication. CP is particularly valuable when scaling to extreme sequence lengths (128k+ tokens) where, even when using full activation recomputation, the memory requirements for attention would be prohibitive on a single GPU.

![CP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_cp.svg)

**Expert Parallelism (EP)** specifically targets the challenge of training Mixture of Experts (MoE) models by sharding specialized "experts" across GPUs and dynamically routing tokens to relevant experts during computation. The key communication operation in EP is the `all-to-all` operations routing tokens to their assigned experts and gathering the results back. While this operation introduces some communication overhead, it enables scaling model capacity significantly since each token is only processed during inference (and training) by a much smaller fraction of the total parameters. In terms of distributed training/inference, partitioning experts across GPUs becomes relevant when models scales to a large number of experts.

For instance DeepSeek V3 uses 256 experts.

![EP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_ep.svg)

📝 Note

This similarity between EP and DP in terms of input handling is why some implementations consider Expert Parallelism to be a subgroup of Data Parallelism, with the key difference being that EP uses specialized expert routing rather than having all GPUs process inputs through identical model copies.

**Scope and focus** Let's also quickly summarize the sub-part of the model where some of these different parallelism strategies have the most impact:

- Tensor Parallelism (and Sequence Parallelism) affects computation throughout the entire model by sharding both weights and activations.
- Context Parallelism primarily impacts attention layers since that's where cross-sequence communication is required, with other layers operating independently on sharded sequences.
- Expert Parallelism primarly affects the MoE layers (which replace standard MLP blocks), leaving attention and other components unchanged
- Pipeline Parallelism and ZeRO are not especially specific to any sub-module or component with the exception that modules and layers need to be balanced in Pipeline Parallelism, the first and last layers are thus often treated differently due to the additional embedding layers.

|**Tensor + Sequence Parallel**|**Context Parallel**|**Expert Parallel**|
|---|---|---|
|shards weights and activations along hidden/seq dim|shards activations along sequence dim|shards specialized expert weights and activations|
|communication for matrix multiply operations (column/row linears)|communication for attention key/values|communication for token routing to experts|
|model-specific implementation needed|model-agnostic except for attention|model-agnostic except for MoE layers|
|Prefers high-bandwidth intra-node communication|Prefers large sequence lengths|Requires MoEs|

**Summarizing it all–** Now what about gathering and combining all the techniques we've seen in a single diagram combining them all. Yes, we're up for the challenge!

In this summary diagram, you will find illustrated activations and modules for a single transformers layer –in it's MoE variant–. We also illustrate the various directions of parallelism and the communication operations we've been discussing in all the previous sections.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_full.svg)

We can also represent side-by-side a **full overview** of the memory savings for each one of these strategies. We'll plot them with different sequence length as well as with selective (top) and full (bottom) recomputation so you can see how they all play with activations:

![5Dparallelism_8Bmemoryusage.svg](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5Dparallelism_8Bmemoryusage.svg)

Let's finish this section with a high level view at all of these techniques, their main underlying idea and major bottleneck:

|**Method**|**Memory savings applies specifically on**|**Parallel/sharding dimension**|**Disadvantage**|
|---|---|---|---|
|DP|Activations (reduce local batch size)|Batch|Limited by max batch size|
|PP|Model parameters|Model layers|Idle bubble and complex schedules|
|TP/SP|Model parameters and activations|Hidden dimension / Sequence length|Requires high bandwidth communication|
|CP|Activations|Sequence length|Add communication overhead in attention modules|
|EP|Experts parameters|Expert dimension|Requires MoE layers, add routing communication overhead|
|ZeRO-1|Optimizer states|Sharded among DP replicas|Params communication overhead|
|ZeRO-2|Optimizer states and gradients|Sharded among DP replicas|Params communication overhead|
|ZeRO-3|Optimizer states, gradients, and model parameters|Sharded among DP replicas|Params communication overhead|

Clearly, none of these techniques is a silver bullet for magical scaling and we'll often have to combine them in one way or another. Can we actually come up with a few rules that would help us find a good starting point to choose among –and combine– them? This will be the topic of our next section.

## Finding the Best Training Configuration

We’ve now covered all the parallelism techniques that are actually used to distribute and train larger models as well as how and why they can be combined together. There remain a general question: which ones should we choose in the end and how to decide on a specific combination?

We touched this a little bit in the previous section but let's now walk in details through a possible decision process, step by step, keeping in mind that you'll always have to run a few experiments to find the definitive optimal setup for your compute cluster given its various physical properties, network bandwidth, GPUs per node, memory per GPU, etc.

### Step 1: Fitting a Training Step in Memory

First, we need to figure out how we can fit a full model instance on our GPUs. There are two general cases.

**GPU-rich case 🤑** - when you have plenty of GPUs available:

- For models under 10B parameters, you can use a single parallelism technique, e.g. Tensor Parallelism or ZeRO-3/DP with Full Recompute across 8 GPUs
- For models between 10B-100B parameters requiring more than 8 GPUs, you have several options:

- Combining Tensor Parallelism (TP=8) with Pipeline Parallelism
- Combining Tensor Parallelism (TP=8) with Data Parallelism (ZeRO-3)
- Using only ZeRO-3 (i.e. only pure Data Parallelism)

- At 512+ GPU scale, pure Data Parallelism/ZeRO-3 will start to becomes inefficient due to communication cost - it can be better to then combine DP with either Tensor or Pipeline Parallelism
- At 1024+ GPU scale, a recommended setup can be Tensor Parallelism TP=8 with Data Parallelism (ZeRO-2) and Pipeline Parallelism

We focus on fitting a single instance for now - even though we may use DP for ZeRO to achieve this goal - we're only interested here in the model-parameters memory savings that it provide when used with ZeRO-3.

Special considerations:

- For very long sequences, you will probably want to add Context Parallelism (CP) across nodes.
- For Mixture of Experts architectures, you will advantageously use Expert Parallelism (EP) across nodes.

**GPU-poor case 😭** - when you might be low on GPU resources:

- You can enable full activation recomputation to trade some compute for memory (and train a bit slower).
- You can increase gradient accumulation to process larger batches with limited memory.

Now that we have a first model instance training, we need to make sure we have the right batch size.

### Step 2: Achieving Target Global Batch Size

Depending on where step 1 left us in terms of micro batch size and DP, our current batch size might be too small or too big. It's now time to hit our target batch size.

To increase our current global batch size:

- We can scale up Data Parallelism or gradient accumulation steps
- For long sequences, we can leverage Context Parallelism

To decrease our current global batch size:

- We can reduce Data Parallelism in favor of other parallelization strategies
- For long sequences, we can reduce Context Parallelism

Ok, now we have the model running in the general configuration we want in terms of model size and batch size, but are we training it the fastest way? Let's now start to optimize throughput as much as possible.

### Step 3: Optimizing Training Throughput

So we want to make sure the training is running as fast as possible so all our precious GPUs are well utilized at all times. As long as memory and communication aren't bottlenecks we can try the following:

- Scale up Tensor Parallelism (using the fast intra-node bandwidth) until we reach a degree close to the node size, so that we can reduce other parallelism
- Increase Data Parallelism with ZeRO-3 while keeping target batch size
- When Data Parallelism communication starts to become a bottleneck, transition to using Pipeline Parallelism
- Try scaling up different parallelisms one by one
- Experiment with several micro batch size (mbs) to aim for an optimal balance between max GBS, model size, compute, and communication.

### Benchmarking thousands of configurations

Now that we've covered the step-by-step, let's implement this search process in real-life.

You will find, in the [nanotron](https://github.com/huggingface/nanotron) repository, several scripts you can use to run all the experiments we discussed above and be able to benchmark your own model and cluster in real life.

We actually ran ourself benchmarks on **several thousands of distributed configurations** covering every model size we've discussed above as well as a very large number of cluster configurations (namely 1-64 nodes of 8xH100s) we could try in order to produce the results we've covered up to now in this book.

We want to take this opportunity to apologize to our co-workers for blocking most of the science cluster and in turn forgive any threats that may have been whispered.

Now let's take a step back to gather and analyze the results of all our benchmarks and see if, beyond theory, we can actually discover on real-world data how various configurations fare against each other.

All the following benchmarks were conducted with a sequence length of 4096 and a global batch size of 1M tokens. We gathered all the top configurations for each model and cluster size and plotted them in the following heatmaps:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/what_we_learnt_heatmap.svg)

Heatmap visualization showing the optimal training configurations across different model sizes and compute node counts (we have 8 GPUs per node). For each combination, the configuration details include Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), Gradient Accumulation Steps (GAS), Micro Batch Size (MBS), and ZeRO optimization stage. The color intensity indicates the Model FLOPs Utilization (MFU), with brighter colors representing higher efficiency.

From this high-level visualization, we can draw several important insights:

First, as we increase the number of nodes (higher parallelism), we observe a decrease in efficiency. This effect is particularly pronounced for smaller models, which have a lower compute-to-model-size ratio. While we might typically compensate for small model size by increasing the batch size, we're constrained by our global batch size limit of 1M.

Second, Larger models present a different challenge. As model size increases, memory requirements grow substantially. This creates two scenarios with fewer nodes: either the model doesn't fit at all, or it barely fits but runs inefficiently due to operating near the GPU memory limits (see for instance the 80B parameter model training on 4 nodes).

Finally, our benchmarks show how performance heavily depends on implementation quality. When we first implemented both parallelism strategies, Tensor Parallelism (TP) outperformed Pipeline Parallelism (PP). After optimizing our PP code, it became the faster option. Now that we're improving the communication overlap in our TP implementation, we expect it to regain the performance lead.

### Lessons learned on benchmarking

Our goal for this book was not only to discuss theory and implementations but provide actual data points as well. So the plan was simple: let's run every possible distributed configuration for every model and a number of cluster sizes (namely 1-64 nodes of 8xH100s). Even after excluding impossible configuration we still needed to run thousands of experiments.

On paper this sounds easy enough: we can easily launch big arrays of jobs on our cluster. However, as soon as we launched the first batches of experiments, troubles began:

- PyTorch processes would sometimes fail to clean up properly
- Slurm job manager would forcefully terminate jobs, leading to node failures
- Simple benchmarks that should take minutes would stretch into hours
- Some jobs would hang indefinitely

Running all experiments in a finite amount of time required additional engineering and we ended up spending a significant amount of time on things like:

- Minimizing cluster restart times and optimize idle time
- Analyzing detailed NCCL debug logs
- Understand memory usage patterns and CUDA memory allocator behaviors
- Improving pipeline parallelism performance on multi-node

These challenges deserve their own story, but they taught us valuable lessons about the complexities of distributed training infrastructure. What looks simple in theory often requires careful attention to many moving parts in practice.

Reproducing theoretical results in practice is challenging, especially given the limited availability of production training code. Through open-source projects like [nanotron](https://github.com/huggingface/nanotron) and [picotron](https://github.com/huggingface/picotron), we hope we can help making distributed training techniques more accessible as well as collaborating on simple and efficient codebases that help researchers and practitioners take the most out of their hardware resources.

---

This concludes our very deep dive into the distribution methods of 5D parallelism.

Taking a step back, our discussion so far has often relied on a critical assumption - that computation and communication can be efficiently overlapped on GPUs without any impact on the computation throughput. The reality is more nuanced. When using common communication primitives like NCCL send/recv, we face hidden contention between computation and communication resources as communication kernels will usually make use of the same GPU streaming multiprocessors (SMs) that are used for computation, leading to decreased throughput when communication is overlapped with computation. To truly optimize our distributed training, we need to dive deeper into the GPU architecture itself.

Additionally, the synchronization patterns when overlapping computation and communication may not always be optimal for our parallel strategies. You can find an example for instance in [this blog post](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) by the Pytorch team.

Time to turn the lights off and activate CUDA mode!

## Diving in the GPUs – fusing, threading, mixing

To add a podcast feeling to your reading experience, feel free to listen to the NotebookLM hosts discussing the following sections of this book as you're reading along.

Up to now our discussion has been focused on the high-level organization of our model operations. We’ve moved around computations on various accelerators, taking into account general memory constraints and high-level scheduling of the compute units.

But this ignored all the optimizations we can do at a much lower level by carefully understanding how our model operations are scheduled and performed on each GPU.

This section will dive into much more details of the GPU architecture and in particular in NVIDIA’s GPU architecture but the general ideas, as often, can be reused on similar accelerator units.

We’ll briefly explain how GPU are organized before covering the Flash-Attention revolution, how to efficiently schedule workload on GPU and finally explain how various precisions can be efficiently used on GPU.

### A primer on GPU

Generally, GPUs have a very hierarchical organization. In this primer we’ll keep the discussion at the concept levels that are necessary for the rest of our presentation.

On the compute side, GPUs consist of an array of compute units called **Streaming Multiprocessors** (SM). Each SM contains and controls a set of streaming processors, also known as cores. For example, an Nvidia H100 GPU has 132 SMs with 128 cores per SM, resulting in a total of 16,896 cores (see [docs for tensor cores](https://resources.nvidia.com/en-us-tensor-core) for details), each capable of handling multiple threads simultaneously.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/diving_primergpu.svg)

Source: https://blog.codingconfessions.com/p/gpu-computing

The memory side is also highly hierarchical with several layers of cache and memory: **Registers** are the smallest units and are private to the threads during executions, **Shared Memory** and **L1 cache are** shared between the threads running on a single SM, higher up is the **L2 cache** shared by all SMs, finally there is the **Global Memory** which is the largest memory on the GPU (the advertised 80 GB for a H100 for instance) but also the slowest to access and query.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/diving_primergpu2.svg)

Source: https://www.youtube.com/watch?v=ZQKMZIP3Fzg

The goal of GPU will be to run as many workloads as possible, in parallel, on the GPU cores, by taking advantage of this hierarchical organization of compute/memory.

A piece of code running on a core of the GPU is called a **kernel**. It can be written at a high-level in **CUDA** or **Triton** for instance, and is then compiled to Parallel Thread Execution, PTX, the low-level assembly used by NVIDIA GPUs.

To run the kernel, you will also need a specific code part, called **host code**, which is executed on the **CPU/host** and will take care of preparing data allocations and loading data and code.

```python
// Host code                
void vecAdd(float* h_A, float *h_B, float *h_c, int n) {
    // Allocate vectors in device memory
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

Host code for a CUDA kernel for adding two vectors. Adapted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/ and https://blog.codingconfessions.com/p/gpu-computing

```python
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

Device code containing the definition of the vector addition kernel adapted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/ and https://blog.codingconfessions.com/p/gpu-computing

Kernels are generally scheduled as follow:

- threads are grouped in **warps** of sizes of 32. All the threads in a warp are synchronized to execute instructions simultaneously but on different parts of the data.
- **warps** are grouped in larger **blocks** of more flexible size (e.g. size 256), each block still being assigned to a single SM. An SM may run several blocks in parallel, however, depending on the resources, not all the blocks may get assigned for execution immediately, some can be waitlisted waiting for resources.

The main thing to remember from these details is that there are various sizing and allocation constraints (size of the various memories, number of concurrent block and threads in the wraps) which need to be taken into account to use the GPU architecture in the most efficient way.

Most of the time you don’t need to go down to this level of precision and you can luckily reuse the kernels and code prepared by other members of the community. But in any case we want to give you a primer on how to get started with kernels!

### How to improve performance with Kernels ?

If you’re looking to add a new operation that lacks an optimized kernel or to speed up an existing PyTorch function, writing kernels from scratch might seem like the most direct route. However, creating high-performance CUDA kernels from scratch requires extensive experience and a steep learning curve. Generally a better way to get started is to leverage `torch.compile`, which dynamically optimizes PyTorch code by capturing your operations and generating lower-level, high-performance kernels in triton.

Let’s suppose you want to write a kernel for an activation function called Exponential Linear Unit:

ELU(x)={ex−1if x<0xif x≥0ELU(x)={ex−1x​if x<0if x≥0​

You can start by a simple pytorch implementation and then just add the `@torch.compile` decorator on top:

```python
@torch.compile
def elu(x, alpha=1.0):
    return torch.where(x < 0, alpha * (torch.exp(x) - 1), x)
```

The distinction between the compiled and non-compiled versions is striking, especially given that we only added a single decorator. This remarkable difference is illustrated in the graph below (N is the number of columns):

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/torch-compile-triton.png)

However, if this performance increase is insufficient, you can consider implementing Triton kernels. As a starting point, you can take a look at the triton kernel generated by @torch.compile . To do so, you simply need to set the environment variable `TORCH_LOGS` to `"output_code"`:

```bash
export TORCH_LOGS="output_code"
```

Once you run the Python script with the `@torch.compile` decorator, it will generate and output the corresponding Triton kernel, which, in this case, is:

```python
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 < tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tl.where(tmp2, tmp5, tmp0)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
```

To enhance readability, we can modify the variable names, add comments, and make slight adjustments (or ask an LLM to do it for us), as demonstrated below:

```python
@triton.jit
def elu_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create an array of indices for this block
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)[:]
    # Create a mask to ensure only valid indices are processed
    valid_mask = block_indices < num_elements
    # Load input values from the input pointer based on valid indices
    input_values = tl.load(input_ptr + block_indices, valid_mask)
    # Define the ELU parameters
    zero_value = 0.0  # Threshold for ELU activation
    negative_mask = input_values < zero_value
    exp_values = tl.math.exp(input_values)
    # Define the ELU output shift
    one_value = 1.0
    shifted_exp_values = exp_values - one_value

    output_values = tl.where(negative_mask, shifted_exp_values, input_values)

    # Store the computed output values back to the output pointer
    tl.store(output_ptr + block_indices, output_values, valid_mask)
```

Here, `tl.program_id(0)` provides a unique block ID, that we use to determine which section of data that block will process. Using this block ID, `block_start` calculates the starting index for each block’s section, while `block_indices` specifies the range of indices within that section. A `valid_mask` ensures that only indices within `num_elements` are processed, safely loading the data with `tl.load`. The ELU function is then applied, modifying values based on whether they're negative, and results are written back to memory with `tl.store`.

When we benchmark the generated kernel using `triton.testing.Benchmark` we have the following performance:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/torch-compile-triton-kernel.png)

This standalone kernel even demonstrates superior performance with smaller sizes compared to `@torch.compile` but this is likely just an artifact of the compilation time of `torch.compile`. In any case, instead of starting from scratch, remember that you can start from such generated kernels and focus your attention to optimizing its performance, saving you a lot of time in the process.

Even in Triton, sometimes, we cannot fully achieve the peak performance of the device due to the language limitations to handle low level details like shared memory and scheduling within streaming multiprocessors (SMs). Triton capabilities are restricted to blocks and scheduling of blocks across SMs. To gain an even deeper control, you will need to implement kernels directly in CUDA, where you will have access to all the underlying low-level details.

Moving down to CUDA, various techniques can be employed to improve the efficiency of kernels. We will just cover a few here: optimizing memory access patterns to reduce latency, using shared memory to store frequently accessed data, and managing thread workloads to minimize idle times.

Before we dive deeper in CUDA examples, let's summarize the tools we've seen that let us write kernel code to execute instructions on the GPU:

1. Pytorch: easy but slow
2. torch.compile: easy, fast, but not flexible
3. triton: harder, faster, and more flexible
4. CUDA: hardest, fastest, and flexiblest (if you get it right)

Let’s talk about one of the most frequent technique we can use in CUDA: optimizing memory access. The global memory in GPUs (the largest memory in our above graph) has a long latency and low bandwidth in comparison to the cache which often creates a major bottleneck for most applications. Efficiently accessing data from global memory can improve performance by a lot.

#### Memory Coalescing

To effectively utilize the bandwidth of global memory, it is essential to understand its architecture. In CUDA devices, global memory is implemented using DRAM.

Memory coalescing takes advantage of how DRAM delivers data in bursts, or ranges of consecutive memory locations, whenever a memory address is accessed. Each time a DRAM location is accessed, a sequence of consecutive locations, including the requested one, is read in parallel by multiple sensors in the DRAM chip. Once read, this data can then be quickly transferred to the processor as a burst. In CUDA, coalescing uses this burst behavior to maximize memory access efficiency by ensuring that threads in a warp—32 threads that execute the same instruction in lockstep (SIMD)—access consecutive memory locations. For instance, if thread 0 accesses location M, thread 1 accesses M + 1, thread 2 accesses M + 2, and so forth, the GPU hardware coalesces or combines these requests into one large, efficient access request for the DRAM burst, rather than handling each access individually.

Let’s take the example of matrix multiplication. A simple, straightforward implementation would have each thread compute a single element of the output matrix, like this:

```clike
__global__ void matmul_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}
```

Here’s an excellent visualization of the kernel from this [fantastic blogpost](https://siboehm.com/articles/22/CUDA-MMM):

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing.png)

However, when profiling this kernel with a tool like `ncu`, we can see issues, including low memory throughput and uncoalesced memory accesses.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing2.png) ![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing3.png)

The reason for this is that in this kernel, two threads in the same block with Thread IDs `(0, 0)` and `(1, 0)` (which will end up in the same warp) will both load from the same column of matrix `B` but different rows of matrix `A`. Since matrix elements are stored in row-major order (meaning row elements are in consecutive memory addresses, as shown in the figure below) thread `(0, 0)` will load A0,0A0,0​, and thread `(1, 0)` will load A1,0A1,0​ in the first iteration `i = 0`. These elements are not stored close to each other in memory, and this misalignment will be present at each iteration, thereby preventing memory accesses from being coalesced.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing4.png)

To improve the performances of our kernel we can change the way coordinates x and `y` are calculated to the following:

```clike
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
float tmp = 0.0;
for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
}
C[x * N + y] = tmp;
}
```

Instead of using a 2D block, we switch to a 1D block and redefine how we determine the values of `x` and `y`. In this new method, threads within the same warp (which have close `threadIdx.x` values) will share the same `x` value but have different `y` values. This means that they will load the same row of matrix `A` but different columns of matrix `B`. As a result, memory accesses can be coalesced for a row-major matrix.

When we profile our new kernel, we notice that the warning about uncoalesced memory accesses has disappeared, and **the GPU's memory throughput has increased by approximately 10 times**.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing5.png)

We also notice that the execution time of the kernel **decreases by 10x**! Amazing.

Now let's cover another technique you will often see mentioned in the litterature: **tiling**.

#### Tiling

Tiling is a technique that leverages _shared memory_ to optimize memory access patterns. As we mentioned above, the shared memory is a small, fast memory accessible by all threads within a block. It allows data to be reused by multiple threads, reducing the need to repeatedly load data from slower global memory.

In matrix multiplication for example, each thread in a block may need elements from two matrices, say A and B. If each thread independently loads the row and column it needs from global memory, we end up with many redundant loads, as multiple threads in a block will access overlapping data. Instead, we can use tiling to load a block (or tile) of A and B into shared memory just once, allowing all threads in that block to reuse the same shared data.

In the tiling approach, each iteration involves all threads within a block to cooperatively load two tiles—one from matrix A and another from matrix B —into shared memory. Specifically, threads load a tile of matrix A (of size `BLOCK_SIZE_M` by `BLOCK_SIZE_K`) and a tile of matrix B (of size `BLOCK_SIZE_K` by `BLOCK_SIZE_N`). Once the tiles are in shared memory, the threads perform matrix multiplication on these tiles, enabling efficient computation since all necessary data is quickly accessible. The results of the tile multiplication are stored in an accumulation matrix that holds intermediate results. After each iteration, the results from the current tile multiplication are added to this accumulation matrix, continuing until all tiles from both matrices have been processed.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tiling.png)

From [https://cnugteren.github.io/tutorial/pages/page4.html](https://cnugteren.github.io/tutorial/pages/page4.html)

Let's take a look at the important parts you need to understand from the implementation:

```clike
// Set pointers to the starting elements
A += blockRow * TILE_SIZE * K; // Start at row = blockRow, column = 0
B += blockCol * TILE_SIZE; // Start at row = 0, column = blockCol
C += blockRow * TILE_SIZE * N + blockCol * TILE_SIZE; // Start at row = blockRow, column = blockCol
float sum = 0.0;
// The outer loop moves through tiles of A (across columns) and B (down rows)
for (int tileIdx = 0; tileIdx < K; tileIdx += TILE_SIZE) {
sharedA[localRow * TILE_SIZE + localCol] = A[localRow * K + localCol];
sharedB[localRow * TILE_SIZE + localCol] = B[localRow * N + localCol];

// Ensure all threads in the block have completed data loading
__syncthreads();

// Shift pointers to the next tile
A += TILE_SIZE;
B += TILE_SIZE * N;

// Compute the partial dot product for this tile
for (int i = 0; i < TILE_SIZE; ++i) {
    sum += sharedA[localRow * TILE_SIZE + i] * sharedB[i * TILE_SIZE + localCol];
}
// Synchronize again to prevent any thread from loading new data
// into shared memory before others have completed their calculations
__syncthreads();
}
C[localRow * N + localCol] = sum;
```

For simplicity we consider a square shaped tile.

Each thread begins by loading one element from both **Matrix A** and **Matrix B** into shared memory. In this scenario, achieving coalesced memory access is straightforward, by assigning `threadIdx.x` as the **local column index (localCol)**, threads within the same warp will access adjacent elements of both matrices. After each thread in the block completes loading its elements into shared memory (ensured by calling `__syncthreads()`), they proceed to compute the dot product of the two tiles. Once the threads have iterated through all the tiles—horizontally for **Matrix A** and vertically for **Matrix B**—the resulting sum is stored in the corresponding location of **Matrix C**.

When benchmarking this kernel using ncu, we noticed that the memory throughput increased to 410 Gb / s, and the kernel execution time decreased by ~43% achieving a ~6.6 TFLOPs performance

#### Thread Coarsening

The tiling technique has significantly improved the performance of our kernel. However, when analyzing the warp states which quantify how many cycles were spent in each state, we observe the following:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/threadcoarsening.png)

The meaning of these cryptic state names can be found in [NVidia's profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference), in the **Warp Stall Reasons** section. There we can read that:

_`"smsp__pcsamp_warps_issue_stalled_mio_throttle`: Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure."_

So it seems warps are stalling waiting for shared memory accesses to return! To solve this issue we can apply a technique called **Thread Coarsening** which involves merging several threads into a single coarsened thread. This will significantly reduce shared memory accesses as each coarsened thread can handle multiple output elements.

Let's briefly go through a last important consideration when writing or improving custom kernels: **Minimizing Control Divergence**.

#### Minimizing Control Divergence

A Streaming Multiprocessor (SM) is built to execute all threads in a warp using the Single Instruction, Multiple Data (SIMD) model. This means that at any given moment, one instruction is fetched and executed simultaneously for all threads within the warp. When a warp is executed, the threads within it operate on different segments of the data but follow the same instruction, hence the name Single Instruction, Multiple Data. The primary advantage of SIMD is its efficiency; the control hardware responsible for instruction fetching and dispatching is shared among multiple execution units. This design minimizes the hardware overhead associated with control functions, allowing a greater portion of the hardware to focus on improving arithmetic throughput.

Control divergence occurs when threads within the same warp take different execution paths. For instance, if a conditional statement (like an `if` statement) leads to some threads executing one block of code while others execute a different block, the warp must serialize these executions, resulting in idle threads waiting for others to complete. To minimize control divergence, we need to design kernels to ensure that threads within the same warp follow the same execution path. This can be achieved by restructuring code to reduce branching, using data structures that ensure all threads follow similar execution paths, or employing techniques such as predication.

---

We have covered some of the main considerations when writing custom kernels and improving the performance and memory footprint of GPU operations. But there’s one more important concept before moving to a real example which is “fusing kernels”.

### Fused Kernels

In several places now we’ve mentioned how GPU and CPU operation can be asynchronous. In particular, the host code on the CPU can schedule workload on the GPU in a non-blocking way.

Non-blocking can be useful for overlapping communication and computation –as we saw many times along our journey– but can be extended to the more general idea of trying to avoid at all cost going back and forth between host and GPU kernel commands.

This idea is beautifully illustrated by [Horace He](https://horace.io/brrr_intro.html) in these diagrams:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fused_kernels1.png)

A sequence of kernels requiring back and forth between global memory and compute units

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fused_kernels2.png)

Instead of sending our triangle back to global memory just to read it back again, we instead just do all of our operations in one go.

How can we avoid this back and forth? Well the best way is to make our GPU as autonomous as possible. This is achieved by packing as many successive compute operations together in a single kernel for the GPU to run, called a “Fused Kernel”.

Fused kernel are especially efficient and simple to write for succession of point-like operations which are performed independently of each other on each input tokens. In this case, there is no point in bringing back computed values in Global Memory before moving them to SM memory and spinning up a new kernel. It’s much more efficient to keep all values locally until the succession of computation has been performed.

There are many places in a Transformer model where this "fusing" approach can be applied: every time we have a succession of point-wise operations e.g. in the computation involved in the Layer norms.

We now have all the understanding necessary to marvel at a true masterpiece of kernel engineering: **_Flash Attention_**

### Flash Attention 1-3

Flash attention was introduced by [Tri Dao](https://tridao.me/) and proposed to optimize the attention computations by writing custom CUDA kernels make them much faster *and* more memory efficient. The idea behind Flash Attention is to make efficient use of the various memories of the GPU to avoid relying too much on the slowest one: the global memory of the GPU.

Note that the global memory of the GPU is confusingly called the "High Bandwidth Memory", HBM 🫠

A basic implementation of the attention mechanism involve a lot of transfer between memory and workers. It requires materializing the S and P matrices in HBM which means that the results need to be sent to HBM and then back to SRAM for the next computations:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/flashattn.png)

Since bandwidth is much lower in HBM this introduces a severe bottleneck in the attention computation. Can we do better? Tri Dao says yes!

The key element is to compute the S matrices in small pieces which can fit in the smaller shared memory of the SM. But we can do even better and avoid materializing the very large S matrix all together in favor of keeping only the necessary statistics for computing the normalization factor of the softmax. So we can compute part of OO directly in one computation in SRAM rather than moving intermediate results back and forth. In this case, not even do we make use of the shared memory but we also release the memory bottleneck resulting from materializing one of the largest activation matrices in the model (at long context length), the attention matrix.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/flashattn2.png)

Source: FlashAttention paper

[13]

The idea of flash attention resolves so many bottlenecks in model training that it has quickly become the default way to perform attention in all transformers:

- By avoiding to materialize the S matrix we **reduce the memory burden of attention**
- We also remove a large part of the **naive impact of the S^2 cost of attention**

As a result as well, all variants of linear attention and sub-quadratic approaches to approximate attention –developed shortly after the invention of the transformers architecture– have been mostly put aside in favor of this exact and fast flash attention implementation and mechanism.

Following Flash-attention 1, two successive improved versions have been released by the same lab: Flash-attention 2 and 3. In comparison to Flash-attention 1, the improvements in Flash-attention 2 and 3 are less about the general attention mechanism than about tailoring its low level implementation more specifically to the GPU by (1) reducing the number of non-matmul operations as much as possible (2) partitioning carefully the workload among wraps and thread blocks (for Flash Attention 2) and carefully optimizing for FP8 and Tensor Core support on the latest Hopper (H100) architecture for Flash Attention 3.

Flash attention puts some restrictions on which attention patterns can be sped up. Check out [FlexAttention](https://pytorch.org/blog/flexattention/) which is a fast _and_ flexible variant.

Flash-Attention is a master demonstration of the breakthrough improvements that can come when you take into account the internal memory/compute design of current GPU accelerators.

---

The techniques described so far in this operation-fusion section have required us to implement modeling code changes and write custom kernels for certain operations in order to speed up training.

In the final section of our low-level dive in the compute operations themselves, we will take a look at a range of methods that are agnostic to the modeling code and can be used for any model and are so widely used that they have become a standard in the industry: **Mixed Precision Training**!

### Mixed Precision Training

In various sections along this book, we've talked about lower precisions formats and their impact on the memory requirements for storing activations, parameters and optimizer states. It's now time to dive deeper in the details of these formats and understand better their trade-offs, advantages and limitations.

Mixed Precision Training, as the name suggests, involves mixing different precisions when training. The default numerical precision of PyTorch tensors is single-precision floating point format or also called FP32 or float32 which means that every number stored takes up 32 bits or 4 bytes. The available bits to represent a number are divided into 3 parts:

- Sign: the first bit determines if the number is positive or negative
- Mantissa: determines the significant figures of a number
- Exponent: controls the magnitude of the number

![sign-mantissa-exponent.svg](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/sign-mantissa-exponent.svg)

The principle of floating point numbers can be easily illustrated by recalling the scientific notation of numbers, e.g. −5.734×107−5.734×107, where we first have the sign, followed by the mantissa an the exponent. As such we can represent numbers across a wide range of magnitudes with an adaptive precision. Although float32 is the default there is a range of floating point formats available in PyTorch:

|**Format**|**Total bits**|**Sign**|**Exponent**|**Mantissa**|
|---|---|---|---|---|
|float32|32|1|8|23|
|float16|16|1|5|10|
|bfloat16|16|1|8|7|
|float8 (e4m3)|8|1|4|3|
|float8 (e5m2)|8|1|5|2|

Note: You might be wondering where the “b” in bfloat16 comes from. The format was developed at Google Brain and thus the “b” stands for “brain”.

Reducing the total number of bits comes at a price (no free lunch here either), but we have some control over how to pay. Either we can sacrifice more bits on the mantissa or exponent. For this reason there exist also two float8 formats, named according to exponent and mantissa, to flexibly choose the most appropriate format. We can look at the possible range of numbers for each format:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/mixedprecision.png)

We can see that float32 spans 80 orders of magnitude and float16 sacrifices a lot of range while bfloat16 maintains the full range. The two float8 formats reduce the range even further where e5e2 can maintain float16 range and e4m3 has an even smaller ranger.

How come some formats are able to maintain the range and others not? Let’s investigate the resolution by plotting 10,000 points between 1 and 2. Each point will be rounded to the nearest representable number in each format:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/mixedprecision_2.png)

We can see here that bfloat16 maintained the range of float32 over float16 but did this with the cost of sacrificing more precision. In case of float8 the situation is even more dire as e4m3 can represent 7 and e5m2 only 3 number on the interval 1-2.

A common metric to measure a formats resolution is epsilon: the first representable number after 1.001.00. We can see that for the float32 format 10−410−4 is an upper bound (it’s actually 1.19−71.19−7). For float16 it is ~ 10−310−3 and for bfloat 10x higher still.

The idea of mixed precision training is to use some of these lower precisions formats while maintaining the performance of full precision training.

It turns out we **can’t** totally abandon float32 and usually will need to maintain some parts in full precision. This is why lower precision training is usually called **_mixed precision_** training.

Let’s now take a look at training models with 16 bits and then see if we can take it a step further all the way down to 8 bits.

#### FP16 and BF16 training

Naively switching all the tensors and operations to float16 unfortunately doesn’t work and the result is usually diverging losses. However, the original mixed precision training paper

[2]

 came up with three tricks to match float32 trainings:

1. **FP32 copy of weights**: There are two possible issues with float16 weights. During training some of the weights can become very small and will be rounded to 0. However, even if the weights themselves are not close to zero, if the updates are very small the difference in magnitude can cause the weights to underflow during the addition. Once the weights are zero they will remain 0 for the rest of training as there is no gradient signal coming through anymore.
2. **Loss scaling**: We have a similar issue with the gradients as well as gradients tend to be much smaller than 1 and are thus at risk to underflow. A simple, yet effective, strategy is to scale the loss before the backward pass and unscale the gradients after the backward pass. This ensures that there is no underflow during the backward pass and the scaling is not affecting training as we unscale before processing the gradients further (e.g. clipping) and the optimization step.
3. **Accumulation**: Finally, when performing certain arithmetic operations in 16-bit precision such as averages or summations, we can also face under or overflows. A solution is then to accumulate intermediate results in float32 during the operation and only cast the final result back to 16 bit precision.

With these techniques, we can get a stable training while benefitting from a higher throughput due to the faster, lower precision arithmetic operations. Naturally, as a curious reader –and by now slightly addicted to maximizing the throughput– you may ask the question: can we go further and faster than 16-bit precision?

Maybe!

#### FP8 pretraining

Even if we perfectly overlap communication with computation, we always eventually run into the low level theoretical FLOPS limit of the hardware itself, i.e. the efficiency of each individual operation on our hardware. This is where numerical precision becomes crucial. For instance, on NVIDIA's H100 GPU, FP8 matrix multiplications (GEMM operations) achieve twice the theoretical FLOPS of bfloat16, making lower-precision training an attractive path for further optimization.

Recent research - including FP8-LM

[14]

, torchao

[15]

, and DeepSeek-V3

[7]

 - has demonstrated the potential of FP8 training for large-scale models. Still, FP8 pretraining introduces a significant challenge: stability. At lower precision, numerical instability often leads to loss divergence, making it difficult to match the accuracy of higher-precision training.

We know that instability increases as learning rates rise for a fixed model size

[16]

, making FP8 pretraining particularly tricky.

Here is an example of a typically divergent loss curve for FP8 training:

The first, successful, very large scale training with FP8 mixed precision was publicly reported on DeepSeek-V3. The authors carefully analyzed each operation of the forward pass (Fprop) as well as the activation (Dgrad) and weight (Wgrad) backward pass. Similar to BF16 mixed precision training, some aggregation and master weights are kept in higher precision while the operations themselves are performed in FP8.

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fp8_diagram.png)

In order to switch from high precision (e.g. FP32 or BF16) to lower precision (e.g. FP16 or FP8) with smaller range, we need to normalize the range of activation values, for instance by computing their absolute maximum. DeepSeek-V3 further introduced a specific quantization scheme where the ranges are normalized per tile: 1x128 for inputs/activations and 128x128 for weights and scale elements. This makes the normalization less strongly impacted by outlier values in the activations. There is a number of additional tricks they proposed to further reduce the memory and communication footprint which you can follow in section 3.3. of the DeepSeek-V3 technical report

[7]

.

Here’s a summary of a few known approaches to FP8 training:

||GEMM's precision|Master model weights|Accumulated gradients|Model weights|Gradients|Optimizer States|Total Memory|
|---|---|---|---|---|---|---|---|
|bfloat16 with fp32 mixed precision baseline|bf16|fp32|fp32|bf16|bf16|fp32 + fp32|4 + 4 + 2 + 2 + 4 + 4 = 20 bytes|
|Above without FP32 grad accumulation|bf16|fp32|n/a|bf16|bf16|fp32 + fp32|4 + 2 + 2 + 4 + 4 = 16 bytes|
|Transformer Engine|fp8|n/a|n/a|fp32|fp32|fp32 + fp32|4 + 4 + 4 + 4 = 16 bytes (20% reduction)|
|FP8-LM's O3 level|fp8|fp16|fp16|fp8|fp8|fp8 + fp16|2 + 2 + 1 + 1 + 1 + 2 = 9 bytes (55%)|
|DeepSeek-V3|fp8|fp32|fp32|fp8|bf16|bf16 + bf16|4+4+1+2+2+2 = 15 (25%)|
|nanotron's FP8|fp8|bf16|fp32|fp8|fp8|fp8 + fp8|2 + 4 + 1 + 1 + 1 + 1 = 10 bytes (50%)|

Overall, FP8 remains –in early 2025– an experimental technique and methods are still evolving. Given its obvious benefits, it will likely become the standard and soon replace bf16 mixed-precision. To follow an open-source implementations of FP8 training techniques, please head to the nanotron’s implementation in [this PR](https://github.com/huggingface/nanotron/pull/70).

Projecting further into the future, Blackwell, the next generation of NVIDIA chips, [have been announced](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) to support FP4 training, further speeding up training but without a doubt also introducing a new training stability challenge.

---

This last section concluded our long journey in the land of fast and large model training on tens to thousands of GPUs. Time to slowly bring our GPU cluster to rest and take a step back to conclude on all we've learned along the way.

## Conclusion

Congratulations, dear reader, you made it to the end! We've completed quite a journey: we started from understanding how to train a simple model on a single GPU, all the way to mastering all the intricate techniques used to efficiently train massive language models like Llama-405B and DeepSeek-V3 on thousands of GPUs. By now, you can read a diagram, like Llama-3's 4D parallel setup, with (relative) ease:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/conclusion_llama3_parallelism.png)

Orchestrating large clusters of GPUs to train LLMs efficiently is no easy feat. We learned how to optimize computations and communications between GPUs such that they run with maximum utilization at all times. It involves choosing the right parallelization strategy for a given model and cluster size, overlapping communication and computation where possible, and writing custom kernels that take into account the hardware layout to perform an operation as fast as possible on the GPU.

You might still believe that this knowledge is a bit niche and only concerns the small set of people that pretrain LLMs. Historically, that may have been true, but as both the [AI builder community](https://huggingface.co/) and model sizes are growing rapidly, the community of people using distributed techniques for inference, fine-tuning and training is increasing exponentially as well making distributed training setups more and more common. Diving deeper into all things distributed might thus prove very timely.

This has been a long learning journey, but not just for you! Running thousands of benchmarks on a GPU cluster was more challenging than we anticipated and we want to share a few highlights of our own learning experience as well.

### So, what’s next?

You now have good overview of the main distributed training concepts but at the same time we just scratched to surface of several of these tools and techniques. There are many ways to dive deep into a subject but here are some steps that we recommend:

- Carefully read some of the landmark or very recent papers. You can find a very extenside list of the most impactful papers, blog posts and books in [References](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#references).
- Start from scratch and implement an algorithm yourself. Often a method only fully “clicks” if you implemented it yourself.
- Dive into one of the widely used frameworks and start contributing: fix bugs, answer issues, or implement a new feature. That’s the best way to get in any ML field!

We hope this book helps you get started in distributed training and that you will train the next generation of awesome models to the hum of your GPU cluster!

---

**One last word** for our first readers. We're so happy with this writing piece that we've decided to distribute a limited number of physical printed editions of it as a gift for our first readers.

If you are among the first 50 people to fill in your email address below, we'll contact you later in the year to send you a real physical edition once we've formatted it as a printed copy.

We expect the book to be around 100-150 pages and to cover the same content as the blog post but we may also decide to shorten or lengthen it depending on what make sense as a printed object.

To get your physical copy, please fill in your email address in the following [google form](https://forms.gle/e1GkAShUCtgcwnne8).

Whether you are one of our first readers or coming much later to this blog post, we've very happy to see that you enjoyed this sharing of knowledge. May the force of open-source and open-science always be with you.

### Acknowledgements

We thank [Elie](https://huggingface.co/eliebak) for conducting thorough reviews and creating the audio components using NotebookLM. Special thanks to [Hynek](https://huggingface.co/hynky) for optimizing the frontend performance. We also thank [Simon](https://huggingface.co/sbrandeis) for resolving some issues on the hub.

### Discussion page

If you want to discuss the content of this blog post, ask questions, propose changes or just say hi, please open a thread on the [discussion page](https://huggingface.co/spaces/nanotron/ultrascale-playbook/discussions).

## References

### Landmark LLM Scaling Papers

[**Megatron-LM**](https://arxiv.org/abs/1909.08053)

Introduces tensor parallelism and efficient model parallelism techniques for training large language models.

[**Megatron-Turing NLG 530B**](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)

Describes the training of a 530B parameter model using a combination of DeepSpeed and Megatron-LM frameworks.

[**PaLM**](https://arxiv.org/abs/2204.02311)

Introduces Google's Pathways Language Model, demonstrating strong performance across hundreds of language tasks and reasoning capabilities.

[**Gemini**](https://arxiv.org/abs/2312.11805)

Presents Google's multimodal model architecture capable of processing text, images, audio, and video inputs.

[**Llama 3**](https://arxiv.org/abs/2407.21783)

The Llama 3 Herd of Models

[**DeepSeek-V3**](https://arxiv.org/abs/2412.19437v1)

DeepSeek's report on architecture and training of the DeepSeek-V3 model.

### Training Frameworks

[**Nanotron**](https://github.com/huggingface/nanotron)

Our framework for training large language models featuring various parallelism strategies

[**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM)

NVIDIA's framework for training large language models featuring various parallelism strategies.

[**DeepSpeed**](https://www.deepspeed.ai/)

Microsoft's deep learning optimization library featuring ZeRO optimization stages and various parallelism strategies.

[**FairScale**](https://github.com/facebookresearch/fairscale/tree/main)

PyTorch extension library for large-scale training, offering various parallelism and optimization techniques.

[**ColossalAI**](https://colossalai.org/)

Integrated large-scale model training system with various optimization techniques.

[**torchtitan**](https://github.com/pytorch/torchtitan)

A PyTorch native library for large model training.

[**GPT-NeoX**](https://github.com/EleutherAI/gpt-neox)

EleutherAI's framework for training large language models, used to train GPT-NeoX-20B.

[**LitGPT**](https://github.com/Lightning-AI/litgpt)

Lightning AI's implementation of state-of-the-art open-source LLMs with focus on reproducibility.

[**DiLoco**](https://github.com/PrimeIntellect-ai/OpenDiLoCo)

Training language models across compute clusters with DiLoCo.

[**torchgpipe**](https://github.com/kakaobrain/torchgpipe)

A GPipe implementation in PyTorch.

[**OSLO**](https://github.com/EleutherAI/oslo)

OSLO: Open Source for Large-scale Optimization.

### Debugging

[**Speed profiling**](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Official PyTorch tutorial on using the profiler to analyze model performance and bottlenecks.

[**Memory profiling**](https://pytorch.org/blog/understanding-gpu-memory-1/)

Comprehensive guide to understanding and optimizing GPU memory usage in PyTorch.

[**Memory profiling walkthrough on a simple example**](https://huggingface.co/blog/train_memory)

Visualize and understand GPU memory in PyTorch.

[**TensorBoard Profiler Tutorial**](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)

Guide to using TensorBoard's profiling tools for PyTorch models.

### Distribution Techniques

[**Data parallelism**](https://siboehm.com/articles/22/data-parallel-training)

Comprehensive explanation of data parallel training in deep learning.

[**ZeRO**](https://arxiv.org/abs/1910.02054)

Introduces Zero Redundancy Optimizer for training large models with memory optimization.

[**FSDP**](https://arxiv.org/abs/2304.11277)

Fully Sharded Data Parallel training implementation in PyTorch.

[**Tensor and Sequence Parallelism + Selective Recomputation**](https://arxiv.org/abs/2205.05198)

Advanced techniques for efficient large-scale model training combining different parallelism strategies.

[**Pipeline parallelism**](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism)

NVIDIA's guide to implementing pipeline parallelism for large model training.

[**Breadth first Pipeline Parallelism**](https://arxiv.org/abs/2211.05953)

Includes broad discussions around PP schedules.

[**All-reduce**](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

Detailed explanation of the ring all-reduce algorithm used in distributed training.

[**Ring-flash-attention**](https://github.com/zhuzilin/ring-flash-attention)

Implementation of ring attention mechanism combined with flash attention for efficient training.

[**Ring attention tutorial**](https://coconut-mode.com/posts/ring-attention/)

Tutorial explaining the concepts and implementation of ring attention.

[**ZeRO and 3D**](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism)

DeepSpeed's guide to understanding tradeoffs between ZeRO and 3D parallelism strategies.

[**Mixed precision training**](https://arxiv.org/abs/1710.03740)

Introduces mixed precision training techniques for deep learning models.

[**Visualizing 6D Mesh Parallelism**](https://main-horse.github.io/posts/visualizing-6d/)

Explains the collective communication involved in a 6D parallel mesh.

### Hardware

[**Fire-Flyer - a 10,000 PCI chips cluster**](https://www.arxiv.org/abs/2408.14158)

DeepSeek's report on designing a cluster with 10k PCI GPUs.

[**Meta's 24k H100 Pods**](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)

Meta's detailed overview of their massive AI infrastructure built with NVIDIA H100 GPUs.

[**Semianalysis - 100k H100 cluster**](https://www.semianalysis.com/p/100000-h100-clusters-power-network)

Analysis of large-scale H100 GPU clusters and their implications for AI infrastructure.

[**Modal GPU Glossary**](https://modal.com/gpu-glossary/readme)

CUDA docs for human

### Others

[**Stas Bekman's Handbook**](https://github.com/stas00/ml-engineering)

Comprehensive handbook covering various aspects of training LLMs.

[**Bloom training chronicles**](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)

Detailed documentation of the BLOOM model training process and challenges.

[**OPT logbook**](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)

Meta's detailed logbook documenting the training process of the OPT-175B model.

[**Harm's law for training smol models longer**](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)

Investigation into the relationship between model size and training overhead.

[**Harm's blog for long context**](https://www.harmdevries.com/post/context-length/)

Investigation into long context training in terms of data and training cost.

[**GPU Mode**](https://www.youtube.com/@GPUMODE/videos)

A GPU reading group and community.

[**EleutherAI Youtube channel**](https://youtube.com/playlist?list=PLvtrkEledFjqOLuDB_9FWL3dgivYqc6-3&si=fKWPotx8BflLAUkf)

ML Scalability & Performance Reading Group

[**Google Jax Scaling book**](https://jax-ml.github.io/scaling-book/)

How to Scale Your Model

[**@fvsmassa & @TimDarcet FSDP**](https://github.com/facebookresearch/capi/blob/main/fsdp.py)

Standalone ~500 LoC FSDP implementation

[**thonking.ai**](https://www.thonking.ai/)

Some of Horace He's blogposts - Making GPUs go BRRR..

[**Aleksa's ELI5 Flash Attention**](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

Easy explanation of Flash Attention

[**TunibAI's 3D parallelism tutorial**](https://github.com/tunib-ai/large-scale-lm-tutorials)

Large-scale language modeling tutorials with PyTorch.

## Appendix

### A0: Parallel Programming Crash Course

Throughout the blogpost we scale LLM training from one to hundreds of GPUs. This will require the communication and synchronization of weights, gradients, and data between all the machines. There’s a set of distributed patterns to achieve exactly that called **_collective operations_**. In this section we’ll do a small crash course of all the operations like _Broadcast, AllReduce, Scatter_ and more. Let’s dive in!

The general setup is that we have a number of independent nodes which could be CPU cores, GPUs, or compute nodes. Each performs some computation and then we want to communicate the result or parts of it to the other nodes for the next computation step (t+1).

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_general.png)

Maybe we need to send the result from one node to all other nodes, or we need to sum all the intermediate results from each node to report the overall result. Usually, there is one node with an elevated status that plays a central role, here denoted with `root` that is the target or source of some operations. Let’s start with one of the simplest primitives: a broadcast operation.

#### Broadcast

A very common pattern is that you have some data on Node 1 and you want to share it with all the other nodes so they can do some computation with the data. The broadcast operation does just that:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_broadcast.png)

Collective operations are natively provided by PyTorch so we can easily write a small example that demonstrates how broadcasting works. We first need to initialize a process group with `dist.initi_process_group` which sets up the communication backend (we’ll talk about NCCL later), it determines how many workers (aka nodes) exists and assigns a rank to each one (which we can get with `dist.get_rank`). Finally, it establishes a connection between the workers.

To showcase the `dist.broadcast` operation, let's create a tensor with non-zero values on `rank=0` and tensors full of zeros on the other workers. We then distribute the `rank=0` tensor to all other ranks with `dist.broadcast(tensor, src=0)` :

```python
import torch
import torch.distributed as dist

def init_process():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    
def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before broadcast on rank {dist.get_rank()}: {tensor}")
    dist.broadcast(tensor, src=0)
    print(f"After broadcast on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_broadcast()
```

You can run the above script with `torchrun --nproc_per_node=3 dist_op.py` (you’ll need 3 GPUs for this or change `nproc_per_node` accordingly) and you should see the following output:

```python
Before broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0')
Before broadcast on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')
Before broadcast on rank 2: tensor([0., 0., 0., 0., 0.], device='cuda:2')

After broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0')
After broadcast on rank 1: tensor([1., 2., 3., 4., 5.], device='cuda:1')
After broadcast on rank 2: tensor([1., 2., 3., 4., 5.], device='cuda:2')
```

Great, seems like it works as expected. Note that the rank messages can be printed out of order as we have no control over which print statement is executed first (we ordered them here for readability). Now let’s move on to the Reduce and AllReduce patterns!

#### Reduce & AllReduce

Reduce patterns are among the most fundamental patterns in distributed data processing. The idea is that you want to combine the data present on each node through a function `f()` which can be for instance summation or averaging. In the Reduce paradigm the result is sent to the root node only, whereas in the AllReduce case the result is broadcasted to all nodes:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_reduce_allreduce.png)

Of course no magic “free flying” node that can perform this operation and generally each node does a partial computation in a ring or tree structure of the nodes. Here is a simple example: let’s say we need to compute a sum of numbers on each nodes and our nodes are connected in a ring pattern. The first node sends its number to a neighbour which adds its number to the received number before forwarding it to the next neighbour. At the end of a round along the ring of nodes, the first node will receive the total sum.

Here’s the code to run a simple Reduce operation summing the tensors, we specify the operation to use with `op=dist.ReduceOp.SUM` (you can find more information on the supported operations in the [Pytorch docs](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)):

```python
def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before reduce on rank {dist.get_rank()}: {tensor}")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"After reduce on rank {rank}: {tensor}")
    
init_process()
example_reduce()
```

Note that in the Reduce operation only the tensor on the `dst` node is updated:

```python
Before reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0')
After reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
After reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')
```

Similarly we can perform an AllReduce (we don’t need to specify a destination in this case):

```python
def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_all_reduce()
```

In this case the result is available on all nodes:

```python
Before all_reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After all_reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0')
After all_reduce on rank 1: tensor([6., 6., 6., 6., 6.], device='cuda:1')
After all_reduce on rank 2: tensor([6., 6., 6., 6., 6.], device='cuda:2')
```

Now let’s turn to our next distributed communication operation. In many real cases, each node individually perform many complex computations and we need to share the final results among nodes. Gather and AllGather are the operations we want to use in this case. Let’s take a look!

#### Gather & AllGather

Gather and AllGather are quite similar to the Broadcast in that they allow distributing data among node without modification. The main difference to Broadcast is that there is not one value we need to share from one node to all other nodes but each node has an individual chunk of data that we want to either gather all data on one node (in case of Gather) or gather all data on all nodes (in the case of AllGather). A picture being worth 1000 words, let’s take a look:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_gather_allgather.png)

Note that the dashed lines indicate that some data actually doesn’t move at all (since it’s already present on the node).

In the case of the gather operation we need to prepare a container objects where the gathered tensors can be stored in this example the `gather_list`:

```python
def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    if dist.get_rank() == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()
            for _ in range(dist.get_world_size())
            ]
    else:
        gather_list = None
    print(f"Before gather on rank {dist.get_rank()}: {tensor}")
    dist.gather(tensor, gather_list, dst=0)
    if dist.get_rank() == 0:
        print(f"After gather on rank 0: {gather_list}")
    
init_process()
example_gather()
```

And we see that the `gather_list` indeed contains the tensors of all ranks:

```python
Before gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before gather on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After gather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                         tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                         tensor([3., 3., 3., 3., 3.], device='cuda:0')]
```

The only thing we need to change for the AllGather example is that every node will need a placeholder for the results:

```python
def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda()
        for _ in range(dist.get_world_size())
        ]
    print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
    dist.all_gather(gather_list, tensor)
    print(f"After all_gather on rank {dist.get_rank()}: {gather_list}")
    
init_process()
example_all_gather()
```

And indeed we can see that now each node has all the data:

```python
Before all_gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_gather on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After all_gather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:0')]
After all_gather on rank 1: [tensor([1., 1., 1., 1., 1.], device='cuda:1'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:0')]
After all_gather on rank 2: [tensor([1., 1., 1., 1., 1.], device='cuda:2'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:2'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:2')]
```

Now what about the inverse of a gather? In this case we would have all the data on one node and want to distribute/slice it among node, possibly with some intermediate processing? We can use the Scatter, or in the case of an operation in between a Reduce Scatter pattern:

#### Scatter & ReduceScatter

As the name subtly suggests, the goal of the Scatter operation is to take data on one node and distribute slices of it to all other nodes. It’s thus different from the Broadcast operation which copy data without slicing and it’s the logical the inverse of the Gather operation.

The ReduceScatter pattern is slightly more complex: imagine you apply an operation like in the Reduce case but instead of moving the result to just one node we also distribute it evenly to all nodes:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_scatter_reducescatter.png)

The Scatter operation is written in code as the opposite of the Gather: instead of preparing a list of tensors as target we prepare the source data as a list of tensors we want to distribute. We also need to specify the `src`:

```python
def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
            ]
        print(f"Rank 0: Tensor to scatter: {scatter_list}")
    else:
        scatter_list = None
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_scatter()
```

As a result we can see how the empty tensors got filled with the contents of the `scatter_list`

```python
Rank 0: Tensor to scatter: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                            tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                            tensor([3., 3., 3., 3., 3.], device='cuda:0')]
Before scatter on rank 0: tensor([0., 0., 0., 0., 0.], device='cuda:0')
Before scatter on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')
Before scatter on rank 2: tensor([0., 0., 0., 0., 0.], device='cuda:2')

After scatter on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
After scatter on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
After scatter on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')
```

Let’s create more interesting data to demonstrate the ReduceScatter logic: on each node we create a list of 2-elements vector on each node with a power exponent and an offset function of the node rank (it’s a bit hard to imagine so just look below for an example):

```python
def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()**(j+1) 
        for j in range(world_size)
        ]
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After ReduceScatter on rank {rank}: {output_tensor}")    
    
init_process()
example_reduce_scatter()
```

Let’s print the pattern of data that we created. We also immediately see the ReduceScatter pattern: the first rank received the sum of the first tensor from each node, and the second rank contains the sum of the second tensor on each node and so on:

```python
Before ReduceScatter on rank 0: [tensor([1., 2.], device='cuda:0'),
											 tensor([1., 4.], device='cuda:0'),
											 tensor([1., 8.], device='cuda:0')]
Before ReduceScatter on rank 1: [tensor([2., 4.], device='cuda:1'),
                                 tensor([ 4., 16.], device='cuda:1'),
                                 tensor([ 8., 64.], device='cuda:1')]
Before ReduceScatter on rank 2: [tensor([3., 6.], device='cuda:2'),
                                 tensor([ 9., 36.], device='cuda:2'),
                                 tensor([ 27., 216.], device='cuda:2')]

After ReduceScatter on rank 0: tensor([ 6., 12.], device='cuda:0')
After ReduceScatter on rank 1: tensor([14., 56.], device='cuda:1')
After ReduceScatter on rank 2: tensor([ 36., 288.], device='cuda:2')
```

Let's have a quick look at a common implementation of AllReduce that uses ReduceScatter and AllGather: Ring AllReduce.

#### A quick focus on Ring AllReduce

**_Ring AllReduce_** is one specific implementation of AllReduce, optimized for scalability. Rather than all devices communicating with each other directly, which could create communication bottlenecks, Ring All-Reduce can be broken down into two key steps: ReduceScatter and AllGather. Here's how it works:

1. **ReduceScatter**

- Each device splits its data (e.g., gradients) into chunks and sends one chunk to its neighbour. Simultaneously, each device receives a chunk from its other neighbour.
- As each device receives a chunk, it adds (reduces) its corresponding chunk to the received one.
- This process continues around the ring until each device holds a partially reduced chunk, representing a sum of the gradients across all devices for that chunk.

3. **AllGather**

- Now, each device needs to collect the fully reduced chunks from other devices.
- The devices start sending their reduced chunks to neighbours.
- Each device forwards the chunks it receives until every device has all the fully reduced chunks, giving each device the complete, summed-up gradient.

Let’s illustrate this with the following gifs, where we have 5 GPUs, each with a tensor of length 5. The first animation shows the ReduceScatter step, where, at the end, each GPU receives the reduced results for a specific chunk of data (orange rectangle).

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_reduce_scatter.gif)

The next animation shows the AllGather step, where, at the end, each GPU obtains the full results of the AllReduce operation:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_all_gather.gif)

You may have noticed that each of the NN GPUs sends and receives values N−1N−1 times during both the reduce-scatter and all-gather steps. Each GPU sends KNNK​ values per transfer, where KK is the total number of values in the array being summed across the GPUs. Therefore, the total amount of data transferred to and from each GPU is 2×(N−1)×KN2×(N−1)×NK​. When NN (the number of GPUs) is large, the total amount of data transferred to and from each GPU is approximately 2×K2×K, where KK is the total number of parameters.

**There are two key things to keep in mind for AllReduce:**

1. The communication cost for AllReduce is approximately 2xK2xK when NN (the number of GPUs) is large.
2. An AllReduce operation can be broken down into a reduce-scatter followed by an all-gather. The communication cost for these two operations is half that of the AllReduce, which is approximately KK.

As we can see this implementation can make efficient use of even a limited bandwidth between nodes.

We now have seen the main building block of distributed operations but before we see them in action let’s have a look at a special operation used for synchronization: the Barrier.

#### Barrier

The Barrier is a simple operation to synchronize all nodes. A barrier is not lifted until all nodes have reached it. Then only are they allowed to continue with further computations:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_barrier.png)

We can easily simulate delayed nodes by setting up a different sleep time on each node and see how long it takes for all of them to pass the barrier:

```python
def example_barrier():
    rank = dist.get_rank()
    t_start = time.time()
    print(f"Rank {rank} sleeps {rank} seconds.")
    time.sleep(rank)  # Simulate different processing times
    dist.barrier()
    print(f"Rank {rank} after barrier time delta: {time.time()-t_start:.4f}")
    
init_process()
example_barrier()
```

We can see that although the first rank didn’t sleep at all it also took it 2sec to pass the barrier:

```python
Rank 0 sleeps 0 seconds.
Rank 1 sleeps 1 seconds.
Rank 2 sleeps 2 seconds.

Rank 0 after barrier time delta: 2.0025
Rank 1 after barrier time delta: 2.0025
Rank 2 after barrier time delta: 2.0024
```

We need to be careful with synchronizing all nodes like this, as this defeat the purpose of parallel independent operations and might thus slow down the whole processing. In many situations it can be just fine if a fast node already starts processing the next job as the fast node could be slower in a next iteration therefore evening out the delay over the whole process.

Before turning to practical distributed training implementations, let’s first solve a mystery: what the heck is NCCL?

#### NCCL: NVIDIA Collective Communications Library

When training large models on many GPUs we may sometimes strike gold but we will always encounter nickel (or NCCL 🥁)! What’s is that?

There are several libraries that implement collective communication and are support by PyTorch: there’s the classic **_MPI_** (Message Passing Interface), there’s **_Gloo_** by Meta, and finally there is `NCCL` (NVIDIA Collective Communications Library). They all provide similar functionality in terms of collective communication patterns but are optimized for different hardware setups; NCCL is designed to serve GPU-GPU communication efficiently while MPI and Gloo are setup for CPU-CPU or CPU-GPU communication. PyTorch provides a [great guide](https://pytorch.org/docs/stable/distributed.html#which-backend-to-use) to decide which one to use:

- GPU training: use NCCL
- CPU training: use Gloo

There are a few finer points in the decision tree that we leave to the reader to explore in the PyTorch guide referenced above.

Now that we covered the fundamental operations for distributed training and you should now be ready to follow the blog post easily.

### A1: Distributed Training Profiling

#### Kernels

Let's begin by assuming for now that the kernels are already integrated into PyTorch. As a simple example, we can look at the Layer Normalization function implemented in PyTorch as `torch.nn.functional.layer_norm`. There are several methods to profile the kernel that underlies this function. The most straightforward approach might be to use the Python `time` module. However, since CUDA operations are asynchronous, measuring time with this method will only capture the overhead associated with launching the kernel in Python, rather than the actual execution time of the kernel itself.

To address this, we can utilize `torch.cuda.Event` for accurate timing and employ the `torch.cuda.synchronize()` directive to ensure we wait for the kernel execution to complete. This approach is demonstrated in the following snippet:

```python
def profile_pytorch(func, input):
    # Create CUDA events to track time. CUDA operations are asynchronous,
    start = torch.cuda.Event(enable_timing=True)  # Event to mark the start time
    end = torch.cuda.Event(enable_timing=True)    # Event to mark the end time
    # Warmup to eliminate any overhead from the first run, which might not reflect 
    # the actual performance.
    for _ in range(10):
        func(input)
    # Record the start time before executing the function
    start.record()  
    func(input)  # Call the function we want to profile
    # Record the end time after the function has completed
    end.record()  
    # Synchronize the CUDA operations to ensure all operations are completed
    # before measuring the elapsed time.
    torch.cuda.synchronize()  
    # Calculate and return the elapsed time in milliseconds.
    return start.elapsed_time(end)
```

A more effective approach to profiling is to utilize the PyTorch Profiler, as explained previously. For example, consider the following code:

```python
import torch
import torch.nn.functional as F

def pytorch_layer_norm(input):
    return F.layer_norm(input, input.size()[1:])

a = torch.randn(10000, 10000).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,  # Profile CPU activities
        torch.profiler.ProfilerActivity.CUDA,  # Profile CUDA activities
    ],
    # Define a schedule for the profiler
    schedule=torch.profiler.schedule(
        wait=1,      # Wait for 1 iteration before starting to profile
        warmup=3,    # Warm up for 3 iterations to stabilize performance
        active=2,    # Profile for 2 active iterations
        repeat=1,    # Repeat the profiling schedule once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('.'),
    
) as p:
    for iter in range(10):
        pytorch_layer_norm(a)
        p.step()

# Print a table of the profiling results, sorted by total CUDA time, limited to the top 10 entries
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
```

This would print aggregated profiling results sorted by the total CUDA time, and the output would be:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_kernels.png)

You can also try to inspect the trace as we previously mentioned on `chrome://tracing/`

💡 Tip

If you're new to this tool, you can navigate the trace by using the right and left arrow keys. Additionally, you can zoom in and out by holding the **Alt** key while scrolling left or right with your mouse.

After zooming in, you can observe the flow of operations when calling `layer_norm` in this trace:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_profile_trace.png)

The sequence begins in the CPU (the upper section) with `aten::layer_norm`, progressing to `aten::native_layer_norm`, and then transitioning to `cudaLaunchKernel`. From there, we move on to the GPU, where the `vectorized_layer_norm_kernel` kernel is called.

📝 Note

You can enable memory profiling by setting `profile_memory` to `True` in the profiler. However, this can lead to more complex traces.

While the PyTorch Profiler offers a quick performance overview, **NVIDIA Nsight Compute (ncu)** provides deeper insights into GPU performance, including detailed execution times and memory usage for each kernel. To run the profiler it's very simple:

```bash
ncu --set full python layer_norm.py
```

Where `layer_norm.py` is a straightforward file that executes the layer normalization function. This command will generate log outputs, but a more effective way to visualize the results is by setting the output flag:

```bash
ncu --set full -o output python layer_norm.py
```

and open the file `output.ncu-rep` with Nsight Compute, you will have a view that looks like this:

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_ncu.png)

With clear warnings about compute and memory utilization, and how to make the kernel better in balancing compute and memory and achieve maximal occupancy.

#### CPP extension

If the kernel you want to profile isn't already integrated into PyTorch, you can use PyTorch's `cpp_extension` module to easily compile and run custom CUDA code. The process is straightforward—just create your CUDA kernel in a `.cu` file, and use the `load` function from the `cpp_extension` module to load it in Python.

The `.cu` file would like this for a simple `add` kernel:

```clike
#include 
#include 
#include 

__global__ void add_kernel(float* x, float* y, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = x[index] + y[index];
    }
}

void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor output) {
    int threads = 1024;
    int blocks = (x.size(0) + threads - 1) / threads;

    add_kernel<<>>(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.size(0));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Vector addition (CUDA)");
}
```

And the python file to load the kernel:

```python
import torch
from torch.utils.cpp_extension import load

# Load and compile the CUDA extension
vector_add = load(
    name="vector_add",
    sources=["add_kernel.cu"],
    verbose=True
)

# Define input tensors
size = 10000
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty(size, device='cuda')

# Run the CUDA kernel
vector_add.add_cuda(x, y, output)
```

Using this method, you can profile the custom CUDA kernel just as we demonstrated earlier with PyTorch's profiler or NVIDIA tools.

### A2: Typical Scales in LLM Training

Let's get a feel for the typical sizes of things in LLM training. When we talk about memory or compute, we're often counting "elements" - think of these as numbers in tensors. To get the actual memory in bytes, you'll need to multiply by the size of each number (e.g., 2 bytes for bf16, 4 bytes for fp32).

Here are some quick ballpark figures:

- **Input tokens:** For each batch, we process seq⋅mbsseq⋅mbs tokens, where mbs is the micro batch size and seq is the sequence length.
- **Activations (hidden states):** For a single layer, the hidden state tensor is of size seq⋅mbs⋅hseq⋅mbs⋅h elements.
- **Model weights and gradients:** Each weight matrix in your model (like in linears) is about h2h2 elements. This is per weight matrix. Gradients have the same size as weights.
- **Optimizer states:** For each weight matrix (of elements h2h2), if you're using an optimizer like Adam with mixed precision training, it keeps momentum and variance states in fp32 precision (2⋅h22⋅h2), plus master weights in fp32 (h2h2). So total optimizer states will be around (6⋅h26⋅h2) per weight matrix.
- **Total model parameters:** For each transformer block:
    - Attention parameters:
        - QKV projections: 3h23h2 parameters
        - Output projection: h2h2 parameters
    - MLP parameters with GLU:
        - Gate and up projections: 8h28h2 parameters (2 matrices of size h×4hh×4h)
        - Down projection: 4h24h2 parameters (1 matrix of size 4h×h4h×h)
    - Total per block: 16h216h2 with GLU MLPs, or 12h212h2 without GLU
    - For full model: 16h2⋅num_layers16h2⋅num_layers (with GLU)
    - Additional parameters:
        - Input embeddings: vocab_size⋅hvocab_size⋅h
        - LM head: vocab_size⋅hvocab_size⋅h (if not tied with input embeddings)
        - Positional embeddings (if used): max_seq_len⋅hmax_seq_len⋅h
- **Forward and backward pass compute (FLOPs):** A very rough estimate for the FLOPs in a forward pass is 2⋅num_tokens⋅num_params2⋅num_tokens⋅num_params. And backward pass compute is twice as that: 4⋅num_tokens⋅num_params4⋅num_tokens⋅num_params.

### A3: Math for Compute/Communication Overlap

Using the formulas from the previous section, we can estimate when computation and communication can effectively overlap in distributed training. Let's look at data parallelism (Zero-0) as an example.

#### Data Parallelism Communication Analysis

The total gradient size that needs to be communicated is:

- Gradients = Parameters ≈ num_layers⋅16h2num_layers⋅16h2

During backward pass, these gradients are communicated in buckets (default 25MB). The communication time to all-reduce each bucket is:

tcomm=tcomm_bucket=bucket_size⋅2(DP−1)DP⋅peak_bwtcomm​=tcomm_bucket​=DP⋅peak_bwbucket_size⋅2(DP−1)​

📝 Note

For bandwidth calculations, we use the bus bandwidth formulas from the [NCCL documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#summary). These formulas account for the specific communication patterns when calculating effective bandwidth between GPUs.

The computation time for backward pass is:

tcompute=4⋅num_tokens⋅num_paramspeak_flopstcompute​=peak_flops4⋅num_tokens⋅num_params​

For effective overlap, we need:

tcommtcompute=num_params2⋅num_tokens⋅DP−1DP⋅peak_flopspeak_bw≤1tcompute​tcomm​​=2⋅num_tokensnum_params​⋅DPDP−1​⋅peak_bwpeak_flops​≤1

This ratio helps determine if communication will become a bottleneck in training. When the ratio is less than 1, communication can be fully overlapped with computation.

#### ZeRO-3 (FSDP) Communication Analysis

For ZeRO-3, parameters and gradients are sharded across GPUs. Let's analyze the communication pattern for a model with transformer blocks of size 16h216h2 parameters each:

- For each transformer block in forward pass:
    - Allgather parameters: 16h2/DP16h2/DP bytes per rank
- For each transformer block in backward pass:
    - Allgather parameters: 16h2/DP16h2/DP bytes per rank
    - Reducescatter gradients: 16h2/DP16h2/DP bytes per rank
- Total communication per block: 3⋅16h2/DP3⋅16h2/DP bytes
- Total communication for full model: 3⋅num_layers⋅16h2/DP3⋅num_layers⋅16h2/DP bytes

The communication time for allgather operations is:

tcomm=16h2⋅DP−1DP⋅peak_bwtcomm​=16h2⋅DP⋅peak_bwDP−1​

The computation time for forward pass of one decoder layer is:

tcompute=32⋅seq_len⋅mbs⋅h2peak_flopstcompute​=peak_flops32⋅seq_len⋅mbs⋅h2​

For effective overlap between computation and communication, we need:

tcommtcompute=12⋅seq_len⋅mbs⋅DP−1DP⋅peak_flopspeak_bw≤1tcompute​tcomm​​=2⋅seq_len⋅mbs1​⋅DPDP−1​⋅peak_bwpeak_flops​≤1

When this ratio is less than 1, the communication of parameters for the next layer can be hidden behind the computation of the current layer.

`

#### TP Communication Analysis

For Tensor Parallel (TP), activations are sharded across GPUs during linears. Let's analyze the communication pattern:

- For each column linear in forward pass:
    - Allgather activations: seq⋅mbs⋅h/TPseq⋅mbs⋅h/TP bytes per rank
- For each column linear in backward pass:
    - Reducescatter gradients: seq⋅mbs⋅h/TPseq⋅mbs⋅h/TP bytes per rank
- And vice-versa for row linears. Each transformer block has 2 column linears and 2 row linears.
- Total communication per block: 8⋅seq⋅mbs⋅h/TP8⋅seq⋅mbs⋅h/TP bytes
- Total communication for full model: 8⋅num_layers⋅seq⋅mbs⋅h/TP8⋅num_layers⋅seq⋅mbs⋅h/TP bytes

Let's analyze if we can overlap the allgather communication for one layer with the computation of the next linear. The communication time for allgather operations is:

tcomm=seq⋅mbs⋅h⋅(TP−1)TP⋅peak_bwtcomm​=TP⋅peak_bwseq⋅mbs⋅h⋅(TP−1)​

While the computation time for the next linear (with parameters h2h2) is:

tcompute=2⋅seq⋅mbs⋅h2TP⋅peak_flopstcompute​=TP⋅peak_flops2⋅seq⋅mbs⋅h2​

For effective overlap, we want the communication time to be less than the compute time:

tcommtcompute=TP−12⋅h⋅peak_flopspeak_bw≤1tcompute​tcomm​​=2⋅hTP−1​⋅peak_bwpeak_flops​≤1

This ratio tells us whether we can successfully hide the allgather communication behind the computation of the next linear. Interestingly, the ratio only depends on the hidden size h and tensor parallelism degree TP, not on sequence length or batch size.

#### PP Communication Analysis

For Pipeline Parallel (PP), activations and gradients are communicated between pipeline stages. Let's analyze the communication pattern:

- For each microbatch in forward pass:
    - Receive and send activations: 2⋅seq⋅mbs⋅h2⋅seq⋅mbs⋅h bytes
- For each microbatch in backward pass:
    - Receive and send gradients: 2⋅seq⋅mbs⋅h2⋅seq⋅mbs⋅h bytes
- Total communication per microbatch: 4⋅seq⋅mbs⋅h4⋅seq⋅mbs⋅h bytes
- For gradient accumulation steps (gas), total communication: 4⋅gas⋅seq⋅mbs⋅h4⋅gas⋅seq⋅mbs⋅h bytes

Let's analyze if we can overlap the communication of activations/gradients with computation of the next transformer block. The computation time for transformer blocks in the next pipeline stage is:

tcompute=32⋅seq⋅mbs⋅h2⋅num_layers_in_next_pppeak_flopstcompute​=peak_flops32⋅seq⋅mbs⋅h2⋅num_layers_in_next_pp​

While the communication time for P2P transfer is:

tcomm=seq⋅mbs⋅hpeak_bwtcomm​=peak_bwseq⋅mbs⋅h​

For effective overlap, we want:

tcommtcompute=peak_flops32⋅h⋅num_layers_in_next_pp⋅peak_bw≤1tcompute​tcomm​​=32⋅h⋅num_layers_in_next_pp⋅peak_bwpeak_flops​≤1

Similar to TP, this ratio is independent of sequence length and batch size. It depends on the hidden size h, number of layers in the next pipeline stage, and the ratio of compute to P2P bandwidth capabilities of the hardware.

### Citation

For attribution in academic contexts, please cite this work as

Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters", 2025.

BibTeX citation

@misc{ultrascale_playbook,
      title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
      author={Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf},
      year={2025},
}

### References

2. Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping  [[PDF]](http://arxiv.org/pdf/2409.15241.pdf)  
    Wang, G., Zhang, C., Shen, Z., Li, A. and Ruwase, O., 2024.
3. Striped Attention: Faster Ring Attention for Causal Transformers  [[PDF]](http://arxiv.org/pdf/2311.09431.pdf)  
    Brandon, W., Nrusimha, A., Qian, K., Ankner, Z., Jin, T., Song, Z. and Ragan-Kelley, J., 2023.
4. Breadth-First Pipeline Parallelism  [[PDF]](http://arxiv.org/pdf/2211.05953.pdf)  
    Lamy-Poirier, J., 2023.
5. DeepSeek-V3 Technical Report  [[PDF]](http://arxiv.org/pdf/2412.19437.pdf)  
    DeepSeek-AI, and others,, 2024.
6. Zero Bubble Pipeline Parallelism  [[PDF]](http://arxiv.org/pdf/2401.10241.pdf)  
    Qi, P., Wan, X., Huang, G. and Lin, M., 2023.
7. Mixtral of Experts  [[PDF]](http://arxiv.org/pdf/2401.04088.pdf)  
    Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L.R., Saulnier, L., Lachaux, M., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T.L., Gervet, T., Lavril, T., Wang, T., Lacroix, T. and Sayed, W.E., 2024.
8. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity  [[PDF]](http://arxiv.org/pdf/2101.03961.pdf)  
    Fedus, W., Zoph, B. and Shazeer, N., 2022.
9. A Survey on Mixture of Experts  [[PDF]](http://arxiv.org/pdf/2407.06204.pdf)  
    Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S. and Huang, J., 2024.
10. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding  [[PDF]](http://arxiv.org/pdf/2006.16668.pdf)  
    Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N. and Chen, Z., 2020.
11. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  [[PDF]](http://arxiv.org/pdf/2205.14135.pdf)  
    Dao, T., Fu, D.Y., Ermon, S., Rudra, A. and Ré, C., 2022.
12. FP8-LM: Training FP8 Large Language Models  [[PDF]](http://arxiv.org/pdf/2310.18313.pdf)  
    Peng, H., Wu, K., Wei, Y., Zhao, G., Yang, Y., Liu, Z., Xiong, Y., Yang, Z., Ni, B., Hu, J., Li, R., Zhang, M., Li, C., Ning, J., Wang, R., Zhang, Z., Liu, S., Chau, J., Hu, H. and Cheng, P., 2023.
13. torchao: PyTorch native quantization and sparsity for training and inference  [[link]](https://github.com/pytorch/ao)  
    maintainers, t. and contributors,, 2024.
14. Small-scale proxies for large-scale Transformer training instabilities  [[PDF]](http://arxiv.org/pdf/2309.14322.pdf)  
    Wortsman, M., Liu, P.J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J.D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-dickstein, J., Xu, K., Lee, J., Gilmer, J. and Kornblith, S., 2023.

[^1]: An Empirical Model of Large-Batch Training  [PDF](http://arxiv.org/pdf/1812.06162.pdf)  McCandlish, S., Kaplan, J., Amodei, D. and Team, O.D., 2018.
[^2]: Mixed Precision Training  [PDF](http://arxiv.org/pdf/1710.03740.pdf)  Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G. and Wu, H., 2018.
[^3]: Reducing Activation Recomputation in Large Transformer Models  [PDF](http://arxiv.org/pdf/2205.05198.pdf)  Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M. and Catanzaro, B., 2022.
