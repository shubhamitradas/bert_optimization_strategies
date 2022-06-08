****Optimzation strategies for speeding up Training****

In a low-resource setup like Google Colab, training/fine-tuning state-of-the-art Transformer based models with GPU becomes a pain as we often run into 
memory issues.

CUDA out of memory errors or GPU Runtime Limit Reached errors are frequent whenever we train such models with a large batch or more epochs even with a shrinked version of the model such as DistillBert.

Here I tried some strategies which helped bring down the training time from ~35 mins to 10 mins on a single GPU machine for 1.5 Lac training data.

**1. Using DataLoader to increase num_worker.**

   PyTorch allows loading data on multiple processes simultaneously, so we can increase the num_workers value to a non-zero value.
   A good rule of thumb is num_worker = 4 * num_GPU.

**2. Using pin_memory.**

   Set pin_memory as True in DataLoader. (DataLoader(dataset, pin_memory = True))

   If we load our samples in the Dataset on CPU and would like to push it during training to the GPU, we can speed up the host to device transfer by    enabling pin_memory.
      This lets our DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.
      More information on the [NVIDIA Blog](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)



**3. Gradient Accumulation.**

When training a neural network, we usually divide our data in mini-batches and go through them one by one. The network predicts batch labels, which are used to compute the loss with respect to the actual targets. Next, we perform backward pass to compute gradients and update model weights in the direction of those gradients.

Gradient accumulation modifies the last step of the training process. Instead of updating the network weights on every batch, we can save gradient values, proceed to the next batch and add up the new gradients. The weight update is then done only after several batches have been processed by the model.
                                                                                                                                                           
**4. Gradient CheckPointing.**

Gradient checkpointing works by omitting some of the activation values from the computational graph. This reduces the memory used by the computational graph, reducing memory pressure overall (and allowing larger batch sizes in the process).

However, the reason that the activations are stored in the first place is that they are needed when calculating the gradient during backpropagation. Omitting them from the computational graph forces PyTorch to recalculate these values wherever they appear, slowing down computation overall.

**5. Mixed Precision Training.**

Mixed-precision training is a technique for substantially reducing neural net training time by performing as many operations as possible in half-precision floating point, fp16, instead of the (PyTorch default) single-precision floating point, fp32. 

The basic idea behind mixed precision training is simple: halve the precision (fp32 → fp16), halve the training time.
    
