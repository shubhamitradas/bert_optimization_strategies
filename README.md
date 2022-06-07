**Optimzation strategies for speeding up Training**

In a low-resource setup like Google Colab, training/fine-tuning state-of-the-art Transformer based models with GPU becomes a pain as we often run into 
memory issues.

CUDA out of memory errors or GPU Runtime Limit Reached errors are frequent whenever we train such models with a large batch or more epochs even with a shrinked version of the model such as DistillBert.

Here I tried some strategies which helped bring down the training time from ~35 mins to 10 mins on a single GPU machine for 1.5 Lac training data.

**1. Using DataLoader to increase num_worker.**

**2. Using pin_memory.**

**3. Gradient Accumulation**

**4. Gradient CheckPointing.**

**5. Mixed Precision Training**
    
