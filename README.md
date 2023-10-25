# numba_examples
[Numba](https://numba.pydata.org/) is a python library that offers Just-in-Time (JIT) compilation and allows you to write GPU kernels in Python. This repo demonstrates a few examples of using Numba: 
* `example_vector_sum_and_average.ipynb` shows how to add two vectors and how to compute the average of elements in a vector. This example uses 1-dimensional blocks and threads.
* `example_image_convolution.ipynb` shows how to run convolution on an image. This example uses 2-dimensional blocks and threads.
* `example_mppi_numba_obstacle_avoidance.ipynb` shows how to parallelize rollouts based on [Model Predictive Path Integral (MPPI) control proposed by Williams et al.](https://ieeexplore.ieee.org/document/7989202). This implementation can run about 100x faster than the CPU implementation in `example_mppi_cpu.ipynb`. (The CPU implementation doesn't account for obstacles.)


[Try these notebooks via Google Collab!](https://drive.google.com/drive/folders/1h-DArD9gMMB3dhAt3l_n2FjD8abe2PDh?usp=sharing) Make sure to choose a GPU instance. "Runtime" -> "Change runtime type" -> "Hardware accelerator".


## Installation
`pip3 install numba scikit-image`

## Terminologies
* **Host**: the CPU
* **Device**: the GPU
* **Host memory**: the system main memory
* **Device memory**: GPU memory
* **Kernel**: a GPU function launched by host and executed on the device
* **Device function**: a GPU function that can only be invoked by kernels or other device functions


## Difference between GPU and CPU
While the CPU is designed to excel at executing a sequence of operations, called a thread, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).


## Thread hierarchy

Blocks are organized into a 1D or 2D or 3D grid of thread blocks as illustrated below. The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system. 

* **Grid (1D/2D/3D)**: a grid consists of blocks. Modern GPUs typically have **more than 65,535 x 65,535 blocks**.
* **Block (1D/2D/3D)**: a block consists of threads. Modern GPUs typically have **about 1024 threads per block**.
* **Thread**: kernel functions are run by threads

**Why organized this way??**
* Some tasks are naturally viewed in 2D/3D (e.g., image processing, ray-tracing).
* The **threads in the same block** have access to some **shared memory**. Therefore, this allows the threads to work together, e.g., for computing convolution. 
* **Global memory** (in a grid) is slower than **shared memory** (in a block) which is slower than **local memory** (in a thread).


Note that threads are excuted asynchronously, so synchronization may be required among the threads in a block and sometimes among the blocks in a grid. 


<p align="center">
    <img src="images/grid-of-thread-blocks.png" height="200" />
</p>
<p align="center">
    <em>Grid of Thread Blocks. (Image from <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html">Nvidia CUDA guide</a>)</em>
</p>


## Where to learn more about GPU and Numba?
* [Numba for CUDA GPUs](https://numba.readthedocs.io/en/stable/cuda/index.html)
* [The official CUDA C programming guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide)
* A pretty good playlist on Youtube (https://www.youtube.com/watch?v=4APkMJdiudU&list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)

## Questions?
Find Xiaoyi (Jeremy) Cai (xyc@mit.edu)
