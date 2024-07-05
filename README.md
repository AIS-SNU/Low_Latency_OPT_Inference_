# Low latency OPT inference

FlexGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexGen allows **high-throughput** generation by IO-efficient offloading, compression, and **large effective batch sizes**. Instead, we

## Motivation

This repo is based on the code [FlexGen](https://github.com/FMInference/FlexGen).

## Installation
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method: From source
```
git clone https://github.com/FMInference/FlexGen.git
cd FlexGen
pip install -e .
```

## Usage and Examples

### Get Started with a Single GPU

In single gpu inference case, low_latency_opt is same as flex_opt with num_gpu_batches 1. But we implemented per layer initialization. It allows the user to adjust the initialization ratio of each layer differently across GPU, CPU, and disk. 
```
# example command
python3 -m flexgen.low_latency_opt --model facebook/opt-30b --cache-percent 100 0 --per-layer-percent 0 100 0 100 0 100 0 100 --gpu-batch-size 1
```
### Distributed Low Latency OPT
FlexGen used pipeline parallelism for inference across multiple GPUs, but we modified the code to use tensor parallelism. 
```
# example command
mpirun -n 4 python -m flexgen.dist_low_latency_opt --head-ip 127.0.1.1 --port 7777 --use-mpi --model facebook/opt-6.7b --gpu-batch-size 24 --cache-percent 100 0 --per-layer-percent 100 0 100 0 100 0 100 0 --comm-device gpu
```

### Balanced OPT
When the batch size is small, resulting in smaller cache and activation sizes, we found that most of the time is spent on transferring model weights from CPU RAM to GPU VRAM. To solve this problem, we implemented tensor parallelism between the CPU and GPU. With tensor parallelism, we can perform independent calculations on each processing unit with the weight parameters initialized on the respective GPU and CPU. At the end of each layer, we only need to add a gather or reduce operation to complete the computation. In FlexGen, during weight initialization, each weight in the required weight list for each layer was assigned to a specific device. To implement tensor parallelism between the GPU and CPU, we modified the code to allocate each weight in the required weight list for each layer to devices based on the initialization percentage.

Furthermore, to handle more computations on the GPU than indicated by the initialized percent, we implemented a feature that transfers a portion of the weights stored in CPU RAM to GPU VRAM. By setting the per-layer-computation-percent to a different value than the per-layer-weight-percent or cache-percent option, the layer weights are distributed according to the computation percent during weight loading. Since communication and computation operate in parallel, adjusting the balance between communication time and computation time allows for a model with lower latency. 
