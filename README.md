# Low latency OPT inference

FlexGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexGen allows **high-throughput** generation by IO-efficient offloading, compression, and **large effective batch sizes**. Instead, we

## Motivation

This repo is based on the code [FlexGen](https://github.com/FMInference/FlexGen).

## Installation
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method: From source
```
git clone https://github.com/fmlini251/FlexGen.git
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

#### Multi GPU performance

Table below is the latency and speedup of LL_OPT on various size of models. For large models such as OPT-6.7B, OPT-30B, we were able to achieve a speedup proportional to the number of GPUs. As the number of GPUs increases, the required VRAM size per GPU decreases, allowing for a higher proportion of weights to be initialized on each GPU. The '*' symbol denotes the result achieved when initializing as high a proportion of weights on GPUs as possible.

|               |Latency(s)|Speedup|Latency(s)|Speedup|Latency(s)|Speedup|
| ---           | :---:    | :---: | :---:    | :---: | :---:    | :---: |
| model         | OPT-1.3B |       | OPT-6.7B |       | OPT-30B  |       |
| LL_OPT(1 GPU) | 0.70     | 1.00  | 28.23    | 1.00  | 317.27   | 1.00  |
| LL_OPT(2 GPU) | 0.74     | 0.95  | 14.27    | 1.98  | 158.71   | 2.00  |
| LL_OPT(2 GPU)*| 0.74     | 0.95  | 1.07     | 26.28 | 107.45   | 2.95  |
| LL_OPT(4 GPU) | 0.78     | 0.90  | 7.27     | 3.88  | 79.74    | 3.98  |
| LL_OPT(4 GPU)*| 0.78     | 0.90  | 1.09     | 25.86 | 27.04    | 11.73 |

Configs for benchmark is [Config](https://github.com/fmlini251/FlexGen/blob/main/docs/benchmark_configs.txt)


The graph below shows the throughput and latency measurements of our model and FlexGen, with varying initial arguments (e.g., init weight percent, init cache percent, batch size, num_batch). Compared to FlexGen, our model demonstrates better throughput and lower latency.
<img src="https://github.com/fmlini251/FlexGen/blob/main/docs/performance.png" alt="image" width="500"></img>

### Balanced OPT
When the batch size is small, resulting in smaller cache and activation sizes, we found that most of the time is spent on transferring model weights from CPU RAM to GPU VRAM. To solve this problem, we implemented tensor parallelism between the CPU and GPU. With tensor parallelism, we can perform independent calculations on each processing unit with the weight parameters initialized on the respective GPU and CPU. At the end of each layer, we only need to add a gather or reduce operation to complete the computation. In FlexGen, during weight initialization, each weight in the required weight list for each layer was assigned to a specific device. To implement tensor parallelism between the GPU and CPU, we modified the code to allocate each weight in the required weight list for each layer to devices based on the initialization percentage.

Furthermore, to handle more computations on the GPU than indicated by the initialized percent, we implemented a feature that transfers a portion of the weights stored in CPU RAM to GPU VRAM. By setting the per-layer-computation-percent to a different value than the per-layer-weight-percent or cache-percent option, the layer weights are distributed according to the computation percent during weight loading. Since communication and computation operate in parallel, adjusting the balance between communication time and computation time allows for a model with lower latency. 
```
# example command
python3 -m flexgen.balanced_opt --model facebook/opt-30b --cache-percent 40 60 --per-layer-weight-percent 0 100 0 100 0 100 0 100 --per-layer-computation-percent 40 40 40 40 --gpu-batch-size 1
```

<img src="https://github.com/fmlini251/FlexGen/blob/main/docs/Balanced_computation.png" alt="image" width="500"></img>

#### Balanced OPT Performance(when weight initialization percent and computation percent are equal)
When using the below command, we were able to achieve a latency of 12.53 seconds. This result is approximately 2.25 times faster than the latency of the original low_latency_opt, which used the same number of GPUs and the same ratio of weight initialization.
```
python3 -m flexgen.balanced_opt --model facebook/opt-6.7b --cache-percent 70 30 --per-layer-weight-percent 70 30 70 30 70 30 70 30 --per-layer-computation-percent 70 70 70 70 --gpu-batch-size 1
```

#### Balanced OPT Performance(when weight initialization percent and computation percent are different)
We observed that we did not achieve the desired performance when weight initialization percent and computation percent differed. The reason as 

Config for benchmark is [Config](https://github.com/fmlini251/FlexGen/blob/main/docs/balanced_opt_benchmark_config.txt)