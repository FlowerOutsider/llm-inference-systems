# CUDA GEMM: Naive vs Shared-Memory Tiled

## 实验目标

实现并比较 naive GEMM 与 shared-memory tiled GEMM，分析数据复用对矩阵乘法性能的影响。

## 实验配置

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- CUDA Toolkit: 12.6
- Matrix size: 1024 x 1024
- Block size: 16 x 16 = 256 threads
- Grid size: 64 x 64 = 4096 blocks
- Benchmark: 5 次预热，20 次 CUDA Event 计时

## 正确性结果

- Naive vs tiled max difference: 0.000
- Sampled reference max error: 0.000

## 性能结果

| Implementation | Latency | Throughput | Relative Speedup |
| --- | ---: | ---: | ---: |
| Naive GEMM | 2.856 ms | 751.802 GFLOPS | 1.000x |
| Tiled GEMM | 2.168 ms | 990.648 GFLOPS | 1.318x |

## Nsight Compute 对比

| Metric | Naive | Tiled |
| --- | ---: | ---: |
| Kernel duration | 3.61 ms | 2.74 ms |
| Achieved occupancy | 98.32% | 98.33% |
| Registers per thread | 40 | 38 |
| Static shared memory per block | 0 KB | 2.05 KB |
| Total DRAM elapsed cycles | 151.4M | 115.0M |

## 结论

两种实现具有相同的线程配置和近似的 occupancy，因此性能提升不是由并发度差异造成。

Tiled GEMM 将 A、B 的局部数据块加载到 shared memory，使同一 block 内的多个线程复用数据，降低了重复的全局内存访问与等待。性能提升 1.318x，吞吐从 751.802 GFLOPS 提升到 990.648 GFLOPS。

该实现仍未使用 register tiling、向量化加载、Tensor Core、warp-level primitive 或 cuBLAS，因此它是后续高性能 GEMM 优化的正确基线，而不是最终实现。