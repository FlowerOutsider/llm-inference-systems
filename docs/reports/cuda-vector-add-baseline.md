# CUDA Vector Add Baseline

## 1. 实验目标

实现并分析一个 CUDA Vector Add kernel，建立后续 GEMM、Attention 等算子优化的性能分析基线。

## 2. 实验环境

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Compute Capability: 8.6
- Global Memory: 6143 MiB
- SM Count: 30
- CUDA Toolkit: 12.6
- Profilers: Nsight Systems 2024.5, Nsight Compute 2024.3

## 3. Kernel 与启动配置

- Elements: 16,777,216
- Block size: 256 threads
- Grid size: 960 blocks
- Total threads: 245,760
- Kernel: `output[i] = a[i] + b[i]`
- Execution model: grid-stride loop
- Benchmark method: 10 次预热，100 次 CUDA Event 计时

## 4. 正确性与基准结果

| Metric | Result |
| --- | ---: |
| Average kernel latency | 0.645 ms |
| Effective bandwidth | 312.364 GB/s |
| Maximum absolute error | 0.000 |

有效带宽按逻辑数据流量计算：每个元素读取两个 `float` 输入并写入一个 `float` 输出，即 `3 * N * sizeof(float)`，再除以单次 kernel 延迟。

## 5. Nsight Compute 结果

| Metric | Result |
| --- | ---: |
| NCU kernel duration | 655.23 us |
| Memory throughput | 91.52% |
| DRAM throughput | 91.52% |
| Compute throughput | 12.83% |
| Theoretical occupancy | 100% |
| Achieved occupancy | 93.83% |
| Registers per thread | 16 |
| Waves per SM | 5.33 |

## 6. 分析结论

该 kernel 是典型的 memory-bound workload。

1. DRAM throughput 达到 91.52%，说明全局显存带宽已接近设备可用上限。
2. Compute throughput 仅为 12.83%，说明浮点计算不是当前瓶颈。
3. Achieved occupancy 为 93.83%，接近理论上限，继续仅靠增加线程或提高 occupancy 难以取得明显收益。
4. 后续优化方向应是减少全局内存访问，或通过 kernel fusion 将 Vector Add 与相邻算子融合，而不是单独优化加法指令。

## 7. 测量注意事项

Nsight Compute 的主程序计时结果会受到多轮 replay 与硬件计数器采集开销影响，不能代替 CUDA Event 基准。性能结论以 CUDA Event 的 0.645 ms 为主，NCU 用于定位瓶颈和分析硬件利用率。