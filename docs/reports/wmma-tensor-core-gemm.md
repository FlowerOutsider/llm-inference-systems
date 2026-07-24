# WMMA Tensor Core GEMM Baseline

## 实验目标

使用 CUDA WMMA API 实现 FP16 输入、FP32 累加的矩阵乘法，理解 Tensor Core 的 warp-level 编程模型，并与手写 tiled GEMM、cuBLAS 基线对比。

## 配置

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Compute Capability: 8.6
- Matrix size: 1024 x 1024
- WMMA tile: 16 x 16 x 16
- Input type: FP16
- Accumulator type: FP32
- Block configuration: 4 warps per block

## 结果

| Implementation | Latency | Throughput | Sampled Max Relative Error |
| --- | ---: | ---: | ---: |
| Hand-written tiled FP32 | 2.125 ms | 1010.507 GFLOPS | 0.000000 |
| WMMA FP16 -> FP32 | 0.389 ms | 5521.000 GFLOPS | 0.000011 |
| cuBLAS FP32 | 0.312 ms | 6880.420 GFLOPS | 0.000001 |
| cuBLAS TF32 | 0.293 ms | 7338.898 GFLOPS | 0.000030 |

## 结论

WMMA 相比手写 tiled FP32 GEMM 提升约 5.46 倍，验证了 Tensor Core 对矩阵乘法的加速能力。

WMMA 仍低于 cuBLAS 的性能，因为当前实现只使用了基本的 warp-level matrix multiply primitive，未实现 shared-memory staging、寄存器分块、异步拷贝、多级流水和库级调度优化。

WMMA 使用 FP16 输入与 FP32 累加，在当前采样验证中最大相对误差为 1.1e-5。该结果仅对当前输入分布与采样点成立，不能替代全量精度评估。