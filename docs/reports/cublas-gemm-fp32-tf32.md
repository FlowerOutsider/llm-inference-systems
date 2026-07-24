# cuBLAS GEMM FP32 and TF32 Baseline

## 实验目标

对比 cuBLAS FP32 与 TF32 GEMM 的吞吐和数值误差，并与手写 shared-memory tiled GEMM 建立性能差距。

## 实验配置

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- CUDA Toolkit: 12.6
- Matrix size: 1024 x 1024
- Layout: column-major
- Benchmark: 10 次预热，100 次 CUDA Event 计时
- FP32 mode: `CUBLAS_PEDANTIC_MATH`
- TF32 mode: `CUBLAS_TF32_TENSOR_OP_MATH`

## 结果

| Implementation | Latency | Throughput | Max Absolute Error | Max Relative Error |
| --- | ---: | ---: | ---: | ---: |
| Hand-written tiled FP32 | 2.125 ms | 1010.507 GFLOPS | 0.000000 | 0.000000 |
| cuBLAS FP32 | 0.312 ms | 6880.420 GFLOPS | 0.000259 | 0.000001 |
| cuBLAS TF32 | 0.293 ms | 7338.898 GFLOPS | 0.007462 | 0.000030 |

## 分析

cuBLAS FP32 相比手写 tiled GEMM 提升约 6.8 倍，说明生产级 GEMM 库通过更复杂的分块、寄存器复用、指令调度和硬件适配获得了明显更高的效率。

TF32 相比 cuBLAS FP32 提升约 1.067 倍，最大相对误差从 1e-6 增加到 3e-5。TF32 使用更低精度的输入乘法并保持更高精度的累加，适合对数值误差有一定容忍度的深度学习推理和训练场景。

FP32 与 CPU 参考值之间存在微小差异，原因是 GPU 与 CPU 的浮点累加顺序及 FMA 行为不同；该差异不能简单视为计算错误。

## 局限性

当前只对 16 个采样输出位置进行 CPU 参考校验；后续应补充全量误差统计、不同矩阵尺寸、不同 batch shape 和端到端模型精度验证。