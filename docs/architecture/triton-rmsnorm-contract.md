# Triton RMSNorm Contract

## 目标

本模块实现 LLM 推理路径中的 RMSNorm 融合 Kernel，并与 PyTorch eager 基线进行正确性和延迟对比。

RMSNorm 定义为：

```text
y = x / sqrt(mean(x^2) + epsilon) * weight
```

归约使用 FP32 累加，以降低 FP16 输入时的数值误差。

输入与输出
-----

-   `x`：CUDA 上的连续张量，最后一维是 hidden size。
-   `weight`：一维 CUDA 张量，长度必须等于 hidden size。
-   `x` 和 `weight` 的 device、dtype 必须一致。
-   支持 `float16`、`bfloat16` 和 `float32`。
-   输出 shape、dtype 与 `x` 一致。

Kernel 映射
---------

一个 Triton program 对应 `x` 的一行，即一个 hidden-state vector：

1.  从全局显存加载一行输入和缩放权重。
2.  在 program 内归约计算 `mean(x^2)`。
3.  计算 inverse RMS。
4.  完成缩放并写回输出。

因此，归约、归一化和缩放都在一次 Kernel launch 中完成。

性能边界
----

-   本实现是正确性优先的单行融合基线，不等同于生产级 TensorRT-LLM 或 vLLM 融合算子。
-   hidden size 会向上补齐到 2 的幂，补齐元素通过 mask 屏蔽。
-   当前限制 block size 不超过 65536。
-   PyTorch eager 基线可能涉及多个 Kernel 和中间张量；Triton 实现减少 launch 与中间读写，但最终收益取决于 shape、缓存、dtype 和 GPU 架构。

验证
--

```
python -m pytest -q tests/test_triton_rmsnorm.py

python kernels/triton/01_rmsnorm/benchmark.py\
  --rows 4096\
  --hidden-size 896\
  --dtype float16
```

Qwen2.5-0.5B 的 hidden size 为 896，因此该实验 shape 与当前真实模型一致。