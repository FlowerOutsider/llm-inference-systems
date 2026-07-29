
# Qwen2.5-0.5B Prefix Cache 真实模型基准

## 1. 实验目标

验证共享长前缀场景下，复用真实模型产生的 KV Cache 能否减少后续请求的 Prefill 计算成本。

实验不使用 Mock 模型。推理对象为 `Qwen/Qwen2.5-0.5B-Instruct`，运行在本地 NVIDIA RTX 3060 Laptop GPU 上。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| GPU 显存 | 6144 MiB |
| CUDA | 12.6 |
| PyTorch | 2.13.0+cu126 |
| Transformers | 4.57.6 |
| 模型精度 | FP16 |
| 模型 | Qwen/Qwen2.5-0.5B-Instruct |
| Benchmark rounds | 50 |
| Warmup rounds | 10 |

## 3. 请求结构

| 项目 | Token 数 |
|---|---:|
| 共享前缀 | 1984 |
| 请求后缀 | 144 |
| 完整请求 | 2128 |

共享前缀代表生产系统中常见的固定系统提示词、RAG 模板、Agent 工具定义或多租户公共上下文。后缀代表每个用户请求的差异化内容。

## 4. 正确性验证

完整请求路径：

```text
Prefill(prefix + suffix)
```

Prefix Cache 路径：

```
Prefill(prefix) -> DynamicCache
DynamicCache + Prefill(suffix)
```

验证结果：

```
top-1 token agreement: 1.0000
maximum suffix logit difference: 0.14453125
mean suffix logit difference: 0.00828552
```

完整 Prefill 和分段 Cache continuation 的 FP16 logits 未严格 allclose。原因是两条路径经过不同的 FlashAttention 分块和浮点归约顺序，存在数值漂移。

但 144 个 suffix 位置的 Top-1 token 全部一致，说明本实验中的 Prefix Cache continuation 语义正确。后续还需要增加多步生成结果一致性测试。

5\. 性能结果
--------

| 指标 | 完整请求 Prefill | 复用前缀后只计算后缀 |
| --- |  --- |  --- |
| Average latency | 119.329 ms | 24.593 ms |
| --- |  --- |  --- |
| P50 latency | 119.103 ms | 24.126 ms |
| P95 latency | 120.603 ms | 27.607 ms |
| Maximum latency | 124.137 ms | 28.506 ms |

```
Average compute saved per request: 94.736 ms
Average compute reduction: 79.39%
P95 latency reduction: 77.11%
```

6\. KV Cache 显存模型
-----------------

该模型的 KV Cache 每 token 显存为：

```
24 layers
× 2 (Key 和 Value)
× 2 KV heads
× 64 head_dim
× 2 bytes (FP16)
= 12,288 bytes/token
= 12 KiB/token
```

共享前缀共有 1984 token：

```
1984 × 12 KiB = 23.250 MiB
```

实验观测值与理论计算一致。

7\. 前缀构建成本与摊销
-------------

共享前缀首次 Prefill 的稳态延迟：

```
115.218 ms
```

若有 N 个请求复用同一个前缀：

```
不使用 Prefix Cache：
N × 119.329 ms

使用 Prefix Cache：
115.218 ms + N × 24.593 ms
```

当：

```
N > 115.218 / (119.329 - 24.593)
N > 1.22
```

从第 2 个共享此前缀的请求开始，Prefix Cache 产生净计算收益。

以 10 个请求为例：

```
不复用：
10 × 119.329 = 1193.290 ms

复用前缀：
115.218 + 10 × 24.593 = 361.148 ms
```

该场景下计算时间减少约 69.7%。

8\. 当前实现边界
----------

本实验使用 Transformers 的 `DynamicCache`，并在计时区间外通过 `deepcopy` 保护共享前缀不被后缀写入污染。

因此，本实验衡量的是：

```
真实模型的 Prefix Cache 计算复用收益
```

而不是生产级物理显存共享收益。

生产级 Prefix Cache 需要：

-   分块式 Paged KV Cache
-   Block Table
-   物理 block 引用计数
-   Prefix 命中索引
-   Copy-on-Write
-   并发请求隔离
-   逐出策略与显存水位控制

这些能力将与项目中已有的 `PagedKVCache`、Prefix Index 和 Prefix Cache Coordinator 继续集成。