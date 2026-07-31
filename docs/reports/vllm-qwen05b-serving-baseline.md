# Qwen2.5-0.5B vLLM 服务基线压测报告

## 1. 目标

在本地 RTX 3060 Laptop GPU 上部署真实 vLLM 服务，量化不同客户端并发下的：

- 请求吞吐量（RPS）
- 生成吞吐量（tokens/s）
- 首 token 延迟（TTFT）
- 单 token 生成时间（TPOT）
- 端到端延迟（E2E）
- GPU Prefix Cache 命中率
- 错误率

本实验用于分析 Continuous Batching 与 Prefix Caching 的吞吐、延迟和资源复用权衡。

## 2. 环境与服务配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6144 MiB |
| Compute Capability | 8.6 |
| CUDA Driver Runtime | 12.7 |
| CUDA Toolkit | 12.6 |
| 模型 | Qwen/Qwen2.5-0.5B-Instruct |
| 推理框架 | vLLM OpenAI Compatible Server v0.8.5 |
| 推理精度 | FP16 |
| 服务模型名 | `qwen2.5-0.5b` |
| 最大上下文长度 | 2048 tokens |
| GPU 显存利用率目标 | 0.70 |
| 最大并发序列数 | 16 |
| Prefix Cache | 启用 |
| 请求接口 | `/v1/chat/completions` |
| 流式模式 | `stream=true` |

服务运行参数：

```bash
--dtype half
--gpu-memory-utilization 0.70
--max-model-len 2048
--max-num-seqs 16
--enable-prefix-caching
```

## 3. 负载设计
--------

每个请求包含：

-   相同的 system prompt
-   相同的长共享前缀，`shared-prefix-repeats=32`
-   不同的请求编号和问题尾部
-   `max_tokens=64`
-   流式返回

共享前缀设计用于触发真实 Prefix Cache；不同尾部用于避免所有请求完全相同。

测试脚本：

```
benchmarks/serving/vllm_streaming_benchmark.py
```

原始结果文件：

```
benchmarks/serving/results/vllm_qwen05b_sequential.json
benchmarks/serving/results/vllm_qwen05b_concurrency4.json
benchmarks/serving/results/vllm_qwen05b_concurrency8.json
```

## 4. 压测结果
--------

| 客户端并发 | 测试请求数 | RPS | 生成吞吐量 | TTFT P95 | TPOT P95 | E2E P95 | 错误率 |
| --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| 1 | 8 | 3.14 | 150.72 tokens/s | 24.53 ms | 6.66 ms | 336.89 ms | 0.00% |
| --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| 4 | 16 | 9.89 | 474.82 tokens/s | 124.89 ms | 8.60 ms | 466.66 ms | 0.00% |
| 8 | 32 | 17.70 | 849.64 tokens/s | 198.96 ms | 7.64 ms | 567.22 ms | 0.00% |

5\. 吞吐与延迟分析
-----------

### 5.1 吞吐扩展

从并发 1 提升到并发 4：

-   RPS 从 3.14 提升到 9.89，提升约 3.15 倍。
-   生成吞吐量从 150.72 提升到 474.82 tokens/s，提升约 3.15 倍。

从并发 1 提升到并发 8：

-   RPS 从 3.14 提升到 17.70，提升约 5.64 倍。
-   生成吞吐量从 150.72 提升到 849.64 tokens/s，提升约 5.64 倍。

吞吐没有达到线性 8 倍增长，原因包括 GPU 计算资源共享、调度开销、KV Cache 访问以及解码阶段的串行 token 依赖。

### 5.2 TTFT 与 TPOT

并发增加时，TTFT P95 明显升高：

-   并发 1：24.53 ms
-   并发 4：124.89 ms
-   并发 8：198.96 ms

TTFT P95 从并发 1 到并发 8 增长约 8.11 倍。主要原因不是单 token 解码变慢，而是请求需要等待 Continuous Batching 进行调度和执行 prefill。

TPOT P95 保持在 6.66 到 8.60 ms 范围内，说明 decode 阶段每一步生成的稳定性较好。吞吐提升的代价主要体现在排队时间和首 token 延迟，而不是每个后续 token 的生成效率。

### 5.3 E2E 延迟

E2E P95 随并发增加：

-   并发 1：336.89 ms
-   并发 4：466.66 ms
-   并发 8：567.22 ms

因此，若线上 SLO 是 TTFT P95 小于 150 ms，则并发 4 可作为候选配置，而并发 8 虽然吞吐更高，但不应作为默认低延迟配置。

## 6. Prefix Cache 观测
-------------------

vLLM 的 Prefix Cache 指标按 KV block 统计，不按用户请求数统计。应使用计数器增量计算命中率，而不能直接将长期累积值作为单轮结果。

| 轮次 | 查询 blocks 增量 | 命中 blocks 增量 | 命中率 |
| --- |  --- |  --- |  --- |
| 并发 1 | 760 | 675 | 88.82% |
| --- |  --- |  --- |  --- |
| 并发 4 | 1520 | 1510 | 99.34% |
| 并发 8 | 2736 | 2720 | 99.42% |

第一轮包含缓存冷启动和预热过程，因此命中率低于后续轮次。缓存建立后，并发 4 和并发 8 的共享前缀 block 命中率均超过 99%。

命中率不是 100%，因为请求仍存在：

-   不同的请求尾部
-   不同的请求编号
-   可能无法完整复用的末尾 block
-   首次进入缓存的冷启动 block

## 7. 结论
------

本实验在真实 vLLM 服务上验证了两项推理系统关键能力：

1.  Continuous Batching 能够显著提高吞吐。并发从 1 提升到 8 时，RPS 提升约 5.64 倍，且没有请求失败。
2.  Prefix Caching 能够复用共享 prompt 的 GPU KV blocks。在缓存预热后，block 命中率超过 99%，避免重复执行大部分共享前缀的 prefill 计算。

但吞吐提升伴随 TTFT P95 上升。生产环境需要根据业务目标，在吞吐、TTFT、TPOT、显存占用和错误率之间选择调度参数，而不能只追求最大 RPS。

## 8. 后续实验
--------

-   关闭 `--enable-prefix-caching` 后，以相同负载复测，量化 Prefix Cache 的真实收益。
-   在相同并发下改变共享前缀长度，分析上下文长度与 TTFT、KV Cache 占用之间的关系。
-   测试并发 12 和 16，观察接近 `max-num-seqs=16` 时的排队、延迟和错误行为。
-   采集 vLLM `/metrics`，接入 Prometheus 与 Grafana，建立 TTFT、TPOT、队列长度、KV Cache 使用率和 Prefix Cache 命中率仪表板。
-   在更大显存 GPU 上对比多模型、多实例和更高并发条件下的调度行为。