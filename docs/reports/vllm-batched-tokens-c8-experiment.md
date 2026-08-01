
# vLLM Continuous Batching Token Budget Experiment

## 目标

评估 `max_num_batched_tokens` 对 vLLM Continuous Batching 推理性能的影响，并为当前单 GPU 服务选择默认预算。

## 固定环境

- Serving engine: vLLM `v0.8.5`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Served model name: `qwen2.5-0.5b`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- Precision: FP16
- GPU memory utilization: `0.70`
- Max model length: `2048`
- Max concurrent sequences: `16`
- Prefix Cache: enabled
- Deployment: Docker Compose
- Client endpoint: `http://127.0.0.1:8002`

## 负载协议

- Warmup requests: `16`
- Measured requests per run: `256`
- Client concurrency: `8`
- Maximum generated tokens: `64`
- Shared prefix repeats: `32`
- Repetitions per configuration: `3`
- Aggregation: median, with `[minimum, maximum]` range

## 结果

| Token Budget | Runs | RPS | Generation Throughput | TTFT P95 | TPOT P95 | E2E P95 | Error Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 3 | 32.84 [32.20, 34.54] | 967.62 [950.07, 1018.22] tokens/s | 46.60 [36.96, 55.77] ms | 7.78 [7.73, 8.28] ms | 260.23 [259.00, 285.37] ms | 0.00% |
| 2048 | 3 | 33.73 [32.92, 33.73] | 995.07 [970.99, 995.69] tokens/s | 48.19 [47.41, 54.34] ms | 7.48 [7.25, 7.75] ms | 257.82 [256.02, 267.74] ms | 0.00% |
| 4096 | 3 | 32.16 [31.97, 32.85] | 949.01 [944.81, 969.74] tokens/s | 48.67 [48.66, 49.69] ms | 8.09 [7.88, 8.22] ms | 276.06 [268.83, 280.53] ms | 0.00% |

## 分析

`1024` 获得了最低的 TTFT P95，但其 TTFT 范围最宽，说明在当前负载下存在更明显的调度波动。较小 token budget 会限制单轮 prefill 的工作量，可能减少部分请求等待，但也会增加调度轮次。

`2048` 在吞吐和端到端延迟之间取得最佳平衡。相较于 `1024`，RPS 提升约 2.7%，生成吞吐提升约 2.8%，TPOT P95 降低约 3.9%，E2E P95 降低约 0.9%。

`4096` 没有在并发 8 下带来更多吞吐，反而使 TPOT P95 和 E2E P95 增大。说明当前队列深度不足以充分利用更大的批处理预算，较大的 prefill 批次可能增加 decode 请求等待时间。

## 结论

当前部署选择：

```dotenv
MAX_NUM_BATCHED_TOKENS=2048
```

选择依据是它在当前模型、共享前缀、并发度和输出长度下提供了最高吞吐、最低 TPOT P95 与最低 E2E P95。该结论不是通用常数，只适用于本报告定义的工作负载。

局限与后续实验
-------

-   仅验证了单 GPU、Qwen 0.5B、并发 8 的工作负载。
-   每组只有 3 次重复，尚未计算置信区间。
-   尚未覆盖不同 prompt 长度、并发 1/16/32、长输出和混合 prefill/decode 队列。
-   后续应记录每轮实验的 GPU 利用率、显存、Prefix Cache 命中增量与 vLLM scheduler 指标。
-   后续可引入随机化实验顺序，降低 GPU 温度和频率状态对配置比较的影响。


可复现命令
-----

```
python scripts/summarize_vllm_benchmarks.py
```