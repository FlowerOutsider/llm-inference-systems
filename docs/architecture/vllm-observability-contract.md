
# vLLM 可观测性契约

## 1. 范围

本项目通过 Docker Compose 部署以下观测链路：

```text
vLLM /metrics -> Prometheus -> Grafana
```

-   vLLM 暴露运行时、调度、KV Cache 与请求延迟指标。
-   Prometheus 每 5 秒抓取一次 `vllm:8000/metrics`。
-   Prometheus recording rules 将原始指标转换为稳定的服务指标。
-   Grafana 自动 provision 数据源和 `vLLM Serving Overview` dashboard。
-   Prometheus alert rules 定义运行风险，但当前本地环境未接入 Alertmanager 通知通道。

2\. 核心服务指标
----------

| 服务指标 | Recording Rule | 含义 |
| --- |  --- |  --- |
| 完成请求速率 | `vllm:completed_requests_per_second:5m` | 最近五分钟平均完成 RPS。 |
| --- |  --- |  --- |
| Prompt 吞吐 | `vllm:prompt_tokens_per_second:5m` | Prefill 输入 token 吞吐。 |
| Generation 吞吐 | `vllm:generation_tokens_per_second:5m` | Decode 输出 token 吞吐。 |
| TTFT P95 | `vllm:ttft_p95_seconds:5m` | 请求从到达到第一个输出 token 的服务端 P95 延迟。 |
| TPOT P95 | `vllm:tpot_p95_seconds:5m` | 输出 token 间隔的服务端 P95 延迟。 |
| Prefix Cache 命中率 | `vllm:prefix_cache_hit_ratio:5m` | GPU Prefix Cache 在查询块中的命中比例。 |
| GPU KV Cache 占用 | `vllm:gpu_cache_usage_ratio` | 当前 GPU KV Cache 使用比例。 |
| 等待请求数 | `vllm:requests_waiting` | 调度器中尚未运行的请求数量。 |
| 运行请求数 | `vllm:requests_running` | 调度器当前运行中的请求数量。 |
| 抢占速率 | `vllm:preemptions_per_second:5m` | 显存压力导致请求抢占的速率。 |

## 3. 指标解释边界
----------

客户端压测器和 Prometheus 的延迟值不要求完全一致：

-   客户端 TTFT 包含 HTTP、SSE、客户端调度等端到端开销。
-   vLLM TTFT 是服务端内部观测指标。
-   `rate(metric[5m])` 是五分钟滚动平均，不等于短压测的瞬时吞吐。
-   结束后查询 `num_requests_waiting` 或 `gpu_cache_usage_perc` 只能看到当前状态，不能反映压测期间的峰值。
-   Prefix Cache 命中率由输入前缀重复程度决定，不应作为所有业务统一的硬性 SLO。

## 4. 本地训练 SLO
------------

以下阈值用于本项目的单卡本地训练环境，不应直接复制到生产环境：

| 指标 | 目标或风险阈值 | 目的 |
| --- |  --- |  --- |
| vLLM target 可抓取 | `up{job="vllm"} == 1` | 确保观测链路与服务端点可用。 |
| --- |  --- |  --- |
| TTFT P95 | 小于 150 ms | 控制交互式请求的首 token 体验。 |
| 等待队列 | 不应持续非空超过 2 分钟 | 识别长期容量不足或调度阻塞。 |
| GPU KV Cache 占用 | 小于 90% | 为突发流量保留 KV Cache 余量。 |
| 抢占 | 5 分钟内应为 0 | 识别显存压力或过度并发。 |

## 5. 告警规则
--------

| 告警 | 条件 | 严重级别 |
| --- |  --- |  --- |
| `VLLMScrapeTargetDown` | vLLM target 不可抓取持续 1 分钟 | critical |
| --- |  --- |  --- |
| `VLLMHighTTFTP95` | TTFT P95 大于 150 ms 持续 5 分钟 | warning |
| `VLLMQueueBacklog` | 等待队列持续非空超过 2 分钟 | warning |
| `VLLMHighGPUKVCacheUsage` | GPU KV Cache 占用高于 90% 持续 2 分钟 | warning |
| `VLLMPreemptionsDetected` | 5 分钟内存在抢占并持续 1 分钟 | warning |

当前 Prometheus 只负责计算告警状态。生产环境还需要配置 Alertmanager，将告警路由到值班系统，并配置抑制、分组、静默和升级策略。

## 6. 常用验证命令
----------

检查 Compose 服务：

```
docker compose\
  --env-file infra/docker/vllm/.env\
  -f infra/docker/vllm/compose.yaml\
  ps
```

检查 vLLM Prometheus target：

```
curl -sS http://127.0.0.1:9090/api/v1/targets | python -m json.tool
```

检查已加载的规则：

```
curl -sS http://127.0.0.1:9090/api/v1/rules | python -m json.tool
```

检查当前活跃告警：

```
curl -sS http://127.0.0.1:9090/api/v1/alerts | python -m json.tool
```