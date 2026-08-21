# 负载感知路由契约

## 目标

为 MaaS 网关选择最合适的 vLLM Worker。路由器只做请求级 Worker 选择；选中 Worker 后，模型内部的 Chunked Prefill、Continuous Batching 和 Paged KV Cache 仍由 vLLM 负责。

## Worker 快照

每个 Worker 需要上报：

- `worker_id`：稳定且唯一的 Worker 标识。
- `base_url`：vLLM OpenAI API 地址。
- `model_ids`：该 Worker 当前可服务的模型集合。
- `healthy`：健康检查结果。
- `waiting_requests`：vLLM 等待队列长度。
- `running_requests`：正在执行的请求数。
- `gpu_cache_usage_perc`：KV Cache 使用率，范围为 `[0, 1]`。
- `observed_at_monotonic`：该快照的单调时间戳。

过期快照不能参与路由。初始实现使用 10 秒最大时效，后续由配置中心管理。

## 准入规则

一个 Worker 必须同时满足以下条件才可参与候选：

1. `healthy` 为真。
2. `model_ids` 包含请求模型。
3. Worker 快照未过期。

若没有候选 Worker，路由器抛出 `NoHealthyWorkerError`。上层网关将其映射为可重试的 `503`，并计入路由失败指标。

## 评分策略

默认评分由以下部分组成：

```text
score =
  waiting_requests * 4
  + running_requests
  + gpu_cache_usage_perc * 5
  - prefix_cache_bonus
```


若请求已知前缀位于该 Worker，`prefix_cache_bonus = 4`；否则为 `0`。

分数越低越优。Prefix Cache 只是一项收益，不允许压过明显的队列积压或显存压力。分数相同时，按 `worker_id` 字典序选择，保证决策可复现。

Prefix Cache 输入
---------------

路由器不直接扫描 KV Cache。调用方根据 Prefix Index 或分布式目录提供 `prefix_candidate_worker_ids`，表示哪些 Worker 已持有可复用前缀。

后续分布式版本将由 Prefix Cache Directory 为路由器提供该集合，并处理副本、过期、逐出和跨 Worker 一致性。

可观测性
----

每次路由决策应记录：

-   请求 ID、模型 ID、选中 Worker。
-   候选 Worker 数量。
-   是否命中前缀候选。
-   选中 Worker 的评分及组成部分。
-   无可用 Worker、指标过期和健康检查失败的次数。