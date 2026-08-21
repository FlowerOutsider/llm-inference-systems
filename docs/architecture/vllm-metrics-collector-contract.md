# vLLM 指标采集器契约

## 目标

采集 vLLM Worker 的健康状态和 Prometheus 指标，转换为 `WorkerSnapshot` 并更新 `WorkerRegistry`。路由器不直接进行 HTTP 调用，只消费快照。

## 采集流程

对于每个已注册 Worker：

1. 请求 `${base_url}/health`。
2. 健康检查成功后，请求 `${base_url}/metrics`。
3. 从 Prometheus 文本格式提取：
   - `vllm:num_requests_waiting`
   - `vllm:num_requests_running`
   - `vllm:gpu_cache_usage_perc`
4. 写入带有单调时间戳的 `WorkerSnapshot`。

## 聚合规则

单个 Worker 可能输出多个带标签的指标样本：

- `waiting_requests`：所有样本求和。
- `running_requests`：所有样本求和。
- `gpu_cache_usage_perc`：取最大值，避免忽略某个 rank 的显存压力。

## 故障语义

以下情况都生成 `healthy=false` 的快照：

- `/health` 非 2xx。
- `/metrics` 非 2xx。
- 网络、连接、读取或超时错误。
- 缺少任何一个必需指标。
- 指标数值无法解析，或 KV Cache 使用率不在 `[0, 1]`。

不健康快照仍写入 Registry，确保 Router 能立即剔除故障 Worker，而不是继续使用旧的健康数据。

## 时效性

采集器写入 `observed_at_monotonic`。Router 根据自己的 `max_snapshot_age_seconds` 剔除过期快照。采集频率应明显小于该时效窗口。

## 当前范围

当前实现为单进程、轮询式采集。生产版本需要支持多实例注册中心、服务发现、指标采集超时、失败告警和指标采集自身的可观测性。