
# Continuous Batching Contract

## 目标

ContinuousBatchScheduler 负责为推理服务生成连续批处理计划。
它将处于 Decode 阶段的活跃请求和处于 Prefill 阶段的新请求组合到同一个调度周期中，
以提高 GPU 利用率，同时控制首 token 延迟和单 token 生成延迟。

## 请求状态机

```text
WAITING -> PREFILL -> DECODE -> FINISHED
    |          |         |
    +----------+---------+-> CANCELLED
```

-   `WAITING`：请求已进入等待队列，尚未执行任何 prompt token。
-   `PREFILL`：prompt 正在分块计算 KV Cache。
-   `DECODE`：prompt 已完成，开始每轮生成一个 token。
-   `FINISHED`：已生成 `max_new_tokens` 个 token。
-   `CANCELLED`：请求被取消，不允许再次调度。

调度输入限制
------

每个调度周期同时受以下限制：

-   `max_num_seqs`：一个 batch 最多包含的请求数。
-   `max_num_batched_tokens`：一个 batch 最多处理的 token 数。
-   `prefill_chunk_size`：单个请求在一个调度周期中最多执行的 Prefill token 数。

Decode 每个请求在一个 tick 中消耗一个 token budget。
Prefill 消耗实际调度的 chunk token 数。

核心策略
----

1.  先调度 Decode 请求，再使用剩余预算调度 Prefill。
2.  Decode 优先是为了降低活跃请求的 TPOT，避免生成过程被新请求拖慢。
3.  Prefill 采用 chunked prefill，防止超长 prompt 独占 GPU。
4.  一个请求在一个 tick 中最多进入一次 Prefill chunk，保持基础公平性。
5.  `schedule()` 只生成 SchedulePlan，不改变请求状态。

执行确认
----

执行器消费 SchedulePlan 后，必须显式确认完成的工作：

-   Prefill 成功后调用 `mark_prefill_executed(request_id, token_count)`。
-   Decode 成功后调用 `mark_decode_executed(request_id, token_id)`。
-   只有确认成功，状态机才会前进。

该设计避免调度计划和真实 GPU 执行结果混淆。若 GPU 执行失败、请求取消或 worker 崩溃，
请求状态不会被调度器提前错误提交。

当前边界
----

当前版本只实现确定性的单进程调度策略，还未绑定真实执行器和 KV Cache slot 生命周期。

后续将增加：

1.  请求提交时分配 PagedKVCache slot。
2.  完成或取消时释放 slot 和物理 KV block。
3.  Prefix Cache 命中后复用已有 KV block。
4.  调度计划执行失败时的 rollback。
5.  调度指标：queue time、TTFT、TPOT、batch token 数、KV block 利用率。
6.  多 worker / 多 GPU 调度、backpressure、SLO 与优先级队列。
