
# 推理运行时契约

## 目标

`InferenceRuntime` 是在线大模型推理服务的控制面集成层，负责协调请求准入、连续批处理、Prefix Cache 资源绑定和请求终态清理。

它不直接执行模型推理。后续由后端适配器负责将已调度的请求批次发送给真实 vLLM Worker。

## 职责边界

- `RequestLifecycle`：负责请求准入和合法的请求状态迁移。
- `ContinuousBatchScheduler`：负责维护等待队列，并选择本轮可运行的请求。
- `PrefixCacheManager` 与 `PrefixCacheCoordinator`：负责 Prefix Cache 的查询、登记、共享和淘汰策略。
- `PagedKVCache`：负责物理 KV Block、Block Table、引用计数和 Copy-on-Write。
- `InferenceRuntime`：只负责跨组件协调，不直接修改其他组件的内部状态。

## 请求状态

请求允许发生如下状态迁移：

```text
NEW -> WAITING -> RUNNING -> FINISHED
                         -> FAILED
WAITING -> CANCELLED
RUNNING -> CANCELLED
```

`FINISHED`、`FAILED`、`CANCELLED` 都是终态。进入终态的请求不能再次被调度。

核心操作
----

### submit

`submit` 通过 `RequestLifecycle` 对请求进行准入，并将通过准入的请求交给 `ContinuousBatchScheduler`。

若准入失败，请求不得进入调度队列，也不得占用 KV Cache 或 Prefix Cache 资源。

### schedule\_once

`schedule_once` 从 `ContinuousBatchScheduler` 获取下一轮可运行的请求批次。

对每个被选中的请求，运行时必须：

1.  确认请求仍处于可执行状态。
2.  将请求状态推进到 `RUNNING`。
3.  通过 Prefix Cache 协调器查询并绑定可复用的前缀资源。
4.  返回待执行批次，但不在此阶段执行模型推理。

后续 vLLM 后端适配器只需要接收该批次并执行推理。

### complete

`complete` 将运行中的请求推进到 `FINISHED`，移除其调度状态，并释放该请求持有的 KV Cache 和 Prefix Cache 引用。

### fail

`fail` 将请求推进到 `FAILED`，记录失败原因，移除其调度状态，并释放该请求持有的资源。

### cancel

`cancel` 将等待中或运行中的请求推进到 `CANCELLED`，阻止其再次被调度，并释放该请求持有的资源。

资源不变量
-----

1.  活跃请求在同一时刻最多只能被调度一次。
2.  终态请求不得回到 `WAITING` 或 `RUNNING`。
3.  请求清理必须具备幂等性。
4.  清理时必须先移除调度器成员关系，再释放请求持有的缓存资源。
5.  Prefix Cache 命中只能共享有效且完整的前缀 Block。
6.  当其他请求或保留的 Prefix Cache 条目仍持有引用时，不得回收共享的物理 Block。
7.  准入拒绝不得残留调度队列记录、Slot、Block 或 Prefix 引用。

失败处理
----

失败可能发生在请求准入、调度、Prefix Cache 绑定、模型执行或客户端取消阶段。

若请求已经完成部分运行时操作后发生失败，运行时必须尽力完成资源清理，并保留原始失败原因。清理失败不能让请求继续保持可调度状态。

可观测性边界
------

后续运行时应暴露以下指标：

-   已准入、被拒绝、被调度、成功、失败和取消的请求数。
-   请求排队等待时间。
-   等待中和运行中的请求数量。
-   Prefix Cache 命中与未命中次数。
-   资源清理失败次数。

第一阶段暂不实现指标，但运行时接口必须保留足够的结构化状态，以便后续加入指标时不改变状态迁移语义。

第一轮测试范围
-------

第一轮集成测试至少覆盖：

1.  正常的提交、调度和完成链路。
2.  准入拒绝后请求不会进入调度队列。
3.  已被调度的请求不会再次出现在下一批次。
4.  失败或取消的请求不会再次被调度。
5.  Prefix Cache 命中和未命中路径。
6.  终态操作具备幂等性。
7.  清理完成后，调度器状态和缓存资源所有权保持一致。