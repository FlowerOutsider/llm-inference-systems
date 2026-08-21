
# MaaS Gateway API 契约

## 端点

```text
POST /v1/chat/completions
```

当前仅支持 `stream=true` 的 OpenAI 兼容流式请求。

请求字段
----

-   `model`：必填，目标模型 ID。
-   `messages`：必填，非空消息数组；每条消息包含 `role` 与 `content`。
-   `max_tokens`：可选，默认 `64`，必须为正整数。
-   `stream`：必须为 `true`。

预处理顺序
-----

在开始 HTTP 流式响应之前，Gateway 必须：

1.  刷新所有已注册 Worker 的快照。
2.  根据模型、健康状态、指标时效、负载和 Prefix Cache 候选完成路由。
3.  获取选中 Worker 的长期复用 Adapter。

因此没有可用 Worker 时，客户端得到标准 JSON `503`，而不是收到 HTTP 200 后才发现流失败。

成功响应
----

响应类型为 `text/event-stream`。每个 Adapter 事件编码为：

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

```

流结束时发送：

```
data: [DONE]

```

错误语义
----

-   `stream=false`：`400`，当前版本明确不支持非流式响应。
-   没有可用 Worker：`503` JSON。
-   流已经开始后的上游 Worker 错误：发送 SSE `error` 事件，再发送 `[DONE]`。
-   FastAPI 请求校验失败：`422` JSON。

当前范围
----

当前请求前同步刷新 Worker 快照，仅用于端到端正确性验证。后续替换为后台周期刷新、熔断、重试预算、认证、限流、租户隔离与分布式 Prefix Cache Directory。