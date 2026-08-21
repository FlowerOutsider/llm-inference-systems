# vLLM Worker Adapter 契约

## 目标

Adapter 将网关控制面与 vLLM OpenAI 兼容接口隔离。它负责 HTTP 协议、SSE 流解析、超时和错误映射；不接管 vLLM 内部的 Continuous Batching、Paged KV Cache 或 token 级调度。

## 职责边界

- 网关控制面：准入、排队、取消、超时策略、限流、请求级观测。
- vLLM Worker：模型执行、Chunked Prefill、Continuous Batching、KV Cache 分配和 token 生成。
- Adapter：将网关请求编码为 `/v1/chat/completions` 请求，并将 SSE 响应转换为类型化事件。

## 请求语义

Adapter 向 `${base_url}/v1/chat/completions` 发起请求，固定启用 `stream=true`，并包含：

- `model`
- `messages`
- `max_tokens`
- `stream`

调用方必须提供非空消息列表和正整数 `max_tokens`。

## 流式响应语义

Adapter 仅处理 `data:` 开头的 SSE 行：

1. `data: [DONE]` 表示流正常结束。
2. JSON 中的 `choices[].delta.content` 转换为文本事件。
3. 仅含 `role` 的首个 chunk 不产生文本事件。
4. 最终 chunk 的 `finish_reason` 作为结束事件返回。
5. 非法 JSON、缺少 `choices` 或错误结构必须抛出协议错误，不能静默吞掉。

## 错误语义

- HTTP 非 2xx：抛出 `VLLMWorkerHTTPError`，携带状态码和响应正文。
- 连接、读取、DNS、超时等 `httpx.RequestError`：抛出 `VLLMWorkerTransportError`。
- SSE/JSON 结构错误：抛出 `VLLMWorkerProtocolError`。
- 调用方可基于错误类型决定重试、熔断、降级或直接返回错误。

## 资源与取消

Adapter 不创建每请求客户端。默认由实例持有一个长期复用的 `httpx.AsyncClient`，通过 `aclose()` 统一关闭。

上层协程取消时，`httpx` 流上下文会退出并关闭本次 HTTP 流。生产网关还需要向 vLLM 传播取消，并将取消结果计入指标。

## 测试策略

单元测试使用 `httpx.MockTransport` 模拟 vLLM，不依赖模型、GPU、Docker 或网络。真实 Compose 环境测试单独执行，用于验证完整 API 连通性与流式指标。