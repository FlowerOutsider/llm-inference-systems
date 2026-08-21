
# MaaS Gateway 运行时契约

## 进程拓扑

```text
Client
  -> Gateway Uvicorn :8003
  -> GatewayService
  -> VLLMOpenAIWorker Pool
  -> vLLM OpenAI API :8002
  -> GPU
```

Prometheus、Grafana 与 vLLM 保持由现有 Compose 栈管理；Gateway 通过 vLLM 的 `/health` 和 `/metrics` 获取实时 Worker 快照。

必需环境变量
------

-   `GATEWAY_VLLM_BASE_URL`：vLLM 地址，默认 `http://127.0.0.1:8002`。
-   `GATEWAY_WORKER_ID`：Worker 稳定标识，默认 `vllm-local-0`。
-   `GATEWAY_MODEL_ID`：可服务模型，默认 `qwen2.5-0.5b`。

可选环境变量
------

-   `GATEWAY_METRICS_TIMEOUT_SECONDS`：健康/指标采集超时，默认 `5`。
-   `GATEWAY_WORKER_TIMEOUT_SECONDS`：上游推理流超时，默认 `30`。
-   `GATEWAY_SNAPSHOT_MAX_AGE_SECONDS`：Worker 快照最大时效，默认 `10`。

客户端复用
-----

`VLLMWorkerPool` 以 `(worker_id, base_url, model_id)` 作为缓存键：

-   键相同：复用同一个 `VLLMOpenAIWorker`，即复用长期 `httpx.AsyncClient` 与底层连接池。
-   Worker 地址或模型变化：建立新客户端，旧客户端在 Gateway 关闭时统一释放。
-   Gateway 进程关闭：调用 Pool 的 `aclose()`，关闭全部 HTTP 客户端。

启动命令
----

```
PYTHONPATH="$PWD"\
uvicorn serving.gateway.server:app\
  --host 0.0.0.0\
  --port 8003
```

启动后应检查：

```
curl -fsS http://127.0.0.1:8003/docs
```

真实聊天接口为：

```
POST /v1/chat/completions
```

当前只支持 `stream=true`。