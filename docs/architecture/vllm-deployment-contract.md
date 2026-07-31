# vLLM 部署契约

## 目标

本部署使用固定版本的官方 vLLM 镜像运行真实 OpenAI 兼容推理服务。部署配置必须能够在具有 NVIDIA GPU、Docker 和已缓存模型权重的主机上复现。

## 工件边界

- 镜像：提供 vLLM、CUDA 运行时和服务程序。
- Compose 文件：声明 GPU、端口、健康检查、卷挂载和服务参数。
- `.env`：保存当前环境的模型与资源参数，不提交到 Git。
- Hugging Face 缓存：保存模型权重与 tokenizer，不打入镜像。
- 编译缓存卷：保存 vLLM 的 torch.compile 产物，避免每次重启重新编译。

## 固定运行参数

| 参数 | 当前值 | 影响 |
|---|---:|---|
| 镜像 | `vllm/vllm-openai:v0.8.5` | 固定运行时与框架版本 |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 当前真实推理后端 |
| 数据类型 | `half` | 降低显存占用并使用 GPU 半精度路径 |
| 显存利用率目标 | `0.70` | 为 KV Cache 预留显存，同时保留稳定性余量 |
| 最大上下文 | `2048` | 单请求 prompt 与生成 token 的总上限 |
| 最大序列数 | `16` | 调度器允许的最大并发活跃序列数 |
| 最大批 token 数 | `2048` | 单次调度的 token 预算，影响 prefill 与连续批处理 |
| Prefix Cache | 启用 | 复用共享 prompt 的 KV blocks |
| Prefix Hash | `sha256` | 降低多租户场景的 hash collision 风险 |
| 请求日志 | 禁用 | 避免在服务日志中记录用户 prompt |

## 运行原则

1. 模型权重不烘焙进镜像，避免镜像膨胀和模型更新时重建镜像。
2. 不使用 `latest` 标签，避免运行时版本漂移。
3. Prefix Cache 必须显式启用或显式关闭；在 vLLM V1 中省略参数不等于关闭。
4. 修改 `MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS` 或显存预算后，必须重新执行压测。
5. 生产服务应保留健康检查、优雅停止时间和持久化编译缓存。
6. 压测或调试日志不能包含真实用户 prompt、密钥或敏感业务数据。

## 启动与验证

```bash
cp infra/docker/vllm/.env.example infra/docker/vllm/.env

docker compose \
  --env-file infra/docker/vllm/.env \
  -f infra/docker/vllm/compose.yaml \
  config


  通过静态配置校验后，启动服务：

```
docker compose\
  --env-file infra/docker/vllm/.env\
  -f infra/docker/vllm/compose.yaml\
  up -d
```

验证服务：

```
curl -fsS http://127.0.0.1:8002/v1/models
```