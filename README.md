# LLM Inference Systems

面向大模型高性能推理与强化学习基础设施的工程项目。

项目以 LLM 推理引擎研发和 AI Infra 正式岗能力要求为标准，覆盖推理调度、GPU Kernel、分布式 KV Cache、性能优化、模型压缩、生产可靠性与 RL Infra。

## 目标能力

- 理解并实践 vLLM、SGLang、TensorRT-LLM 等推理系统核心机制。
- 掌握 Prefill、Decode、KV Cache、PagedAttention、Continuous Batching、Chunked Prefill、Prefix Cache 与 PD Disaggregation。
- 使用 CUDA、Triton、Nsight 实现并分析高性能算子。
- 掌握 NCCL、张量并行、通信分析与分布式推理基础。
- 实践量化、推测解码、KV Cache 管理与推理成本优化。
- 构建具备可观测性、SLO、故障恢复和弹性扩缩容能力的推理系统。
- 研究 veRL、Ray、rollout 与训练推理协同的 RL Infra 链路。

## Repository Layout

```text
serving/       推理服务、调度、路由与 PD 分离
kernels/       CUDA 与 Triton 算子实验
distributed/   通信、并行与分布式 KV Cache
rl_infra/      rollout、调度与强化学习基础设施
infra/         Docker、Kubernetes 与可观测性
benchmarks/    性能实验、基准脚本与结果
docs/          架构设计、技术决策与实验报告
scripts/       可复现实验脚本
tests/         单元、集成与回归测试
```

Compute Strategy
----------------

-   本地 RTX 3060：CUDA、Triton、Profiling、轻量 PyTorch 与量化小模型推理。
-   AutoDL：多 GPU 推理、NCCL、PD 分离、TensorRT-LLM、Kubernetes、Ray 与 RL Infra 实验。

Engineering Principles
----------------------

-   最终实验必须基于真实模型、真实 GPU 与可复现工作负载。
-   每项优化必须提供基线、方法、指标、结果和瓶颈分析。
-   Mock 仅用于测试与故障注入，不作为性能结论或项目主体。
-   代码、实验配置、监控指标和技术结论必须可追溯。

Current Status
--------------

-   WSL、Docker Desktop 与 NVIDIA GPU 容器链路验证完成。
-   CUDA Toolkit 与 Profiling 工具链。
-   推理引擎部署与源码分析。
-   CUDA/Triton Kernel 优化。
-   分布式推理与 KV Cache 实验。
-   RL Infra 与训练推理协同。