# Prefix Index Contract

## 目标

Prefix Index 在请求进入推理调度器时，为给定模型与 token 序列找到最长可复用的 KV Cache 前缀。

## 隔离边界

索引键必须同时包含：

- model_id
- model_revision
- tokenizer_revision
- rope_config
- token_ids

不同模型版本、Tokenizer 版本或 RoPE 配置不得共享 KV Cache，即使 token ID 序列相同。

## 查询语义

- 查询返回相同执行语义下的最长 token 前缀。
- 命中结果包含 source_slot、prefix_length 和稳定 fingerprint。
- 未命中返回 `None`。
- 相同 scope 下较短前缀可作为较长请求的降级命中。
- 索引匹配必须基于完整 token 序列比较，哈希仅用于索引加速，不能单独作为正确性判断。

## 生命周期

- source_slot 被释放前，调度器必须调用 `remove_slot` 清理该 slot 的所有索引项。
- 被清理的条目不能再返回给后续请求。
- 本阶段不实现 LRU、TTL、跨进程同步和持久化。

## 指标

Index 至少记录：

- lookups
- hits
- misses
- reused_tokens
- registered_entries
- evictions