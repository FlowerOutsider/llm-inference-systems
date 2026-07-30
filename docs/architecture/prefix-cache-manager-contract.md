
# Prefix Cache Manager Contract

## 目标

PrefixCacheManager 是推理服务控制面中的缓存策略组件。
它不存储 GPU KV 数据，而是基于 PagedKVCache 提供的块管理能力，
负责 Prefix Cache 的准入、淘汰、容量保护与统计。

## 职责边界

- PagedKVCache：KV 数据、分页块、引用计数、Copy-on-Write。
- PrefixCacheIndex：按模型和 token 前缀进行最长匹配。
- PrefixCacheCoordinator：连接索引与数据面，执行 prefix attach 和 slot release。
- PrefixCacheManager：准入策略、LRU 淘汰、空闲块水位线、策略统计。

## 准入规则

1. token 数小于 `min_prefix_tokens` 时拒绝缓存。
2. 缓存条目达到 `max_entries` 时，淘汰最久未使用条目。
3. 空闲 GPU KV 块少于 `min_free_blocks` 时，持续淘汰 LRU 条目。
4. 淘汰后仍不满足容量条件时，拒绝新条目。

## 生命周期

1. 请求完成 Prefill，并将 KV 写入 source slot。
2. Manager 调用 Coordinator 发布该 slot 的 prefix 元数据。
3. 后续请求根据 scope 和 token 序列查找最长可复用前缀。
4. 命中后，Coordinator 调用 PagedKVCache.fork_prefix 共享完整 KV 块。
5. 若后续 Decode 写入共享尾块，PagedKVCache 执行 Copy-on-Write。
6. LRU 淘汰时，Manager 调用 Coordinator.release_slots。
7. PagedKVCache 根据 block refcount 决定物理块是否真正回收。

## 安全不变量

- 活跃请求 slot 与缓存 source slot 的所有权必须区分。
- 释放缓存 source slot 不得破坏仍在使用共享块的请求。
- prefix token 数必须与 source slot 中已提交的 KV 长度一致。
- 缓存命中只能发生在相同 PrefixScope 内。
- 不完整多层 KV 写入不能被发布为可复用前缀。

## 当前限制

- 当前 Manager 是单进程、内存内策略实现。
- 尚未加入并发锁、TTL、跨副本目录、分层 CPU/NVMe KV Cache。
- 尚未接入真实 vLLM/SGLang 的调度器和 PagedAttention kernel。
