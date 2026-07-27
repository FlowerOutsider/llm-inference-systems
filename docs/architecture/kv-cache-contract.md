# 连续 KV Cache 基线契约

## 目标

实现 GPU 上预分配的多层 K/V Cache，用于验证请求 slot 管理、批量追加 token、容量控制和缓存复用。

这是 Paged KV Cache 的对照基线，不是最终方案。

## 物理布局

K 和 V 分别预分配为：

```text
[num_layers, max_slots, max_sequence_length, num_kv_heads, head_dim]