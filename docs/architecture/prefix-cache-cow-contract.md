
# Prefix Cache、Block Refcount 与 Copy-on-Write 契约

## 目标

在 Paged KV Cache 上复用多个请求的公共前缀，降低重复 Prefill 计算与 KV Cache 显存占用。

## Prefix Cache Key

只有以下上下文完全一致时，KV block 才允许复用：

```text
model identity
model revision / weights revision
token IDs
tokenizer revision
RoPE configuration
attention implementation and KV layout
adapter / LoRA identity
```

token 文本相同但 tokenizer、模型版本或 LoRA 不同，均不得复用 KV Cache。

Block 级复用
---------

Prefix Cache 优先共享完整 block：

```
block_size = 16
prefix length = 35

共享 block:
[0..15], [16..31]

未共享尾部:
[32..34]
```

完整 block 共享避免了追加 token 时覆盖其他请求的数据。

Reference Count
---------------

每个物理 block 维护引用计数：

```
block allocated for one request: refcount = 1
block attached to another request: refcount += 1
request release: refcount -= 1
refcount == 0: return block to free list
```

任何被活跃请求或 Prefix Cache 索引引用的 block 都不得被回收或覆盖。

Copy-on-Write
-------------

如果两个请求共享一个未填满的尾部 block，而某个请求需要继续写入该 block：

1.  分配新物理 block。
2.  复制该 block 在所有 layer 的 K/V 数据。
3.  将写入请求的 block table 改指向新 block。
4.  原 block 的 refcount 减一，新 block 的 refcount 为一。
5.  继续追加 token。

这样一个请求的 decode 写入不会污染另一个请求的 KV。

事务约束
----

`begin_append` 必须在 block 分配与 COW 完成后返回 reservation。

`abort_append` 必须回滚：

-   新分配的尾部 block；
-   COW 替换后的 block table；
-   相关 block refcount。

`commit_append` 仅推进逻辑 sequence length，不改变已经完成的 block 所有权关系。

后续 Prefix Index
---------------

Prefix Index 保存 token block hash 到物理 block 的映射，并具有：

-   模型隔离；
-   LRU / TTL 驱逐；
-   引用计数保护；
-   命中率、节省 Prefill token 数等可观测指标。



```
这份设计明确了一个面试里很关键的判断：

> Prefix Cache 不是"相同文本就复用"，而是"相同模型执行语义下的相同 token 前缀对应的 KV block 才可复用"。

下一步我们会先给 `PagedKVCache` 加 block refcount 和 `fork_prefix()`，再实现 partial block 的 COW。这样 P
```


## 当前实现边界

当前 `PagedKVCache` 已实现单 GPU 进程内的物理块共享与引用计数：

- `fork_prefix` 允许共享任意正长度前缀，包括最后一个不完整块。
- 完整块共享不产生新的 K/V 物理块。
- 若追加写入会覆盖一个引用计数大于 1 的不完整尾块，`begin_append` 必须先执行 Copy-on-Write。
- COW 会复制该物理块中所有 layer 的 K/V 数据，并仅替换当前 slot 的 block table 映射。
- `commit_append` 后新块成为该 slot 的私有尾块。
- `abort_append` 必须恢复原 block table、原块引用计数和空闲块池状态。
- `release` 仅在物理块引用计数降至 0 时才将其归还给 free block pool。

## 当前非目标

- 不支持跨进程或跨节点 KV Cache 共享。
- 不包含 token prefix 哈希索引、LRU、TTL、租户隔离或逐出策略。
- 不包含 GPU kernel 直接消费 block table 的 PagedAttention 实现；`get_kv` 仅用于正确性验证，会临时 gather 为连续张量。
- 不支持并发线程安全控制；当前元数据控制面假设由单个调度线程串行访问。