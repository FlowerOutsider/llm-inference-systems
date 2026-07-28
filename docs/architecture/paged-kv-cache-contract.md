
# Paged KV Cache 契约

## 动机

连续 KV Cache 为每个请求预留完整上下文窗口。请求长度不均衡时，未使用的尾部容量形成内部碎片。

Paged KV Cache 将序列切分为固定长度 token block，仅在请求实际增长时分配物理 block。

## 物理布局

K 和 V 分别预分配为：

```text
[num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim]
```

每个 block 可保存 `block_size` 个 token 的 K 或 V。

逻辑布局
----

每个活跃请求 slot 持有：

```
sequence_length
block_table = [physical_block_id_0, physical_block_id_1, ...]
```

token position 到物理位置的映射：

```
logical_block_index = token_position // block_size
offset_in_block = token_position % block_size
physical_block_id = block_table[logical_block_index]
```

生命周期不变量
-------

-   block 只能从 free list 分配，释放后才能再次分配。
-   活跃 slot 的 block table 不得含有重复或无效 block。
-   每个 slot 的逻辑长度不得超过 `max_sequence_length`。
-   追加 token 前必须先保证所需 block 已分配。
-   `release(slot)` 必须归还该 slot 独占的全部 block。
-   block table 使用 `-1` 表示未映射。

内存模型
----

物理池容量：

```
num_layers × 2 × num_gpu_blocks × block_size
× num_kv_heads × head_dim × dtype_bytes
```

逻辑请求长度不再直接决定预留容量，而是按：

```
ceil(sequence_length / block_size)
```

分配 block。

第一版边界
-----

第一版实现真实 block allocator、slot 生命周期、批量 append 与 block table。

第一版 `get_kv` 可以通过 gather 重建连续 tensor，仅用于正确性验证；生产级 paged-attention kernel 不应在每次 decode 时 materialize 整段连续 KV。

后续演进
----

1.  block refcount。
2.  Prefix Cache 共享与 Copy-on-Write。
3.  CUDA/Triton paged-attention kernel。
4.  Continuous Batching。
5.  多 GPU KV Router 与 Prefill/Decode 分离。


```
这份契约完成后，我们会实现第一版 block allocator，并让它通过比连续 Cache 更严格的测试：跨 block 追加、block 耗尽、释放归还、slot 复用和 block table 映射正确性。
```

## 多层追加事务

一个 Transformer token 的 K/V 会在每一层分别生成，但请求的逻辑长度只能推进一次。

begin_append
    -> 预检容量并分配新增 block，不推进 sequence length

write_layer(layer_idx)
    -> 每层写入同一段已预留的逻辑位置

commit_append
    -> 确认全部 layer 已完成写入后，统一推进 sequence length

abort_append
    -> 若任意 layer 失败，归还本次新增 block，不暴露未提交 token