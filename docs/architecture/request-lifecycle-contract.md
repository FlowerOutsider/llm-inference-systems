# Request Lifecycle Contract

RequestLifecycleManager 持有活跃推理请求的 KV slot 所有权。

- submit：分配 slot，提交 Scheduler；若 Prefix Cache 命中，则复用已存在 KV block，并推进 prefill offset。
- cancel：标记请求取消并立即释放活跃 slot。
- finalize：仅允许 FINISHED 请求释放 slot。
- PrefixCacheManager 持有缓存 source slot；RequestLifecycleManager 持有在线请求 slot，两者不可混淆。
- PagedKVCache 通过 block refcount 保证释放 target slot 不会破坏共享 prefix block。