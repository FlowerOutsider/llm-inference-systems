import asyncio

from serving.control_plane.worker_registry import WorkerSnapshot
from serving.gateway.worker_pool import VLLMWorkerPool


def run(coroutine):
    return asyncio.run(coroutine)


def make_snapshot(
    *,
    worker_id: str = "worker-a",
    base_url: str = "http://worker-a:8000",
) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        base_url=base_url,
        model_ids=frozenset({"qwen2.5-0.5b", "deepseek-r1"}),
        healthy=True,
        waiting_requests=0,
        running_requests=0,
        gpu_cache_usage_perc=0.0,
        observed_at_monotonic=100.0,
    )


def test_pool_reuses_worker_for_same_worker_address_and_model() -> None:
    pool = VLLMWorkerPool(timeout_seconds=12.0)
    snapshot = make_snapshot()

    first = pool.resolve(snapshot, "qwen2.5-0.5b")
    second = pool.resolve(snapshot, "qwen2.5-0.5b")

    try:
        assert first is second
        assert pool.worker_count == 1
    finally:
        run(pool.aclose())


def test_pool_creates_distinct_workers_for_different_models() -> None:
    pool = VLLMWorkerPool()
    snapshot = make_snapshot()

    qwen_worker = pool.resolve(snapshot, "qwen2.5-0.5b")
    deepseek_worker = pool.resolve(snapshot, "deepseek-r1")

    try:
        assert qwen_worker is not deepseek_worker
        assert pool.worker_count == 2
    finally:
        run(pool.aclose())


def test_pool_creates_new_worker_when_address_changes() -> None:
    pool = VLLMWorkerPool()
    original_snapshot = make_snapshot()
    moved_snapshot = make_snapshot(
        base_url="http://worker-a-new:8000",
    )

    original_worker = pool.resolve(
        original_snapshot,
        "qwen2.5-0.5b",
    )
    moved_worker = pool.resolve(
        moved_snapshot,
        "qwen2.5-0.5b",
    )

    try:
        assert original_worker is not moved_worker
        assert pool.worker_count == 2
    finally:
        run(pool.aclose())