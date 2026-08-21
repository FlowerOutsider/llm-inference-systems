import pytest

from serving.control_plane.load_aware_router import (
    LoadAwareRouter,
    NoHealthyWorkerError,
    RoutingRequest,
)
from serving.control_plane.worker_registry import (
    WorkerRegistry,
    WorkerSnapshot,
)


def make_snapshot(
    worker_id: str,
    *,
    models: tuple[str, ...] = ("qwen2.5-0.5b",),
    healthy: bool = True,
    waiting_requests: int = 0,
    running_requests: int = 0,
    gpu_cache_usage_perc: float = 0.0,
    observed_at_monotonic: float = 100.0,
) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        base_url=f"http://{worker_id}:8000",
        model_ids=frozenset(models),
        healthy=healthy,
        waiting_requests=waiting_requests,
        running_requests=running_requests,
        gpu_cache_usage_perc=gpu_cache_usage_perc,
        observed_at_monotonic=observed_at_monotonic,
    )


def make_router(
    registry: WorkerRegistry,
    *,
    now: float = 100.0,
) -> LoadAwareRouter:
    return LoadAwareRouter(
        registry=registry,
        max_snapshot_age_seconds=10.0,
        clock=lambda: now,
    )


def test_router_selects_only_healthy_worker_that_supports_model() -> None:
    registry = WorkerRegistry()
    registry.upsert(make_snapshot("qwen-worker"))
    registry.upsert(
        make_snapshot(
            "other-model-worker",
            models=("deepseek-r1",),
        )
    )
    registry.upsert(
        make_snapshot(
            "unhealthy-qwen-worker",
            healthy=False,
        )
    )

    decision = make_router(registry).route(
        RoutingRequest(
            request_id="request-1",
            model_id="qwen2.5-0.5b",
        )
    )

    assert decision.worker_id == "qwen-worker"
    assert decision.prefix_cache_candidate is False


def test_router_prefers_prefix_candidate_when_load_is_comparable() -> None:
    registry = WorkerRegistry()
    registry.upsert(
        make_snapshot(
            "prefix-worker",
            running_requests=2,
            gpu_cache_usage_perc=0.4,
        )
    )
    registry.upsert(
        make_snapshot(
            "cold-worker",
            running_requests=0,
            gpu_cache_usage_perc=0.4,
        )
    )

    decision = make_router(registry).route(
        RoutingRequest(
            request_id="request-prefix",
            model_id="qwen2.5-0.5b",
            prefix_candidate_worker_ids=frozenset({"prefix-worker"}),
        )
    )

    assert decision.worker_id == "prefix-worker"
    assert decision.prefix_cache_candidate is True


def test_router_avoids_overloaded_prefix_candidate() -> None:
    registry = WorkerRegistry()
    registry.upsert(
        make_snapshot(
            "overloaded-prefix-worker",
            waiting_requests=4,
            running_requests=2,
            gpu_cache_usage_perc=0.4,
        )
    )
    registry.upsert(
        make_snapshot(
            "available-worker",
            gpu_cache_usage_perc=0.4,
        )
    )

    decision = make_router(registry).route(
        RoutingRequest(
            request_id="request-overloaded-prefix",
            model_id="qwen2.5-0.5b",
            prefix_candidate_worker_ids=frozenset(
                {"overloaded-prefix-worker"}
            ),
        )
    )

    assert decision.worker_id == "available-worker"
    assert decision.prefix_cache_candidate is False


def test_router_rejects_when_no_healthy_matching_worker_exists() -> None:
    registry = WorkerRegistry()
    registry.upsert(
        make_snapshot(
            "unhealthy-worker",
            healthy=False,
        )
    )

    with pytest.raises(
        NoHealthyWorkerError,
        match="no eligible worker",
    ):
        make_router(registry).route(
            RoutingRequest(
                request_id="request-no-worker",
                model_id="qwen2.5-0.5b",
            )
        )


def test_router_rejects_stale_worker_snapshot() -> None:
    registry = WorkerRegistry()
    registry.upsert(
        make_snapshot(
            "stale-worker",
            observed_at_monotonic=80.0,
        )
    )

    with pytest.raises(
        NoHealthyWorkerError,
        match="no eligible worker",
    ):
        make_router(registry, now=100.0).route(
            RoutingRequest(
                request_id="request-stale",
                model_id="qwen2.5-0.5b",
            )
        )


def test_router_breaks_equal_scores_by_worker_id() -> None:
    registry = WorkerRegistry()
    registry.upsert(make_snapshot("worker-b"))
    registry.upsert(make_snapshot("worker-a"))

    decision = make_router(registry).route(
        RoutingRequest(
            request_id="request-tie",
            model_id="qwen2.5-0.5b",
        )
    )

    assert decision.worker_id == "worker-a"