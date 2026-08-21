from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from serving.control_plane.worker_registry import (
    WorkerRegistry,
    WorkerSnapshot,
)


class NoHealthyWorkerError(RuntimeError):
    """没有满足模型、健康和时效要求的可路由 Worker。"""


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    model_id: str
    prefix_candidate_worker_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")

        object.__setattr__(
            self,
            "prefix_candidate_worker_ids",
            frozenset(self.prefix_candidate_worker_ids),
        )


@dataclass(frozen=True)
class RoutingDecision:
    worker_id: str
    score: float
    prefix_cache_candidate: bool
    eligible_worker_count: int


class LoadAwareRouter:
    """
    基于排队压力、执行压力、KV Cache 使用率和前缀位置的 Worker 路由器。

    分数越低越优；同分时使用 worker_id 进行确定性排序。
    """

    _WAITING_REQUEST_WEIGHT = 4.0
    _RUNNING_REQUEST_WEIGHT = 1.0
    _GPU_CACHE_USAGE_WEIGHT = 5.0
    _PREFIX_CACHE_BONUS = 4.0

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        max_snapshot_age_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_snapshot_age_seconds <= 0:
            raise ValueError(
                "max_snapshot_age_seconds must be positive"
            )

        self._registry = registry
        self._max_snapshot_age_seconds = max_snapshot_age_seconds
        self._clock = clock

    def route(self, request: RoutingRequest) -> RoutingDecision:
        now = self._clock()
        eligible_workers = [
            worker
            for worker in self._registry.workers_for_model(request.model_id)
            if self._is_eligible(worker, now=now)
        ]

        if not eligible_workers:
            raise NoHealthyWorkerError(
                f"no eligible worker for model {request.model_id!r}"
            )

        scored_workers = [
            (
                self._score(
                    worker,
                    prefix_cache_candidate=(
                        worker.worker_id
                        in request.prefix_candidate_worker_ids
                    ),
                ),
                worker.worker_id,
                worker,
            )
            for worker in eligible_workers
        ]
        score, _, selected_worker = min(scored_workers)

        return RoutingDecision(
            worker_id=selected_worker.worker_id,
            score=score,
            prefix_cache_candidate=(
                selected_worker.worker_id
                in request.prefix_candidate_worker_ids
            ),
            eligible_worker_count=len(eligible_workers),
        )

    def _is_eligible(
        self,
        worker: WorkerSnapshot,
        *,
        now: float,
    ) -> bool:
        if not worker.healthy:
            return False

        snapshot_age_seconds = max(
            0.0,
            now - worker.observed_at_monotonic,
        )
        return snapshot_age_seconds <= self._max_snapshot_age_seconds

    def _score(
        self,
        worker: WorkerSnapshot,
        *,
        prefix_cache_candidate: bool,
    ) -> float:
        score = (
            worker.waiting_requests * self._WAITING_REQUEST_WEIGHT
            + worker.running_requests * self._RUNNING_REQUEST_WEIGHT
            + worker.gpu_cache_usage_perc * self._GPU_CACHE_USAGE_WEIGHT
        )

        if prefix_cache_candidate:
            score -= self._PREFIX_CACHE_BONUS

        return score