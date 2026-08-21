from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerSnapshot:
    """某个推理 Worker 在一个时刻的可路由状态。"""

    worker_id: str
    base_url: str
    model_ids: frozenset[str]
    healthy: bool
    waiting_requests: int
    running_requests: int
    gpu_cache_usage_perc: float
    observed_at_monotonic: float

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model_ids:
            raise ValueError("model_ids must not be empty")
        if any(not model_id.strip() for model_id in self.model_ids):
            raise ValueError("model_ids must not contain empty values")
        if self.waiting_requests < 0:
            raise ValueError("waiting_requests must not be negative")
        if self.running_requests < 0:
            raise ValueError("running_requests must not be negative")
        if not 0.0 <= self.gpu_cache_usage_perc <= 1.0:
            raise ValueError(
                "gpu_cache_usage_perc must be between 0.0 and 1.0"
            )
        if self.observed_at_monotonic < 0:
            raise ValueError(
                "observed_at_monotonic must not be negative"
            )


class WorkerRegistry:
    """网关进程内的 Worker 快照注册表。"""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerSnapshot] = {}

    def upsert(self, snapshot: WorkerSnapshot) -> None:
        self._workers[snapshot.worker_id] = snapshot

    def remove(self, worker_id: str) -> WorkerSnapshot:
        return self._workers.pop(worker_id)

    def get(self, worker_id: str) -> WorkerSnapshot:
        return self._workers[worker_id]

    def all_workers(self) -> tuple[WorkerSnapshot, ...]:
        return tuple(self._workers.values())

    def workers_for_model(
        self,
        model_id: str,
    ) -> tuple[WorkerSnapshot, ...]:
        return tuple(
            worker
            for worker in self._workers.values()
            if model_id in worker.model_ids
        )