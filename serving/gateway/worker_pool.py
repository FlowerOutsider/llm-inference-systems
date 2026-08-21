from __future__ import annotations

import asyncio

from serving.control_plane.worker_registry import WorkerSnapshot
from serving.data_plane.vllm_worker_adapter import VLLMOpenAIWorker


class VLLMWorkerPool:
    """
    复用 Gateway 到 vLLM Worker 的长期 HTTP 客户端。

    同一个 (worker_id, base_url, model_id) 组合只创建一个
    VLLMOpenAIWorker，因此不会按用户请求反复创建 AsyncClient。
    """

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._timeout_seconds = timeout_seconds
        self._workers: dict[
            tuple[str, str, str],
            VLLMOpenAIWorker,
        ] = {}

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def resolve(
        self,
        snapshot: WorkerSnapshot,
        model_id: str,
    ) -> VLLMOpenAIWorker:
        normalized_base_url = snapshot.base_url.rstrip("/")
        key = (
            snapshot.worker_id,
            normalized_base_url,
            model_id,
        )

        worker = self._workers.get(key)
        if worker is None:
            worker = VLLMOpenAIWorker(
                base_url=normalized_base_url,
                model=model_id,
                timeout_seconds=self._timeout_seconds,
            )
            self._workers[key] = worker

        return worker

    async def aclose(self) -> None:
        workers = tuple(self._workers.values())
        self._workers.clear()

        if workers:
            await asyncio.gather(
                *(worker.aclose() for worker in workers),
            )