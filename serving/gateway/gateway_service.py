from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from serving.control_plane.load_aware_router import (
    LoadAwareRouter,
    NoHealthyWorkerError,
    RoutingRequest,
)
from serving.control_plane.vllm_metrics_collector import (
    VLLMMetricsCollector,
    WorkerRegistration,
)
from serving.control_plane.worker_registry import (
    WorkerRegistry,
    WorkerSnapshot,
)
from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    StreamingChatCompletionEvent,
)


class GatewayNoAvailableWorkerError(RuntimeError):
    """网关没有可用于处理该模型请求的 Worker。"""


@dataclass(frozen=True)
class GatewayChatRequest:
    request_id: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int
    prefix_candidate_worker_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(
            self,
            "prefix_candidate_worker_ids",
            frozenset(self.prefix_candidate_worker_ids),
        )


class StreamingWorker(Protocol):
    async def stream_chat_completion(
        self,
        *,
        messages: list[ChatMessage],
        max_tokens: int,
    ) -> AsyncIterator[StreamingChatCompletionEvent]: ...


class WorkerResolver(Protocol):
    def __call__(
        self,
        snapshot: WorkerSnapshot,
        model_id: str,
    ) -> StreamingWorker: ...


class MetricsCollector(Protocol):
    async def collect_and_upsert(
        self,
        *,
        registry: WorkerRegistry,
        registration: WorkerRegistration,
    ) -> WorkerSnapshot: ...


class GatewayService:
    """
    串联 Worker 快照刷新、路由决策和流式 Worker 调用的应用服务层。

    当前版本在每个请求前刷新快照，优先保证端到端语义正确。
    后续将替换为后台周期刷新任务。
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        router: LoadAwareRouter,
        collector: MetricsCollector,
        registrations: tuple[WorkerRegistration, ...],
        worker_resolver: WorkerResolver,
    ) -> None:
        if not registrations:
            raise ValueError("registrations must not be empty")

        worker_ids = [registration.worker_id for registration in registrations]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("registrations must have unique worker_id values")

        self._registry = registry
        self._router = router
        self._collector = collector
        self._registrations = registrations
        self._worker_resolver = worker_resolver

    async def stream_chat_completion(
        self,
        request: GatewayChatRequest,
    ) -> AsyncIterator[StreamingChatCompletionEvent]:
        await self._refresh_worker_snapshots()

        try:
            decision = self._router.route(
                RoutingRequest(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    prefix_candidate_worker_ids=(
                        request.prefix_candidate_worker_ids
                    ),
                )
            )
        except NoHealthyWorkerError as exc:
            raise GatewayNoAvailableWorkerError(str(exc)) from exc

        selected_worker = self._registry.get(decision.worker_id)
        worker = self._worker_resolver(
            selected_worker,
            request.model_id,
        )

        async for event in worker.stream_chat_completion(
            messages=list(request.messages),
            max_tokens=request.max_tokens,
        ):
            yield event

    async def _refresh_worker_snapshots(self) -> None:
        for registration in self._registrations:
            await self._collector.collect_and_upsert(
                registry=self._registry,
                registration=registration,
            )