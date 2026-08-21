import asyncio
from dataclasses import dataclass, field

import pytest

from serving.control_plane.load_aware_router import LoadAwareRouter
from serving.control_plane.vllm_metrics_collector import WorkerRegistration
from serving.control_plane.worker_registry import (
    WorkerRegistry,
    WorkerSnapshot,
)
from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    StreamingChatCompletionEvent,
)
from serving.gateway.gateway_service import (
    GatewayChatRequest,
    GatewayNoAvailableWorkerError,
    GatewayService,
)


def run(coroutine):
    return asyncio.run(coroutine)


@dataclass
class FakeCollector:
    snapshots: dict[str, WorkerSnapshot]
    collected_worker_ids: list[str] = field(default_factory=list)

    async def collect_and_upsert(
        self,
        *,
        registry: WorkerRegistry,
        registration: WorkerRegistration,
    ) -> WorkerSnapshot:
        self.collected_worker_ids.append(registration.worker_id)
        snapshot = self.snapshots[registration.worker_id]
        registry.upsert(snapshot)
        return snapshot


@dataclass
class FakeWorker:
    events: list[StreamingChatCompletionEvent]
    calls: list[tuple[list[ChatMessage], int]] = field(default_factory=list)

    async def stream_chat_completion(
        self,
        *,
        messages: list[ChatMessage],
        max_tokens: int,
    ):
        self.calls.append((messages, max_tokens))
        for event in self.events:
            yield event


@dataclass
class FakeWorkerResolver:
    workers: dict[str, FakeWorker]
    resolved: list[tuple[str, str]] = field(default_factory=list)

    def __call__(
        self,
        snapshot: WorkerSnapshot,
        model_id: str,
    ) -> FakeWorker:
        self.resolved.append((snapshot.worker_id, model_id))
        return self.workers[snapshot.worker_id]


def make_snapshot(
    worker_id: str,
    *,
    healthy: bool = True,
    waiting_requests: int = 0,
    running_requests: int = 0,
    gpu_cache_usage_perc: float = 0.0,
) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        base_url=f"http://{worker_id}:8000",
        model_ids=frozenset({"qwen2.5-0.5b"}),
        healthy=healthy,
        waiting_requests=waiting_requests,
        running_requests=running_requests,
        gpu_cache_usage_perc=gpu_cache_usage_perc,
        observed_at_monotonic=100.0,
    )


def make_request(
    *,
    prefix_candidate_worker_ids: frozenset[str] = frozenset(),
) -> GatewayChatRequest:
    return GatewayChatRequest(
        request_id="gateway-request-1",
        model_id="qwen2.5-0.5b",
        messages=(ChatMessage(role="user", content="hello"),),
        max_tokens=16,
        prefix_candidate_worker_ids=prefix_candidate_worker_ids,
    )


async def collect_events(
    service: GatewayService,
    request: GatewayChatRequest,
) -> list[StreamingChatCompletionEvent]:
    return [
        event
        async for event in service.stream_chat_completion(request)
    ]


def make_service(
    *,
    snapshots: dict[str, WorkerSnapshot],
    workers: dict[str, FakeWorker],
) -> tuple[GatewayService, FakeCollector, FakeWorkerResolver]:
    registry = WorkerRegistry()
    collector = FakeCollector(snapshots=snapshots)
    resolver = FakeWorkerResolver(workers=workers)

    registrations = tuple(
        WorkerRegistration(
            worker_id=snapshot.worker_id,
            base_url=snapshot.base_url,
            model_ids=snapshot.model_ids,
        )
        for snapshot in snapshots.values()
    )

    service = GatewayService(
        registry=registry,
        router=LoadAwareRouter(
            registry=registry,
            clock=lambda: 100.0,
        ),
        collector=collector,
        registrations=registrations,
        worker_resolver=resolver,
    )
    return service, collector, resolver


def test_gateway_refreshes_routes_and_streams_selected_worker() -> None:
    event = StreamingChatCompletionEvent(
        request_id="chatcmpl-1",
        delta_content="hello",
        finish_reason="stop",
    )
    worker = FakeWorker(events=[event])
    service, collector, resolver = make_service(
        snapshots={"worker-a": make_snapshot("worker-a")},
        workers={"worker-a": worker},
    )

    events = run(collect_events(service, make_request()))

    assert events == [event]
    assert collector.collected_worker_ids == ["worker-a"]
    assert resolver.resolved == [("worker-a", "qwen2.5-0.5b")]
    assert worker.calls == [
        ([ChatMessage(role="user", content="hello")], 16)
    ]


def test_gateway_passes_prefix_candidate_to_router() -> None:
    prefix_event = StreamingChatCompletionEvent(
        request_id="chatcmpl-prefix",
        delta_content="cached",
        finish_reason="stop",
    )
    prefix_worker = FakeWorker(events=[prefix_event])
    cold_worker = FakeWorker(events=[])

    service, _, resolver = make_service(
        snapshots={
            "prefix-worker": make_snapshot(
                "prefix-worker",
                running_requests=2,
                gpu_cache_usage_perc=0.4,
            ),
            "cold-worker": make_snapshot(
                "cold-worker",
                gpu_cache_usage_perc=0.4,
            ),
        },
        workers={
            "prefix-worker": prefix_worker,
            "cold-worker": cold_worker,
        },
    )

    events = run(
        collect_events(
            service,
            make_request(
                prefix_candidate_worker_ids=frozenset(
                    {"prefix-worker"}
                )
            ),
        )
    )

    assert events == [prefix_event]
    assert resolver.resolved == [
        ("prefix-worker", "qwen2.5-0.5b")
    ]


def test_gateway_rejects_request_without_healthy_worker() -> None:
    worker = FakeWorker(events=[])
    service, _, resolver = make_service(
        snapshots={
            "unhealthy-worker": make_snapshot(
                "unhealthy-worker",
                healthy=False,
            )
        },
        workers={"unhealthy-worker": worker},
    )

    with pytest.raises(
        GatewayNoAvailableWorkerError,
        match="no eligible worker",
    ):
        run(collect_events(service, make_request()))

    assert resolver.resolved == []


def test_gateway_refreshes_worker_snapshots_for_every_request() -> None:
    worker = FakeWorker(
        events=[
            StreamingChatCompletionEvent(
                request_id="chatcmpl-repeat",
                delta_content="ok",
                finish_reason="stop",
            )
        ]
    )
    service, collector, _ = make_service(
        snapshots={"worker-a": make_snapshot("worker-a")},
        workers={"worker-a": worker},
    )

    run(collect_events(service, make_request()))
    run(collect_events(service, make_request()))

    assert collector.collected_worker_ids == [
        "worker-a",
        "worker-a",
    ]