import asyncio

import httpx

from serving.control_plane.vllm_metrics_collector import (
    VLLMMetricsCollector,
    WorkerRegistration,
)


def run(coroutine):
    return asyncio.run(coroutine)


def make_registration() -> WorkerRegistration:
    return WorkerRegistration(
        worker_id="worker-a",
        base_url="http://worker-a:8000",
        model_ids=frozenset({"qwen2.5-0.5b"}),
    )


def make_collector(handler) -> VLLMMetricsCollector:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    )
    return VLLMMetricsCollector(
        client=client,
        clock=lambda: 100.0,
    )


def test_collects_healthy_vllm_snapshot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(status_code=200)

        assert request.url.path == "/metrics"
        return httpx.Response(
            status_code=200,
            text="\n".join(
                [
                    "vllm:num_requests_waiting 3",
                    "vllm:num_requests_running 2",
                    "vllm:gpu_cache_usage_perc 0.42",
                ]
            ),
        )

    collector = make_collector(handler)

    try:
        snapshot = run(collector.collect(make_registration()))
    finally:
        run(collector.aclose())

    assert snapshot.worker_id == "worker-a"
    assert snapshot.healthy is True
    assert snapshot.waiting_requests == 3
    assert snapshot.running_requests == 2
    assert snapshot.gpu_cache_usage_perc == 0.42
    assert snapshot.observed_at_monotonic == 100.0


def test_aggregates_queue_metrics_and_uses_maximum_cache_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(status_code=200)

        return httpx.Response(
            status_code=200,
            text="\n".join(
                [
                    'vllm:num_requests_waiting{engine="0"} 1',
                    'vllm:num_requests_waiting{engine="1"} 2',
                    'vllm:num_requests_running{engine="0"} 3',
                    'vllm:num_requests_running{engine="1"} 4',
                    'vllm:gpu_cache_usage_perc{engine="0"} 0.25',
                    'vllm:gpu_cache_usage_perc{engine="1"} 0.80',
                ]
            ),
        )

    collector = make_collector(handler)

    try:
        snapshot = run(collector.collect(make_registration()))
    finally:
        run(collector.aclose())

    assert snapshot.healthy is True
    assert snapshot.waiting_requests == 3
    assert snapshot.running_requests == 7
    assert snapshot.gpu_cache_usage_perc == 0.80


def test_marks_worker_unhealthy_when_health_check_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(status_code=503)

    collector = make_collector(handler)

    try:
        snapshot = run(collector.collect(make_registration()))
    finally:
        run(collector.aclose())

    assert snapshot.healthy is False
    assert snapshot.waiting_requests == 0
    assert snapshot.running_requests == 0
    assert snapshot.gpu_cache_usage_perc == 0.0


def test_marks_worker_unhealthy_when_required_metric_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(status_code=200)

        return httpx.Response(
            status_code=200,
            text="\n".join(
                [
                    "vllm:num_requests_waiting 1",
                    "vllm:num_requests_running 2",
                ]
            ),
        )

    collector = make_collector(handler)

    try:
        snapshot = run(collector.collect(make_registration()))
    finally:
        run(collector.aclose())

    assert snapshot.healthy is False


def test_marks_worker_unhealthy_on_transport_error() -> None:
    class FailingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self,
            request: httpx.Request,
        ) -> httpx.Response:
            raise httpx.ConnectError(
                "connection refused",
                request=request,
            )

    client = httpx.AsyncClient(
        transport=FailingTransport(),
    )
    collector = VLLMMetricsCollector(
        client=client,
        clock=lambda: 100.0,
    )

    try:
        snapshot = run(collector.collect(make_registration()))
    finally:
        run(collector.aclose())

    assert snapshot.healthy is False