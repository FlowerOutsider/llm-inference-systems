from __future__ import annotations

import os

from serving.control_plane.load_aware_router import LoadAwareRouter
from serving.control_plane.vllm_metrics_collector import (
    VLLMMetricsCollector,
    WorkerRegistration,
)
from serving.control_plane.worker_registry import WorkerRegistry
from serving.gateway.api import create_app
from serving.gateway.gateway_service import GatewayService
from serving.gateway.worker_pool import VLLMWorkerPool


def _read_positive_float(
    name: str,
    *,
    default: float,
) -> float:
    raw_value = os.getenv(name, str(default))

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive number, got {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def build_app():
    worker_id = os.getenv("GATEWAY_WORKER_ID", "vllm-local-0")
    base_url = os.getenv(
        "GATEWAY_VLLM_BASE_URL",
        "http://127.0.0.1:8002",
    )
    model_id = os.getenv("GATEWAY_MODEL_ID", "qwen2.5-0.5b")

    metrics_timeout_seconds = _read_positive_float(
        "GATEWAY_METRICS_TIMEOUT_SECONDS",
        default=5.0,
    )
    worker_timeout_seconds = _read_positive_float(
        "GATEWAY_WORKER_TIMEOUT_SECONDS",
        default=30.0,
    )
    snapshot_max_age_seconds = _read_positive_float(
        "GATEWAY_SNAPSHOT_MAX_AGE_SECONDS",
        default=10.0,
    )

    registry = WorkerRegistry()
    collector = VLLMMetricsCollector(
        timeout_seconds=metrics_timeout_seconds,
    )
    router = LoadAwareRouter(
        registry=registry,
        max_snapshot_age_seconds=snapshot_max_age_seconds,
    )
    worker_pool = VLLMWorkerPool(
        timeout_seconds=worker_timeout_seconds,
    )
    service = GatewayService(
        registry=registry,
        router=router,
        collector=collector,
        registrations=(
            WorkerRegistration(
                worker_id=worker_id,
                base_url=base_url,
                model_ids=frozenset({model_id}),
            ),
        ),
        worker_resolver=worker_pool.resolve,
    )

    app = create_app(service=service)

    @app.on_event("shutdown")
    async def close_gateway_resources() -> None:
        await collector.aclose()
        await worker_pool.aclose()

    return app


app = build_app()