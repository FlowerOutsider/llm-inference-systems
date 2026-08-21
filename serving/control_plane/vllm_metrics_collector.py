from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Callable

import httpx

from serving.control_plane.worker_registry import (
    WorkerRegistry,
    WorkerSnapshot,
)


class VLLMMetricsError(RuntimeError):
    """vLLM 指标采集失败的基类。"""


class VLLMMetricsProtocolError(VLLMMetricsError):
    """vLLM 指标缺失或不符合 Prometheus 文本格式。"""


@dataclass(frozen=True)
class WorkerRegistration:
    """静态 Worker 配置，由服务发现或部署配置提供。"""

    worker_id: str
    base_url: str
    model_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model_ids:
            raise ValueError("model_ids must not be empty")


class VLLMMetricsCollector:
    """
    将 vLLM 的健康检查和 Prometheus 指标转换为 WorkerSnapshot。

    采集失败不会向上抛出，而是写入 healthy=false 快照，以便 Router
    能立即停止向问题 Worker 路由请求。
    """

    _METRIC_PATTERN = re.compile(
        r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
        r"(?:\{[^}]*\})?"
        r"\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?)"
        r"(?:\s+\d+)?$"
    )

    _WAITING_METRIC = "vllm:num_requests_waiting"
    _RUNNING_METRIC = "vllm:num_requests_running"
    _GPU_CACHE_USAGE_METRIC = "vllm:gpu_cache_usage_perc"

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._clock = clock

    async def aclose(self) -> None:
        await self._client.aclose()

    async def collect(
        self,
        registration: WorkerRegistration,
    ) -> WorkerSnapshot:
        observed_at_monotonic = self._clock()
        base_url = registration.base_url.rstrip("/")

        try:
            health_response = await self._client.get(
                f"{base_url}/health"
            )
            if not health_response.is_success:
                return self._unhealthy_snapshot(
                    registration,
                    observed_at_monotonic=observed_at_monotonic,
                )

            metrics_response = await self._client.get(
                f"{base_url}/metrics"
            )
            if not metrics_response.is_success:
                return self._unhealthy_snapshot(
                    registration,
                    observed_at_monotonic=observed_at_monotonic,
                )

            (
                waiting_requests,
                running_requests,
                gpu_cache_usage_perc,
            ) = self._parse_metrics(metrics_response.text)
        except (httpx.HTTPError, VLLMMetricsProtocolError):
            return self._unhealthy_snapshot(
                registration,
                observed_at_monotonic=observed_at_monotonic,
            )

        return WorkerSnapshot(
            worker_id=registration.worker_id,
            base_url=base_url,
            model_ids=registration.model_ids,
            healthy=True,
            waiting_requests=waiting_requests,
            running_requests=running_requests,
            gpu_cache_usage_perc=gpu_cache_usage_perc,
            observed_at_monotonic=observed_at_monotonic,
        )

    async def collect_and_upsert(
        self,
        *,
        registry: WorkerRegistry,
        registration: WorkerRegistration,
    ) -> WorkerSnapshot:
        snapshot = await self.collect(registration)
        registry.upsert(snapshot)
        return snapshot

    def _parse_metrics(
        self,
        metrics_text: str,
    ) -> tuple[int, int, float]:
        samples: dict[str, list[float]] = {
            self._WAITING_METRIC: [],
            self._RUNNING_METRIC: [],
            self._GPU_CACHE_USAGE_METRIC: [],
        }

        for line in metrics_text.splitlines():
            match = self._METRIC_PATTERN.match(line.strip())
            if match is None:
                continue

            metric_name = match.group("name")
            if metric_name not in samples:
                continue

            value = float(match.group("value"))
            if not math.isfinite(value):
                raise VLLMMetricsProtocolError(
                    f"metric {metric_name!r} is not finite"
                )

            samples[metric_name].append(value)

        missing_metrics = [
            metric_name
            for metric_name, values in samples.items()
            if not values
        ]
        if missing_metrics:
            raise VLLMMetricsProtocolError(
                f"missing required vLLM metrics: {missing_metrics}"
            )

        waiting_requests = self._sum_request_count(
            self._WAITING_METRIC,
            samples[self._WAITING_METRIC],
        )
        running_requests = self._sum_request_count(
            self._RUNNING_METRIC,
            samples[self._RUNNING_METRIC],
        )
        gpu_cache_usage_perc = max(
            samples[self._GPU_CACHE_USAGE_METRIC]
        )

        if not 0.0 <= gpu_cache_usage_perc <= 1.0:
            raise VLLMMetricsProtocolError(
                "vLLM GPU cache usage must be between 0.0 and 1.0"
            )

        return (
            waiting_requests,
            running_requests,
            gpu_cache_usage_perc,
        )

    @staticmethod
    def _sum_request_count(
        metric_name: str,
        values: list[float],
    ) -> int:
        if any(value < 0 or not value.is_integer() for value in values):
            raise VLLMMetricsProtocolError(
                f"metric {metric_name!r} must contain non-negative integers"
            )

        return int(sum(values))

    @staticmethod
    def _unhealthy_snapshot(
        registration: WorkerRegistration,
        *,
        observed_at_monotonic: float,
    ) -> WorkerSnapshot:
        return WorkerSnapshot(
            worker_id=registration.worker_id,
            base_url=registration.base_url.rstrip("/"),
            model_ids=registration.model_ids,
            healthy=False,
            waiting_requests=0,
            running_requests=0,
            gpu_cache_usage_perc=0.0,
            observed_at_monotonic=observed_at_monotonic,
        )