from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict

from serving.control_plane.load_aware_router import (
    LoadAwareRouter,
    RoutingRequest,
)
from serving.control_plane.vllm_metrics_collector import (
    VLLMMetricsCollector,
    WorkerRegistration,
)
from serving.control_plane.worker_registry import WorkerRegistry


async def run(args: argparse.Namespace) -> None:
    registry = WorkerRegistry()
    collector = VLLMMetricsCollector(
        timeout_seconds=args.timeout_seconds,
    )

    registration = WorkerRegistration(
        worker_id=args.worker_id,
        base_url=args.base_url,
        model_ids=frozenset({args.model}),
    )

    try:
        snapshot = await collector.collect_and_upsert(
            registry=registry,
            registration=registration,
        )
    finally:
        await collector.aclose()

    router = LoadAwareRouter(registry=registry)
    prefix_candidates = (
        frozenset({args.worker_id})
        if args.prefix_cache_candidate
        else frozenset()
    )

    decision = router.route(
        RoutingRequest(
            request_id="routing-control-plane-smoke",
            model_id=args.model,
            prefix_candidate_worker_ids=prefix_candidates,
        )
    )

    print("[vLLM Routing Control Plane Smoke Test]")
    print("worker snapshot:")
    for key, value in asdict(snapshot).items():
        print(f"  {key}: {value}")

    print("routing decision:")
    for key, value in asdict(decision).items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one real vLLM Worker snapshot and route one request."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--worker-id", default="vllm-local-0")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--prefix-cache-candidate",
        action="store_true",
        help="Treat this Worker as a known Prefix Cache candidate.",
    )

    args = parser.parse_args()

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))