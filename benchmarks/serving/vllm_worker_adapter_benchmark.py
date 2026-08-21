from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    VLLMOpenAIWorker,
)


@dataclass(frozen=True)
class RequestMeasurement:
    ttft_ms: float | None
    e2e_latency_ms: float
    generated_characters: int
    finish_reason: str | None
    error: str | None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None

    return {
        "average": statistics.fmean(values),
        "minimum": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


async def prefix_cache_snapshot(
    base_url: str,
) -> dict[str, float] | None:
    pattern = re.compile(
        r"^vllm:gpu_prefix_cache_(queries|hits)_total(?:\{.*\})?\s+"
        r"(?P<value>[0-9.eE+-]+)$"
    )
    values: dict[str, float] = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{base_url.rstrip('/')}/metrics")
        response.raise_for_status()

    for line in response.text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue

        metric_name = match.group(1)
        values[metric_name] = values.get(metric_name, 0.0) + float(
            match.group("value")
        )

    if not values:
        return None

    return {
        "queries_total": values.get("queries", 0.0),
        "hits_total": values.get("hits", 0.0),
    }


async def execute_one(
    worker: VLLMOpenAIWorker,
    *,
    prompt: str,
    max_tokens: int,
) -> RequestMeasurement:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    generated_characters = 0
    finish_reason: str | None = None

    try:
        async for event in worker.stream_chat_completion(
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
        ):
            if event.delta_content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()

                generated_characters += len(event.delta_content)

            if event.finish_reason is not None:
                finish_reason = event.finish_reason

        completed_at = time.perf_counter()
        return RequestMeasurement(
            ttft_ms=(
                (first_token_at - started_at) * 1000
                if first_token_at is not None
                else None
            ),
            e2e_latency_ms=(completed_at - started_at) * 1000,
            generated_characters=generated_characters,
            finish_reason=finish_reason,
            error=None,
        )
    except Exception as exc:
        completed_at = time.perf_counter()
        return RequestMeasurement(
            ttft_ms=None,
            e2e_latency_ms=(completed_at - started_at) * 1000,
            generated_characters=0,
            finish_reason=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    worker = VLLMOpenAIWorker(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    try:
        print(f"warmup requests: {args.warmup_requests}")
        for _ in range(args.warmup_requests):
            result = await execute_one(
                worker,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            if result.error is not None:
                raise RuntimeError(f"warmup request failed: {result.error}")

        before_cache = await prefix_cache_snapshot(args.base_url)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_execute() -> RequestMeasurement:
            async with semaphore:
                return await execute_one(
                    worker,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                )

        started_at = time.perf_counter()
        measurements = await asyncio.gather(
            *(bounded_execute() for _ in range(args.requests))
        )
        elapsed_seconds = time.perf_counter() - started_at
        after_cache = await prefix_cache_snapshot(args.base_url)
    finally:
        await worker.aclose()

    successful = [item for item in measurements if item.error is None]
    failed = [item for item in measurements if item.error is not None]
    ttft_values = [
        item.ttft_ms
        for item in successful
        if item.ttft_ms is not None
    ]
    e2e_values = [item.e2e_latency_ms for item in successful]

    prefix_cache: dict[str, float] | None = None
    if before_cache is not None and after_cache is not None:
        queries_delta = (
            after_cache["queries_total"] - before_cache["queries_total"]
        )
        hits_delta = after_cache["hits_total"] - before_cache["hits_total"]
        prefix_cache = {
            "queries_delta": queries_delta,
            "hits_delta": hits_delta,
            "hit_ratio": hits_delta / queries_delta if queries_delta else 0.0,
        }

    return {
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "warmup_requests": args.warmup_requests,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "prompt": args.prompt,
        },
        "results": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "error_rate_percent": len(failed) / args.requests * 100,
            "wall_time_seconds": elapsed_seconds,
            "request_throughput_rps": len(successful) / elapsed_seconds,
            "ttft_ms": summarize(ttft_values),
            "e2e_latency_ms": summarize(e2e_values),
            "prefix_cache": prefix_cache,
        },
        "failed_requests": [
            asdict(item)
            for item in failed
        ],
        "per_request": [
            asdict(item)
            for item in measurements
        ],
    }


def print_summary(report: dict[str, object]) -> None:
    configuration = report["configuration"]
    results = report["results"]

    assert isinstance(configuration, dict)
    assert isinstance(results, dict)

    print("\n[vLLM Worker Adapter Benchmark]")
    print(
        f"model: {configuration['model']}, "
        f"requests: {configuration['requests']}, "
        f"concurrency: {configuration['concurrency']}"
    )
    print(
        f"success: {results['successful_requests']}, "
        f"failed: {results['failed_requests']}, "
        f"error rate: {results['error_rate_percent']:.2f}%"
    )
    print(
        "request throughput (RPS): "
        f"{results['request_throughput_rps']:.2f}"
    )

    for metric_name, label in (
        ("ttft_ms", "TTFT"),
        ("e2e_latency_ms", "E2E latency"),
    ):
        metric = results[metric_name]
        if metric is None:
            print(f"{label}: unavailable")
            continue

        assert isinstance(metric, dict)
        print(
            f"{label} (ms): "
            f"avg={metric['average']:.2f}, "
            f"p50={metric['p50']:.2f}, "
            f"p95={metric['p95']:.2f}, "
            f"max={metric['maximum']:.2f}"
        )

    prefix_cache = results["prefix_cache"]
    if prefix_cache is not None:
        assert isinstance(prefix_cache, dict)
        print(
            "prefix cache: "
            f"queries delta={prefix_cache['queries_delta']:.0f}, "
            f"hits delta={prefix_cache['hits_delta']:.0f}, "
            f"hit ratio={prefix_cache['hit_ratio']:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real streaming requests through VLLMOpenAIWorker."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--prompt",
        default=(
            "In one concise sentence, explain why KV cache matters "
            "for autoregressive LLM inference."
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "benchmarks/serving/results/"
            "vllm_worker_adapter_benchmark.json"
        ),
    )

    args = parser.parse_args()

    for name in (
        "warmup_requests",
        "requests",
        "concurrency",
        "max_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    return args


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nraw result written to: {output_path}")


if __name__ == "__main__":
    main()