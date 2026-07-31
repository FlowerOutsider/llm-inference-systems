from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RequestMeasurement:
    request_index: int
    success: bool
    ttft_ms: float | None
    e2e_latency_ms: float | None
    tpot_ms: float | None
    completion_tokens: int | None
    content_chunk_count: int
    error: str | None = None


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile for an empty list")

    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None

    return {
        "average": statistics.mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "maximum": max(values),
    }


def build_messages(
    *,
    shared_prefix_repeats: int,
    request_index: int,
) -> list[dict[str, str]]:
    shared_prefix = "\n".join(
        [
            "背景资料：Continuous Batching 将不同请求的 Prefill 和 Decode "
            "按 token 预算动态组合，以提高 GPU 利用率并控制延迟。"
        ]
        * shared_prefix_repeats
    )

    return [
        {
            "role": "system",
            "content": "你是一名严谨的 LLM 推理系统工程师。",
        },
        {
            "role": "user",
            "content": (
                f"{shared_prefix}\n\n"
                f"请求编号：{request_index}\n"
                "请用两句话说明 Continuous Batching 对 TTFT 和 TPOT 的影响。"
            ),
        },
    ]


async def stream_request(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    request_index: int,
    max_tokens: int,
    shared_prefix_repeats: int,
) -> RequestMeasurement:
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_messages(
            shared_prefix_repeats=shared_prefix_repeats,
            request_index=request_index,
        ),
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens: int | None = None
    content_chunk_count = 0

    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                usage = chunk.get("usage")
                if usage is not None:
                    completion_tokens = usage.get("completion_tokens")

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    content_chunk_count += 1

        finished_at = time.perf_counter()

        if first_token_at is None:
            raise RuntimeError("stream completed without a content token")

        ttft_ms = (first_token_at - started_at) * 1000
        e2e_latency_ms = (finished_at - started_at) * 1000

        # completion_tokens 来自服务端 tokenizer；它比直接计数 SSE chunk 更可靠。
        if (
            completion_tokens is not None
            and completion_tokens > 1
            and last_token_at is not None
        ):
            tpot_ms = (
                (last_token_at - first_token_at)
                / (completion_tokens - 1)
                * 1000
            )
        else:
            tpot_ms = None

        return RequestMeasurement(
            request_index=request_index,
            success=True,
            ttft_ms=ttft_ms,
            e2e_latency_ms=e2e_latency_ms,
            tpot_ms=tpot_ms,
            completion_tokens=completion_tokens,
            content_chunk_count=content_chunk_count,
        )

    except Exception as exc:
        return RequestMeasurement(
            request_index=request_index,
            success=False,
            ttft_ms=None,
            e2e_latency_ms=None,
            tpot_ms=None,
            completion_tokens=None,
            content_chunk_count=content_chunk_count,
            error=str(exc),
        )


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        print(
            f"warmup: requests={args.warmup_requests}, "
            f"shared_prefix_repeats={args.shared_prefix_repeats}"
        )

        for index in range(args.warmup_requests):
            result = await stream_request(
                client=client,
                endpoint=endpoint,
                model=args.model,
                request_index=-index - 1,
                max_tokens=args.max_tokens,
                shared_prefix_repeats=args.shared_prefix_repeats,
            )
            if not result.success:
                raise RuntimeError(f"warmup request failed: {result.error}")

        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_one(index: int) -> RequestMeasurement:
            async with semaphore:
                return await stream_request(
                    client=client,
                    endpoint=endpoint,
                    model=args.model,
                    request_index=index,
                    max_tokens=args.max_tokens,
                    shared_prefix_repeats=args.shared_prefix_repeats,
                )

        benchmark_started_at = time.perf_counter()
        results = await asyncio.gather(
            *(run_one(index) for index in range(args.requests))
        )
        benchmark_elapsed_seconds = time.perf_counter() - benchmark_started_at

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    ttft_values = [
        result.ttft_ms
        for result in successful
        if result.ttft_ms is not None
    ]
    e2e_values = [
        result.e2e_latency_ms
        for result in successful
        if result.e2e_latency_ms is not None
    ]
    tpot_values = [
        result.tpot_ms
        for result in successful
        if result.tpot_ms is not None
    ]

    generated_tokens = sum(
        result.completion_tokens or 0
        for result in successful
    )

    return {
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "warmup_requests": args.warmup_requests,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "shared_prefix_repeats": args.shared_prefix_repeats,
        },
        "results": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "error_rate_percent": len(failed) / args.requests * 100,
            "wall_time_seconds": benchmark_elapsed_seconds,
            "request_throughput_rps": len(successful)
            / benchmark_elapsed_seconds,
            "generation_throughput_tokens_per_second": generated_tokens
            / benchmark_elapsed_seconds,
            "generated_tokens": generated_tokens,
            "ttft_ms": summarize(ttft_values),
            "e2e_latency_ms": summarize(e2e_values),
            "tpot_ms": summarize(tpot_values),
        },
        "failed_requests": [asdict(result) for result in failed],
        "per_request": [asdict(result) for result in results],
    }


def print_summary(report: dict[str, Any]) -> None:
    config = report["configuration"]
    results = report["results"]

    print("\n[vLLM streaming benchmark]")
    print(f"model: {config['model']}")
    print(
        "load: "
        f"requests={config['requests']}, "
        f"concurrency={config['concurrency']}, "
        f"max_tokens={config['max_tokens']}"
    )
    print(
        "outcome: "
        f"success={results['successful_requests']}, "
        f"failed={results['failed_requests']}, "
        f"error_rate={results['error_rate_percent']:.2f}%"
    )
    print(f"request throughput (RPS): {results['request_throughput_rps']:.2f}")
    print(
        "generation throughput (tokens/s): "
        f"{results['generation_throughput_tokens_per_second']:.2f}"
    )

    for metric_name, label in (
        ("ttft_ms", "TTFT"),
        ("tpot_ms", "TPOT"),
        ("e2e_latency_ms", "E2E latency"),
    ):
        metric = results[metric_name]
        if metric is None:
            print(f"{label}: unavailable")
            continue

        print(
            f"{label} (ms): "
            f"avg={metric['average']:.2f}, "
            f"p50={metric['p50']:.2f}, "
            f"p95={metric['p95']:.2f}, "
            f"max={metric['maximum']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TTFT, TPOT, and throughput against a vLLM server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--shared-prefix-repeats", type=int, default=64)
    parser.add_argument(
        "--output",
        default="benchmarks/serving/results/vllm_streaming_benchmark.json",
    )

    args = parser.parse_args()

    for name in (
        "warmup_requests",
        "requests",
        "concurrency",
        "max_tokens",
        "shared_prefix_repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

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