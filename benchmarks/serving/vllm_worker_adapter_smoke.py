from __future__ import annotations

import argparse
import asyncio
import time

from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    VLLMOpenAIWorker,
)


async def run(args: argparse.Namespace) -> None:
    worker = VLLMOpenAIWorker(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    response_parts: list[str] = []
    first_token_at: float | None = None
    finish_reason: str | None = None
    started_at = time.perf_counter()

    try:
        async for event in worker.stream_chat_completion(
            messages=[
                ChatMessage(
                    role="user",
                    content=args.prompt,
                )
            ],
            max_tokens=args.max_tokens,
        ):
            if event.delta_content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()

                response_parts.append(event.delta_content)
                print(event.delta_content, end="", flush=True)

            if event.finish_reason is not None:
                finish_reason = event.finish_reason
    finally:
        await worker.aclose()

    completed_at = time.perf_counter()

    if first_token_at is None:
        raise RuntimeError("vLLM completed the request without generating text")

    print("\n")
    print("[vLLM Worker Adapter Smoke Test]")
    print(f"model: {args.model}")
    print(f"finish reason: {finish_reason}")
    print(
        "TTFT (ms): "
        f"{(first_token_at - started_at) * 1000:.2f}"
    )
    print(
        "E2E latency (ms): "
        f"{(completed_at - started_at) * 1000:.2f}"
    )
    print(f"generated text length: {len(''.join(response_parts))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real streaming request through VLLMOpenAIWorker."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--prompt",
        default=(
            "In one concise sentence, explain why KV cache matters "
            "for autoregressive LLM inference."
        ),
    )

    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))