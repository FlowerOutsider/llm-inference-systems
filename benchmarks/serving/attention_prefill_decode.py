import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pathlib import Path

from torch.profiler import ProfilerActivity, profile


@dataclass
class BenchmarkResult:
    name: str
    latency_ms: float
    tokens_per_second: float
    peak_memory_mib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch SDPA prefill and decode attention."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
    "--profile",
    action="store_true",
    help="Export PyTorch profiler traces.",
    )
    return parser.parse_args()


def benchmark(
    name: str,
    operation,
    processed_tokens: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        operation()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        operation()
    end.record()

    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iterations
    tokens_per_second = processed_tokens / (latency_ms / 1000)
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return BenchmarkResult(
        name=name,
        latency_ms=latency_ms,
        tokens_per_second=tokens_per_second,
        peak_memory_mib=peak_memory_mib,
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"\n[{result.name}]")
    print(f"Average latency: {result.latency_ms:.3f} ms")
    print(f"Throughput: {result.tokens_per_second:.2f} tokens/s")
    print(f"Peak allocated memory: {result.peak_memory_mib:.2f} MiB")

def profile_operation(
    name: str,
    operation,
    iterations: int,
    trace_directory: Path,
) -> None:
    trace_directory.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(iterations):
            operation()

        torch.cuda.synchronize()

    trace_path = trace_directory / f"{name}_attention_trace.json"
    profiler.export_chrome_trace(str(trace_path))

    print(f"\n[{name} profiler summary]")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )
    print(f"Chrome trace: {trace_path}")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this benchmark on a CUDA-enabled PyTorch.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    dtype = torch.float16

    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    prompt_length = args.prompt_length

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"dtype: {dtype}")
    print(
        "Shape: "
        f"batch={batch_size}, heads={num_heads}, "
        f"head_dim={head_dim}, prompt_length={prompt_length}"
    )

    # Prefill: Q、K、V 都覆盖完整 prompt，计算量随序列长度近似二次增长。
    prefill_query = torch.randn(
        batch_size, num_heads, prompt_length, head_dim, device=device, dtype=dtype
    )
    prefill_key = torch.randn_like(prefill_query)
    prefill_value = torch.randn_like(prefill_query)

    def prefill_operation() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            prefill_query,
            prefill_key,
            prefill_value,
            is_causal=True,
        )

    # Decode: 新 token 的 Q 长度为 1；但它仍需读取整个历史 KV Cache。
    decode_query = torch.randn(
        batch_size, num_heads, 1, head_dim, device=device, dtype=dtype
    )
    decode_key_cache = torch.randn(
        batch_size,
        num_heads,
        prompt_length + 1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    decode_value_cache = torch.randn_like(decode_key_cache)

    def decode_operation() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            decode_query,
            decode_key_cache,
            decode_value_cache,
            is_causal=False,
        )

    prefill_result = benchmark(
        name="prefill",
        operation=prefill_operation,
        processed_tokens=batch_size * prompt_length,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    decode_result = benchmark(
        name="decode",
        operation=decode_operation,
        processed_tokens=batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print_result(prefill_result)
    print_result(decode_result)

    trace_directory = Path("benchmarks/serving/results")
    if args.profile:
        trace_directory = Path("benchmarks/serving/results")
        profile_operation(
            name="prefill",
            operation=prefill_operation,
            iterations=20,
            trace_directory=trace_directory,
        )
        profile_operation(
            name="decode",
            operation=decode_operation,
            iterations=20,
            trace_directory=trace_directory,
        )

if __name__ == "__main__":
    main()