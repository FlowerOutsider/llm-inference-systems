from __future__ import annotations

import argparse

import torch

from rmsnorm import rmsnorm_reference, rmsnorm_triton


def measure_latency_ms(
    fn,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    warmup_rounds: int,
    benchmark_rounds: int,
) -> float:
    for _ in range(warmup_rounds):
        fn(x, weight, epsilon)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(benchmark_rounds):
        fn(x, weight, epsilon)
    end.record()

    end.synchronize()
    return start.elapsed_time(end) / benchmark_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a fused Triton RMSNorm kernel against PyTorch eager."
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup-rounds", type=int, default=50)
    parser.add_argument("--benchmark-rounds", type=int, default=200)
    parser.add_argument(
        "--num-warps",
        type=int,
        choices=(1, 2, 4, 8),
        default=4,
    )
    args = parser.parse_args()

    for name in ("rows", "hidden_size", "warmup_rounds", "benchmark_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")

    return args





def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(7)
    x = torch.randn(
        (args.rows, args.hidden_size),
        device="cuda",
        dtype=dtype,
    )
    weight = torch.randn(
        (args.hidden_size,),
        device="cuda",
        dtype=dtype,
    )

    reference = rmsnorm_reference(x, weight, args.epsilon)
    triton_output = rmsnorm_triton(
        x,
        weight,
        args.epsilon,
        num_warps=args.num_warps,
    )
    torch.cuda.synchronize()

    max_absolute_error = (reference - triton_output).abs().max().item()
    max_relative_error = (
        (reference - triton_output).abs()
        / reference.abs().clamp_min(1e-5)
    ).max().item()

    reference_ms = measure_latency_ms(
        rmsnorm_reference,
        x,
        weight,
        args.epsilon,
        args.warmup_rounds,
        args.benchmark_rounds,
    )

    def run_triton(
        input_tensor: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        return rmsnorm_triton(
            input_tensor,
            scale,
            epsilon,
            num_warps=args.num_warps,
        )



    triton_ms = measure_latency_ms(
        run_triton,
        x,
        weight,
        args.epsilon,
        args.warmup_rounds,
        args.benchmark_rounds,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}")
    print(f"shape: ({args.rows}, {args.hidden_size})")
    print(f"num_warps: {args.num_warps}")
    print(f"epsilon: {args.epsilon}")
    print()
    print(f"PyTorch eager latency (ms): {reference_ms:.4f}")
    print(f"Triton fused latency (ms): {triton_ms:.4f}")
    print(f"Triton speedup: {reference_ms / triton_ms:.3f}x")
    print(f"Maximum absolute error: {max_absolute_error:.6f}")
    print(f"Maximum relative error: {max_relative_error:.6f}")


if __name__ == "__main__":
    main()