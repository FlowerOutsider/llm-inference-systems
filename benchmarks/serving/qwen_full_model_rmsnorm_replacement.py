from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "kernels" / "triton" / "01_rmsnorm"
sys.path.insert(0, str(KERNEL_DIR))

from qwen_adapter import TritonRMSNorm


def measure_latency_ms(fn, warmup_rounds: int, benchmark_rounds: int) -> float:
    for _ in range(warmup_rounds):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(benchmark_rounds):
        fn()
    end.record()

    end.synchronize()
    return start.elapsed_time(end) / benchmark_rounds


def replace_qwen_rmsnorm_modules(module: torch.nn.Module) -> int:
    replaced = 0

    for name, child in list(module.named_children()):
        if type(child).__name__ == "Qwen2RMSNorm":
            epsilon = getattr(child, "variance_epsilon", 1e-6)
            setattr(
                module,
                name,
                TritonRMSNorm(
                    weight=child.weight,
                    epsilon=epsilon,
                    num_warps=4,
                ),
            )
            replaced += 1
        else:
            replaced += replace_qwen_rmsnorm_modules(child)

    return replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure model-level Qwen prefill after replacing RMSNorm with Triton."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--benchmark-rounds", type=int, default=50)
    args = parser.parse_args()

    for name in ("sequence_length", "warmup_rounds", "benchmark_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    return args


def main() -> None:
    args = parse_args()

    print(f"loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    model.eval()

    input_ids = (
        torch.arange(args.sequence_length, device="cuda")
        .unsqueeze(0)
        .remainder(model.config.vocab_size)
    )

    def run_model() -> torch.Tensor:
        with torch.inference_mode():
            return model(input_ids=input_ids, use_cache=False).logits

    baseline_logits = run_model()
    baseline_ms = measure_latency_ms(
        run_model,
        args.warmup_rounds,
        args.benchmark_rounds,
    )

    replaced_count = replace_qwen_rmsnorm_modules(model)
    if replaced_count == 0:
        raise RuntimeError("no Qwen2RMSNorm modules were replaced")

    triton_logits = run_model()
    triton_ms = measure_latency_ms(
        run_model,
        args.warmup_rounds,
        args.benchmark_rounds,
    )

    difference = (baseline_logits - triton_logits).abs()
    top1_agreement = (
        baseline_logits.argmax(dim=-1) == triton_logits.argmax(dim=-1)
    ).float().mean().item()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"input shape: {tuple(input_ids.shape)}")
    print(f"replaced RMSNorm modules: {replaced_count}")
    print()
    print(f"Baseline Qwen prefill latency (ms): {baseline_ms:.4f}")
    print(f"Triton RMSNorm Qwen prefill latency (ms): {triton_ms:.4f}")
    print(f"End-to-end speedup: {baseline_ms / triton_ms:.3f}x")
    print(f"Maximum logit difference: {difference.max().item():.6f}")
    print(f"Mean logit difference: {difference.mean().item():.6f}")
    print(f"Top-1 token agreement: {top1_agreement:.6f}")


if __name__ == "__main__":
    main()