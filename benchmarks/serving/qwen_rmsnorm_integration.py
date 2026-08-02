from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "kernels" / "triton" / "01_rmsnorm"
sys.path.insert(0, str(KERNEL_DIR))

from rmsnorm import rmsnorm_triton


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Triton RMSNorm kernel against Qwen's real first layer."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=1000)
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

    layers = model.model.layers
    rms_norm = layers[0].input_layernorm
    epsilon = getattr(rms_norm, "variance_epsilon", 1e-6)

    input_ids = (
        torch.arange(args.sequence_length, device="cuda")
        .unsqueeze(0)
        .remainder(model.config.vocab_size)
    )

    captured_inputs: list[torch.Tensor] = []

    def capture_input(_module, inputs) -> None:
        captured_inputs.append(inputs[0].detach())

    hook = rms_norm.register_forward_pre_hook(capture_input)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    hook.remove()

    hidden_states = captured_inputs[0].contiguous()
    weight = rms_norm.weight.detach().contiguous()

    with torch.inference_mode():
        qwen_output = rms_norm(hidden_states)
        triton_output = rmsnorm_triton(
            hidden_states,
            weight,
            epsilon=epsilon,
            num_warps=4,
        )
    torch.cuda.synchronize()

    max_absolute_error = (qwen_output - triton_output).abs().max().item()
    max_relative_error = (
        (qwen_output - triton_output).abs()
        / qwen_output.abs().clamp_min(1e-5)
    ).max().item()

    qwen_ms = measure_latency_ms(
        lambda: rms_norm(hidden_states),
        args.warmup_rounds,
        args.benchmark_rounds,
    )
    triton_ms = measure_latency_ms(
        lambda: rmsnorm_triton(
            hidden_states,
            weight,
            epsilon=epsilon,
            num_warps=4,
        ),
        args.warmup_rounds,
        args.benchmark_rounds,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"RMSNorm module: {type(rms_norm).__name__}")
    print(f"hidden-state shape: {tuple(hidden_states.shape)}")
    print(f"weight shape: {tuple(weight.shape)}")
    print(f"epsilon: {epsilon}")
    print()
    print(f"Qwen RMSNorm latency (ms): {qwen_ms:.4f}")
    print(f"Triton RMSNorm latency (ms): {triton_ms:.4f}")
    print(f"Triton speedup: {qwen_ms / triton_ms:.3f}x")
    print(f"Maximum absolute error: {max_absolute_error:.6f}")
    print(f"Maximum relative error: {max_relative_error:.6f}")


if __name__ == "__main__":
    main()