import argparse
import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TimingResult:
    average_ms: float
    minimum_ms: float
    p50_ms: float
    p95_ms: float
    maximum_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real-model prefix KV-cache compute reuse."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prefix-repeats",
        type=int,
        default=64,
        help="Repeated shared-prefix sentences.",
    )
    parser.add_argument(
        "--suffix-repeats",
        type=int,
        default=8,
        help="Repeated request-specific suffix sentences.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=10,
        help="Warmup rounds excluded from timing statistics.",
    )
    parser.add_argument(
        "--benchmark-rounds",
        type=int,
        default=50,
        help="Measured rounds for each path.",
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize(values: list[float]) -> TimingResult:
    return TimingResult(
        average_ms=sum(values) / len(values),
        minimum_ms=min(values),
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        maximum_ms=max(values),
    )


def measure_cuda_ms(fn: Callable[[], object]) -> tuple[float, object]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 确保前序 CUDA 操作（例如 DynamicCache 的 deepcopy）不混入本次计时。
    torch.cuda.synchronize()
    start_event.record()

    result = fn()

    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event), result


def cache_size_bytes(past_key_values) -> int:
    total_bytes = 0

    for layer_idx in range(len(past_key_values)):
        keys, values = past_key_values[layer_idx]
        total_bytes += keys.numel() * keys.element_size()
        total_bytes += values.numel() * values.element_size()

    return total_bytes


def print_timing(name: str, result: TimingResult) -> None:
    print(f"\n[{name}]")
    print(f"average latency (ms): {result.average_ms:.3f}")
    print(f"minimum latency (ms): {result.minimum_ms:.3f}")
    print(f"P50 latency (ms): {result.p50_ms:.3f}")
    print(f"P95 latency (ms): {result.p95_ms:.3f}")
    print(f"maximum latency (ms): {result.maximum_ms:.3f}")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark requires a CUDA GPU.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
    ).to(device)
    model.eval()

    prefix_sentence = (
        "这是共享的系统上下文：请以严谨、准确、工程化的方式回答问题。"
        "回答应说明关键假设、性能影响和资源约束。\n"
    )
    suffix_sentence = (
        "这是当前请求的业务内容：请解释 KV Cache 对在线推理吞吐的影响。\n"
    )

    prefix_text = prefix_sentence * args.prefix_repeats
    suffix_text = suffix_sentence * args.suffix_repeats

    prefix_input_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    suffix_input_ids = tokenizer(
        suffix_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    full_input_ids = torch.cat([prefix_input_ids, suffix_input_ids], dim=1)

    prefix_attention_mask = torch.ones_like(prefix_input_ids, device=device)
    full_attention_mask = torch.ones_like(full_input_ids, device=device)

    prefix_length = prefix_input_ids.shape[1]
    full_length = full_input_ids.shape[1]

    # Prefix 在序列中的绝对位置：[0, 1, ..., prefix_length - 1]
    prefix_cache_position = torch.arange(prefix_length, device=device)
    prefix_position_ids = prefix_cache_position.unsqueeze(0)

    # 完整请求的绝对位置：[0, 1, ..., full_length - 1]
    full_cache_position = torch.arange(full_length, device=device)
    full_position_ids = full_cache_position.unsqueeze(0)

    # 复用前缀 Cache 时，suffix 必须从 prefix_length 开始编号。
    # 这是正确处理 RoPE 位置编码和 KV Cache continuation 的关键。
    suffix_cache_position = torch.arange(
        prefix_length,
        full_length,
        device=device,
    )
    suffix_position_ids = suffix_cache_position.unsqueeze(0)

    def run_prefix():
        return model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            cache_position=prefix_cache_position,
            use_cache=True,
            return_dict=True,
        )

    def run_full_request():
        return model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            cache_position=full_cache_position,
            use_cache=True,
            return_dict=True,
        )

    def run_cached_suffix(reuse_cache):
        return model(
            input_ids=suffix_input_ids,
            attention_mask=full_attention_mask,
            position_ids=suffix_position_ids,
            cache_position=suffix_cache_position,
            past_key_values=reuse_cache,
            use_cache=True,
            return_dict=True,
        )

    print()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"model: {args.model_id}")
    print(f"dtype: {next(model.parameters()).dtype}")
    print(f"shared prefix token count: {prefix_length}")
    print(f"request suffix token count: {suffix_input_ids.shape[1]}")
    print(f"full request token count: {full_length}")
    print(f"warmup rounds: {args.warmup_rounds}")
    print(f"benchmark rounds: {args.benchmark_rounds}")

    with torch.inference_mode():
        # 前缀路径先热身，避免将 CUDA 初始化、内核选择等冷启动成本
        # 误记为“构建 Prefix Cache 的真实成本”。
        for _ in range(args.warmup_rounds):
            run_prefix()

        torch.cuda.synchronize()

        prefix_latency_ms, prefix_outputs = measure_cuda_ms(run_prefix)
        prefix_cache = prefix_outputs.past_key_values
        prefix_cache_bytes = cache_size_bytes(prefix_cache)

        # 正确性基准：完整请求一次性 Prefill。
        _, baseline_reference = measure_cuda_ms(run_full_request)

        # 正确性对照：复用前缀 Cache，只计算 suffix。
        # deepcopy 保护 prefix_cache，避免 suffix 写入污染共享前缀。
        reuse_cache_for_check = copy.deepcopy(prefix_cache)
        _, reused_reference = measure_cuda_ms(
            lambda: run_cached_suffix(reuse_cache_for_check)
        )

        baseline_suffix_logits = baseline_reference.logits[
            :, -suffix_input_ids.shape[1] :, :
        ]
        reused_suffix_logits = reused_reference.logits

        absolute_difference = (
            baseline_suffix_logits - reused_suffix_logits
        ).abs()

        logits_match = torch.allclose(
            baseline_suffix_logits,
            reused_suffix_logits,
            rtol=1e-2,
            atol=1e-2,
        )
        max_logit_difference = absolute_difference.max().item()
        mean_logit_difference = absolute_difference.mean().item()

        baseline_top1 = baseline_suffix_logits.argmax(dim=-1)
        reused_top1 = reused_suffix_logits.argmax(dim=-1)
        top1_agreement = (
            baseline_top1 == reused_top1
        ).float().mean().item()

        # 热身完整请求路径。
        for _ in range(args.warmup_rounds):
            run_full_request()

        # 热身 Prefix Cache 复用路径。
        for _ in range(args.warmup_rounds):
            reuse_cache = copy.deepcopy(prefix_cache)
            run_cached_suffix(reuse_cache)

        torch.cuda.synchronize()

        baseline_latencies = []
        for _ in range(args.benchmark_rounds):
            latency_ms, _ = measure_cuda_ms(run_full_request)
            baseline_latencies.append(latency_ms)

        reused_latencies = []
        for _ in range(args.benchmark_rounds):
            # 注意：复制不计入当前计时区间。
            # 本实验专门测量“少计算共享前缀”的 GPU 计算收益。
            # 生产系统不会深拷贝完整 Tensor，而会以 block table、
            # 引用计数和 Copy-on-Write 复用物理 KV block。
            reuse_cache = copy.deepcopy(prefix_cache)

            latency_ms, _ = measure_cuda_ms(
                lambda: run_cached_suffix(reuse_cache)
            )
            reused_latencies.append(latency_ms)

    baseline_result = summarize(baseline_latencies)
    reused_result = summarize(reused_latencies)

    saved_ms = baseline_result.average_ms - reused_result.average_ms
    saved_percent = saved_ms / baseline_result.average_ms * 100

    print("\n[correctness]")
    print(f"logits allclose: {logits_match}")
    print(f"maximum suffix logit difference: {max_logit_difference:.8f}")
    print(f"mean suffix logit difference: {mean_logit_difference:.8f}")
    print(f"top-1 token agreement: {top1_agreement:.4f}")

    print("\n[shared prefix cache]")
    print(f"prefix prefill latency (ms): {prefix_latency_ms:.3f}")
    print(f"cache type: {type(prefix_cache).__name__}")
    print(f"layer count: {len(prefix_cache)}")
    print(f"shared prefix KV cache (MiB): {prefix_cache_bytes / 1024 / 1024:.3f}")

    print_timing("baseline: full request prefill", baseline_result)
    print_timing("reuse: cached prefix plus suffix only", reused_result)

    print("\n[compute reuse result]")
    print(f"average compute saved per request (ms): {saved_ms:.3f}")
    print(f"average compute reduction (%): {saved_percent:.2f}")

    print("\n[scope boundary]")
    print(
        "This benchmark measures real-model prefix compute reuse. "
        "DynamicCache is deep-copied outside the timed region to protect "
        "the shared prefix from mutation. Therefore the result is not a "
        "production memory-sharing benchmark. Production prefix caching "
        "requires paged KV blocks, reference counting, and copy-on-write."
    )


if __name__ == "__main__":
    main()