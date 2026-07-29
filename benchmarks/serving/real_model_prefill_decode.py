import argparse
import time
from dataclasses import dataclass
import math
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
        description="Benchmark real LLM prefill and decode on a CUDA GPU."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prompt-repeats",
        type=int,
        default=16,
        help="Repeat the base sentence to increase prompt length.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=5,
        help="Warmup rounds excluded from benchmark statistics.",
    )
    parser.add_argument(
        "--benchmark-rounds",
        type=int,
        default=20,
        help="Measured prefill rounds.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=32,
        help="Number of autoregressive decode steps to measure.",
    )
    return parser.parse_args()


def measure_cuda_ms(fn) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event)

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

def cache_size_bytes(past_key_values) -> int:
    total_bytes = 0

    for layer_idx in range(len(past_key_values)):
        keys, values = past_key_values[layer_idx]
        total_bytes += keys.numel() * keys.element_size()
        total_bytes += values.numel() * values.element_size()

    return total_bytes


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

    base_sentence = "请从推理系统工程角度解释 KV Cache 的作用。"
    prompt = "\n".join([base_sentence] * args.prompt_repeats)

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(input_ids, device=device)

    print()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"model: {args.model_id}")
    print(f"dtype: {next(model.parameters()).dtype}")
    print(f"input token count: {input_ids.shape[1]}")
    print(f"warmup rounds: {args.warmup_rounds}")
    print(f"benchmark rounds: {args.benchmark_rounds}")
    print(f"decode steps: {args.decode_steps}")

    with torch.inference_mode():
        for _ in range(args.warmup_rounds):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)

        prefill_latencies = []
        latest_outputs = None

        for _ in range(args.benchmark_rounds):

            def run_prefill() -> None:
                nonlocal latest_outputs
                latest_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

            prefill_latencies.append(measure_cuda_ms(run_prefill))

        prefill_result = summarize(prefill_latencies)
        prefill_cache = latest_outputs.past_key_values
        first_key, first_value = prefill_cache[0]
        kv_bytes = cache_size_bytes(prefill_cache)

        prefill_peak_mib = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        decode_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = decode_outputs.past_key_values
        next_token = decode_outputs.logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        decode_latencies = []
        current_attention_mask = attention_mask

        for _ in range(args.decode_steps):
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

            def run_decode() -> None:
                nonlocal decode_outputs, past_key_values, next_token
                decode_outputs = model(
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = decode_outputs.past_key_values
                next_token = decode_outputs.logits[:, -1:].argmax(dim=-1)

            decode_latencies.append(measure_cuda_ms(run_decode))

        decode_result = summarize(decode_latencies)
        decode_peak_mib = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    print("\n[prefill: steady state]")
    print(f"average latency (ms): {prefill_result.average_ms:.3f}")
    print(f"minimum latency (ms): {prefill_result.minimum_ms:.3f}")
    print(f"P50 latency (ms): {prefill_result.p50_ms:.3f}")
    print(f"P95 latency (ms): {prefill_result.p95_ms:.3f}")
    print(f"maximum latency (ms): {prefill_result.maximum_ms:.3f}")
    print(
        "throughput (tokens/s): "
        f"{input_ids.shape[1] / (prefill_result.average_ms / 1000):.2f}"
    )
    print(f"peak allocated memory (MiB): {prefill_peak_mib:.3f}")

    print("\n[decode: autoregressive steady state]")
    print(f"average latency per token (ms): {decode_result.average_ms:.3f}")
    print(f"minimum latency per token (ms): {decode_result.minimum_ms:.3f}")
    print(f"P50 latency per token (ms): {decode_result.p50_ms:.3f}")
    print(f"P95 latency per token (ms): {decode_result.p95_ms:.3f}")
    print(f"maximum latency per token (ms): {decode_result.maximum_ms:.3f}")
    print(f"decode throughput (tokens/s): {1000 / decode_result.average_ms:.2f}")
    print(f"peak allocated memory (MiB): {decode_peak_mib:.3f}")

    print("\n[kv cache after prefill]")
    print(f"cache type: {type(prefill_cache).__name__}")
    print(f"layer count: {len(prefill_cache)}")
    print(f"first layer key shape: {tuple(first_key.shape)}")
    print(f"first layer value shape: {tuple(first_value.shape)}")
    print(f"KV cache size (bytes): {kv_bytes}")
    print(f"KV cache size (MiB): {kv_bytes / 1024 / 1024:.3f}")


if __name__ == "__main__":
    main()