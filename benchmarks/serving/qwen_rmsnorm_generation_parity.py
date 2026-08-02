from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_full_model_rmsnorm_replacement import replace_qwen_rmsnorm_modules


DEFAULT_PROMPTS = (
    "Explain why GPU memory bandwidth matters for RMSNorm in one sentence.",
    "Explain the purpose of a KV cache in one sentence.",
    "Write a Python function that returns the square of an integer.",
    "The capital of France is",
)


def generate_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> list[int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    prompt_length = input_ids.shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    return output_ids[0, prompt_length:].tolist()


def token_agreement(
    baseline: list[int],
    triton: list[int],
) -> tuple[float, int | None]:
    total_positions = max(len(baseline), len(triton))

    if total_positions == 0:
        return 1.0, None

    matched = 0
    first_difference: int | None = None

    for index in range(total_positions):
        baseline_token = baseline[index] if index < len(baseline) else None
        triton_token = triton[index] if index < len(triton) else None

        if baseline_token == triton_token:
            matched += 1
        elif first_difference is None:
            first_difference = index

    return matched / total_positions, first_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare greedy generation before and after Qwen RMSNorm replacement."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit with a nonzero status if any prompt has non-identical output tokens.",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    return args


def main() -> None:
    args = parse_args()

    print(f"loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    print(f"loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    model.eval()

    baseline_outputs = [
        generate_tokens(model, tokenizer, prompt, args.max_new_tokens)
        for prompt in DEFAULT_PROMPTS
    ]

    replaced_count = replace_qwen_rmsnorm_modules(model)
    if replaced_count == 0:
        raise RuntimeError("no Qwen2RMSNorm modules were replaced")

    triton_outputs = [
        generate_tokens(model, tokenizer, prompt, args.max_new_tokens)
        for prompt in DEFAULT_PROMPTS
    ]

    exact_matches = 0
    total_matched_tokens = 0
    total_tokens = 0

    print(f"\nreplaced RMSNorm modules: {replaced_count}")
    print(f"max new tokens: {args.max_new_tokens}")

    for index, (prompt, baseline, triton) in enumerate(
        zip(DEFAULT_PROMPTS, baseline_outputs, triton_outputs),
        start=1,
    ):
        agreement, first_difference = token_agreement(baseline, triton)
        exact_match = baseline == triton
        exact_matches += int(exact_match)
        total_matched_tokens += round(agreement * max(len(baseline), len(triton)))
        total_tokens += max(len(baseline), len(triton))

        print(f"\n[prompt {index}] {prompt}")
        print(f"baseline token count: {len(baseline)}")
        print(f"triton token count: {len(triton)}")
        print(f"exact token match: {exact_match}")
        print(f"token agreement: {agreement:.6f}")
        print(f"first differing token index: {first_difference}")
        print(f"baseline text: {tokenizer.decode(baseline, skip_special_tokens=True)!r}")
        print(f"triton text: {tokenizer.decode(triton, skip_special_tokens=True)!r}")

    overall_agreement = (
        total_matched_tokens / total_tokens if total_tokens else 1.0
    )

    print("\n[summary]")
    print(f"exact prompt matches: {exact_matches}/{len(DEFAULT_PROMPTS)}")
    print(f"overall token agreement: {overall_agreement:.6f}")

    if args.require_exact and exact_matches != len(DEFAULT_PROMPTS):
        raise SystemExit("generation parity failed")


if __name__ == "__main__":
    main()