#!/usr/bin/env python3
"""Summarize repeated vLLM concurrency experiments into Markdown."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FILE_PATTERN = re.compile(
    r"vllm_concurrency_(?P<concurrency>\d+)_batch(?P<batch_tokens>\d+)_run(?P<run>\d+)\.json$"
)
COMPARABLE_CONFIG_KEYS = (
    "base_url",
    "model",
    "warmup_requests",
    "requests",
    "max_tokens",
    "shared_prefix_repeats",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated vLLM concurrency benchmark JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/serving/results",
        help="Directory containing benchmark result JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="vllm_concurrency_*_batch*_run*.json",
        help="Glob pattern relative to --results-dir.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="Expected number of repetitions for each concurrency level.",
    )
    parser.add_argument(
        "--output",
        help="Optional Markdown output path. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def format_summary(values: list[float], suffix: str = "") -> str:
    return (
        f"{statistics.median(values):.2f}{suffix} "
        f"[{min(values):.2f}, {max(values):.2f}]"
    )


def load_experiments(
    results_dir: Path,
    pattern: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for file_name in sorted(glob.glob(str(results_dir / pattern))):
        path = Path(file_name)
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"result filename does not match the expected format: {path.name}"
            )

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if "configuration" not in payload or "results" not in payload:
            raise ValueError(f"missing configuration or results in {path}")

        groups[
            (
                int(match.group("concurrency")),
                int(match.group("batch_tokens")),
            )
        ].append(payload)

    if not groups:
        raise ValueError(f"no result files matched {results_dir / pattern}")

    return groups


def validate_groups(
    groups: dict[tuple[int, int], list[dict[str, Any]]],
    expected_runs: int,
) -> list[str]:
    warnings: list[str] = []
    baseline_config: dict[str, Any] | None = None

    for (concurrency, batch_tokens), payloads in sorted(groups.items()):
        if len(payloads) != expected_runs:
            warnings.append(
                f"warning: concurrency={concurrency}, batch_tokens={batch_tokens} "
                f"has {len(payloads)} runs, expected {expected_runs}."
            )

        reference = payloads[0]["configuration"]

        for payload in payloads[1:]:
            for key in COMPARABLE_CONFIG_KEYS:
                if payload["configuration"].get(key) != reference.get(key):
                    raise ValueError(
                        f"configuration mismatch within concurrency={concurrency}, "
                        f"batch_tokens={batch_tokens}: key={key}"
                    )

        if baseline_config is None:
            baseline_config = reference
            continue

        for key in COMPARABLE_CONFIG_KEYS:
            if reference.get(key) != baseline_config.get(key):
                raise ValueError(
                    "configuration mismatch across concurrency groups: "
                    f"key={key}, concurrency={concurrency}, "
                    f"batch_tokens={batch_tokens}"
                )

    return warnings


def render_markdown(
    groups: dict[tuple[int, int], list[dict[str, Any]]],
    warnings: list[str],
) -> str:
    lines = [
        "# vLLM 并发阶梯实验汇总",
        "",
        "统计口径：每个单元格为 `中位数 [最小值, 最大值]`。",
        "",
        "| 并发数 | Batch Token Budget | Runs | RPS | 生成吞吐 | TTFT P95 | TPOT P95 | E2E P95 | 错误率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for (concurrency, batch_tokens), payloads in sorted(groups.items()):
        results = [payload["results"] for payload in payloads]

        lines.append(
            "| "
            f"{concurrency} | "
            f"{batch_tokens} | "
            f"{len(results)} | "
            f"{format_summary([row['request_throughput_rps'] for row in results])} | "
            f"{format_summary([row['generation_throughput_tokens_per_second'] for row in results], ' tokens/s')} | "
            f"{format_summary([row['ttft_ms']['p95'] for row in results], ' ms')} | "
            f"{format_summary([row['tpot_ms']['p95'] for row in results], ' ms')} | "
            f"{format_summary([row['e2e_latency_ms']['p95'] for row in results], ' ms')} | "
            f"{format_summary([row['error_rate_percent'] for row in results], '%')} |"
        )

    first_payload = next(iter(groups.values()))[0]
    config = first_payload["configuration"]

    lines.extend(
        [
            "",
            "## 实验负载",
            "",
            f"- base_url={config['base_url']}",
            f"- model={config['model']}",
            f"- warmup_requests={config['warmup_requests']}",
            f"- requests={config['requests']}",
            f"- max_tokens={config['max_tokens']}",
            f"- shared_prefix_repeats={config['shared_prefix_repeats']}",
        ]
    )

    if warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    try:
        groups = load_experiments(Path(args.results_dir), args.pattern)
        warnings = validate_groups(groups, args.expected_runs)
        markdown = render_markdown(groups, warnings)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"summary written to: {output_path}")
    else:
        print(markdown, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())