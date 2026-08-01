#!/usr/bin/env python3
"""Summarize repeated vLLM streaming benchmark JSON results."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUDGET_PATTERN = re.compile(r"vllm_batched_tokens_(\d+)_c8_long_run\d+\.json$")

METRICS = (
    ("request_throughput_rps", "RPS", ""),
    ("generation_throughput_tokens_per_second", "生成吞吐", "tokens/s"),
    ("ttft_ms.p95", "TTFT P95", "ms"),
    ("tpot_ms.p95", "TPOT P95", "ms"),
    ("e2e_latency_ms.p95", "E2E P95", "ms"),
    ("error_rate_percent", "错误率", "%"),
)


@dataclass(frozen=True)
class BenchmarkRecord:
    path: Path
    budget: int
    configuration: dict[str, Any]
    results: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总重复执行的 vLLM 流式压测结果。"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/serving/results"),
        help="JSON 结果所在目录。",
    )
    parser.add_argument(
        "--pattern",
        default="vllm_batched_tokens_*_c8_long_run*.json",
        help="结果文件匹配模式。",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="每个 token budget 预期的重复次数。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选。将 Markdown 汇总写入指定文件。",
    )
    return parser.parse_args()


def nested_value(payload: dict[str, Any], path: str) -> float:
    current: Any = payload
    for key in path.split("."):
        current = current[key]
    return float(current)


def read_record(path: Path) -> BenchmarkRecord:
    match = BUDGET_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"文件名不符合实验命名协议: {path.name}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    configuration = payload["configuration"]
    results = payload["results"]

    return BenchmarkRecord(
        path=path,
        budget=int(match.group(1)),
        configuration=configuration,
        results=results,
    )


def validate_group(records: list[BenchmarkRecord]) -> None:
    fields = (
        "base_url",
        "model",
        "requests",
        "concurrency",
        "max_tokens",
        "shared_prefix_repeats",
    )

    for field in fields:
        values = {record.configuration.get(field) for record in records}
        if len(values) != 1:
            raise ValueError(
                f"token budget={records[0].budget} 的配置不一致: "
                f"{field}={sorted(values, key=str)}"
            )


def format_summary(values: list[float], unit: str) -> str:
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)

    if unit == "%":
        return f"{median:.2f}% [{minimum:.2f}, {maximum:.2f}]"
    if unit:
        return f"{median:.2f} {unit} [{minimum:.2f}, {maximum:.2f}]"
    return f"{median:.2f} [{minimum:.2f}, {maximum:.2f}]"


def render_markdown(groups: dict[int, list[BenchmarkRecord]]) -> str:
    lines = [
        "# vLLM 批处理 Token 预算实验汇总",
        "",
        "统计口径：每个单元格为 `中位数 [最小值, 最大值]`。",
        "",
        "| Token Budget | Runs | RPS | 生成吞吐 | TTFT P95 | TPOT P95 | E2E P95 | 错误率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for budget in sorted(groups):
        records = groups[budget]
        cells = [str(budget), str(len(records))]

        for metric_path, _, unit in METRICS:
            values = [
                nested_value(record.results, metric_path)
                for record in records
            ]
            cells.append(format_summary(values, unit))

        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 实验负载",
            "",
        ]
    )

    for budget in sorted(groups):
        configuration = groups[budget][0].configuration
        lines.append(
            f"- `budget={budget}`: "
            f"base_url={configuration['base_url']}, "
            f"model={configuration['model']}, "
            f"requests={configuration['requests']}, "
            f"concurrency={configuration['concurrency']}, "
            f"max_tokens={configuration['max_tokens']}, "
            f"shared_prefix_repeats={configuration['shared_prefix_repeats']}."
        )

    lines.extend(
        [
            "",
            "## 结果文件",
            "",
        ]
    )

    for budget in sorted(groups):
        for record in sorted(groups[budget], key=lambda item: item.path.name):
            lines.append(f"- `{record.path.as_posix()}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = sorted(args.results_dir.glob(args.pattern))

    if not paths:
        print(
            f"没有匹配结果文件: {args.results_dir / args.pattern}",
            file=sys.stderr,
        )
        return 1

    groups: dict[int, list[BenchmarkRecord]] = defaultdict(list)
    for path in paths:
        record = read_record(path)
        groups[record.budget].append(record)

    for budget, records in sorted(groups.items()):
        validate_group(records)
        if len(records) != args.expected_runs:
            print(
                f"warning: budget={budget} 发现 {len(records)} 轮，"
                f"预期 {args.expected_runs} 轮。",
                file=sys.stderr,
            )

    markdown = render_markdown(groups)
    print(markdown, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Markdown report written to: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())