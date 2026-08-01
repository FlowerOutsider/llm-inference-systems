from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_vllm_benchmarks.py"


def make_payload(rps: float, concurrency: int = 8) -> dict:
    return {
        "configuration": {
            "base_url": "http://127.0.0.1:8002",
            "model": "qwen2.5-0.5b",
            "requests": 256,
            "concurrency": concurrency,
            "max_tokens": 64,
            "shared_prefix_repeats": 32,
        },
        "results": {
            "request_throughput_rps": rps,
            "generation_throughput_tokens_per_second": rps * 30.0,
            "error_rate_percent": 0.0,
            "ttft_ms": {"p95": rps + 10.0},
            "tpot_ms": {"p95": rps + 1.0},
            "e2e_latency_ms": {"p95": rps + 100.0},
        },
    }


def write_result(
    results_dir: Path,
    budget: int,
    run: int,
    payload: dict,
) -> None:
    path = results_dir / (
        f"vllm_batched_tokens_{budget}_c8_long_run{run}.json"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_summary(results_dir: Path, expected_runs: int = 3) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--results-dir",
            str(results_dir),
            "--expected-runs",
            str(expected_runs),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_summary_reports_median_and_range(tmp_path: Path) -> None:
    for run, rps in enumerate((10.0, 20.0, 30.0), start=1):
        write_result(tmp_path, budget=1024, run=run, payload=make_payload(rps))

    completed = run_summary(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "| 1024 | 3 | 20.00 [10.00, 30.00] |" in completed.stdout
    assert "600.00 tokens/s [300.00, 900.00]" in completed.stdout
    assert "30.00 ms [20.00, 40.00]" in completed.stdout
    assert "error_rate=0.00%" not in completed.stdout


def test_summary_warns_when_run_count_is_incomplete(tmp_path: Path) -> None:
    write_result(tmp_path, budget=2048, run=1, payload=make_payload(25.0))

    completed = run_summary(tmp_path, expected_runs=3)

    assert completed.returncode == 0, completed.stderr
    assert "warning: budget=2048 发现 1 轮，预期 3 轮。" in completed.stderr


def test_summary_rejects_mixed_load_configurations(tmp_path: Path) -> None:
    write_result(tmp_path, budget=4096, run=1, payload=make_payload(20.0))
    write_result(
        tmp_path,
        budget=4096,
        run=2,
        payload=make_payload(30.0, concurrency=16),
    )

    completed = run_summary(tmp_path, expected_runs=2)

    assert completed.returncode != 0
    assert "配置不一致" in completed.stderr