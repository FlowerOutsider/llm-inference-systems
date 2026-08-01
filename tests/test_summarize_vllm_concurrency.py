import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/summarize_vllm_concurrency.py")


def write_result(
    directory: Path,
    *,
    concurrency: int,
    run: int,
    rps: float,
    model: str = "qwen2.5-0.5b",
) -> None:
    payload = {
        "configuration": {
            "base_url": "http://127.0.0.1:8002",
            "model": model,
            "warmup_requests": 16,
            "requests": 256,
            "concurrency": concurrency,
            "max_tokens": 64,
            "shared_prefix_repeats": 32,
        },
        "results": {
            "request_throughput_rps": rps,
            "generation_throughput_tokens_per_second": rps * 30,
            "error_rate_percent": 0.0,
            "ttft_ms": {"p95": rps + 10},
            "tpot_ms": {"p95": rps + 1},
            "e2e_latency_ms": {"p95": rps + 100},
        },
    }

    path = directory / f"vllm_concurrency_{concurrency}_batch2048_run{run}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_summary(results_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-dir",
            str(results_dir),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_summary_reports_median_and_range(tmp_path: Path) -> None:
    for run, rps in enumerate((10.0, 20.0, 30.0), start=1):
        write_result(tmp_path, concurrency=4, run=run, rps=rps)

    result = run_summary(tmp_path)

    assert result.returncode == 0
    assert "| 4 | 2048 | 3 | 20.00 [10.00, 30.00] |" in result.stdout
    assert "600.00 tokens/s [300.00, 900.00]" in result.stdout
    assert "http://127.0.0.1:8002" in result.stdout


def test_summary_warns_about_incomplete_repetitions(tmp_path: Path) -> None:
    write_result(tmp_path, concurrency=1, run=1, rps=10.0)
    write_result(tmp_path, concurrency=1, run=2, rps=20.0)

    result = run_summary(tmp_path)

    assert result.returncode == 0
    assert "## 警告" in result.stdout
    assert "has 2 runs, expected 3" in result.stdout


def test_summary_rejects_cross_group_configuration_mismatch(tmp_path: Path) -> None:
    for run in range(1, 4):
        write_result(tmp_path, concurrency=1, run=run, rps=10.0)
        write_result(
            tmp_path,
            concurrency=4,
            run=run,
            rps=20.0,
            model="different-model",
        )

    result = run_summary(tmp_path)

    assert result.returncode == 1
    assert "configuration mismatch across concurrency groups" in result.stderr