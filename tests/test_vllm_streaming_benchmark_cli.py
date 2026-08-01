import subprocess
import sys
from pathlib import Path


SCRIPT = Path("benchmarks/serving/vllm_streaming_benchmark.py")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_base_url_is_required() -> None:
    result = run_cli("--requests", "1")

    assert result.returncode == 2
    assert "--base-url" in result.stderr
    assert "required" in result.stderr


def test_explicit_base_url_reaches_other_argument_validation() -> None:
    result = run_cli(
        "--base-url",
        "http://127.0.0.1:8002",
        "--requests",
        "0",
    )

    assert result.returncode == 2
    assert "--requests must be positive" in result.stderr