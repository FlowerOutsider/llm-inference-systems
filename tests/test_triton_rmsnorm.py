from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels" / "triton" / "01_rmsnorm"
sys.path.insert(0, str(KERNEL_DIR))

from rmsnorm import rmsnorm_reference, rmsnorm_triton


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton RMSNorm tests require CUDA",
)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((8, 896), torch.float16),
        ((32, 1024), torch.float16),
        ((4, 2, 896), torch.float16),
    ],
)
def test_triton_rmsnorm_matches_fp32_accumulation_reference(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)

    expected = rmsnorm_reference(x, weight)
    actual = rmsnorm_triton(x, weight)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_triton_rmsnorm_rejects_cpu_input() -> None:
    x = torch.randn((2, 8), dtype=torch.float16)
    weight = torch.ones((8,), dtype=torch.float16)

    with pytest.raises(ValueError, match="CUDA"):
        rmsnorm_triton(x, weight)


def test_triton_rmsnorm_rejects_wrong_weight_size() -> None:
    x = torch.randn((2, 8), device="cuda", dtype=torch.float16)
    weight = torch.ones((7,), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="weight size"):
        rmsnorm_triton(x, weight)