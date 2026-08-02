from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols

    x = tl.load(
        x_ptr + row_idx * n_cols + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inverse_rms = tl.rsqrt(mean_square + epsilon)
    output = x * inverse_rms * weight

    tl.store(
        output_ptr + row_idx * n_cols + offsets,
        output,
        mask=mask,
    )

@triton.jit
def _qwen_rmsnorm_fp16_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols

    x_fp16 = tl.load(
        x_ptr + row_idx * n_cols + offsets,
        mask=mask,
        other=0.0,
    )
    weight_fp16 = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    x_fp32 = x_fp16.to(tl.float32)
    mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    inverse_rms = tl.rsqrt(mean_square + epsilon)

    # Match Qwen: normalize in FP32, cast to FP16, then multiply weight.
    normalized_fp16 = (x_fp32 * inverse_rms).to(tl.float16)
    output_fp16 = (weight_fp16 * normalized_fp16).to(tl.float16)

    tl.store(
        output_ptr + row_idx * n_cols + offsets,
        output_fp16,
        mask=mask,
    )



def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """FP32-accumulation RMSNorm reference implementation."""
    _validate_inputs(x, weight, epsilon)

    mean_square = x.float().square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + epsilon)
    return (x.float() * inverse_rms * weight.float()).to(dtype=x.dtype)


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    num_warps: int = 4,
) -> torch.Tensor:
    """RMSNorm for a contiguous CUDA tensor whose last dimension is hidden size."""
    _validate_inputs(x, weight, epsilon)

    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of: 1, 2, 4, 8")

    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    block_size = 1 << (hidden_size - 1).bit_length()

    if block_size > 65536:
        raise ValueError(
            f"hidden size {hidden_size} requires block size {block_size}, "
            "which exceeds this baseline kernel limit"
        )

    output = torch.empty_like(x)
    x_2d = x.view(rows, hidden_size)
    output_2d = output.view(rows, hidden_size)

    _rmsnorm_kernel[(rows,)](
        x_2d,
        weight,
        output_2d,
        n_cols=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )

    return output


def rmsnorm_triton_qwen_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(x, weight, epsilon)

    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of: 1, 2, 4, 8")

    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("Qwen-compatible kernel requires float16 inputs")

    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    block_size = 1 << (hidden_size - 1).bit_length()

    if block_size > 65536:
        raise ValueError(f"hidden size {hidden_size} exceeds this kernel limit")

    output = torch.empty_like(x)

    _qwen_rmsnorm_fp16_kernel[(rows,)](
        x.view(rows, hidden_size),
        weight,
        output.view(rows, hidden_size),
        n_cols=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )

    return output


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")

    if x.device.type != "cuda":
        raise ValueError(f"x must be a CUDA tensor, got {x.device}")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported x dtype: {x.dtype}")

    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    if weight.ndim != 1:
        raise ValueError("weight must be one-dimensional")

    if weight.device != x.device:
        raise ValueError(
            f"x and weight must use the same device, got {x.device} and {weight.device}"
        )

    if weight.dtype != x.dtype:
        raise ValueError(
            f"x and weight must use the same dtype, got {x.dtype} and {weight.dtype}"
        )

    if weight.numel() != x.shape[-1]:
        raise ValueError(
            f"weight size must equal hidden size, got {weight.numel()} and {x.shape[-1]}"
        )

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")