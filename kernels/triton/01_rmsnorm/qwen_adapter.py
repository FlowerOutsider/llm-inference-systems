from __future__ import annotations

import torch
from torch import nn

from rmsnorm import rmsnorm_triton


class TritonRMSNorm(nn.Module):
    """Inference-only RMSNorm module backed by the Triton kernel."""

    def __init__(
        self,
        weight: torch.Tensor,
        epsilon: float,
        num_warps: int = 4,
    ) -> None:
        super().__init__()

        if weight.ndim != 1:
            raise ValueError("weight must be one-dimensional")

        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.variance_epsilon = float(epsilon)
        self.num_warps = num_warps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rmsnorm_triton(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon,
            num_warps=self.num_warps,
        )