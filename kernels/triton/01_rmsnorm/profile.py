from __future__ import annotations

import torch

from rmsnorm import rmsnorm_triton


def main() -> None:
    torch.manual_seed(7)

    x = torch.randn((4096, 896), device="cuda", dtype=torch.float16)
    weight = torch.randn((896,), device="cuda", dtype=torch.float16)

    for _ in range(20):
        rmsnorm_triton(x, weight)

    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("rmsnorm_triton")
    output = rmsnorm_triton(x, weight)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("output checksum:", output.float().sum().item())


if __name__ == "__main__":
    main()