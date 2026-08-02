# Qwen 0.5B Triton RMSNorm Evaluation

## Scope

This is an experimental evaluation of a custom Triton RMSNorm kernel on
NVIDIA GeForce RTX 3060 Laptop GPU.

The kernel is not integrated into vLLM or any production serving path.

## Kernel-Level Result

Workload:

- Shape: `(4096, 896)`
- Dtype: FP16
- RMSNorm hidden size: 896
- Selected configuration: `num_warps=4`

| Implementation | Latency | Result |
|---|---:|---:|
| PyTorch eager reference | 0.5673 ms | Baseline |
| Triton fused RMSNorm | 0.0488 ms | 11.637x faster |

NCU profile:

- Kernel duration: 48.42 us
- DRAM throughput: 85.65%
- Compute throughput: 28.68%
- Achieved occupancy: 90.14%
- Registers per thread: 30

Conclusion: the fused RMSNorm kernel is memory-bandwidth bound. Increasing
`num_warps` beyond 4 does not produce a meaningful benefit.

## Qwen Module-Level Result

Real module:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Module: first-layer `Qwen2RMSNorm`
- Hidden-state shape: `(1, 512, 896)`

| Implementation | Latency | Speedup | Max Absolute Error |
|---|---:|---:|---:|
| Qwen2RMSNorm | 0.1124 ms | Baseline | N/A |
| Generic Triton RMSNorm | 0.0323 ms | 3.476x | 0.001953 |
| Qwen-compatible Triton experiment | 0.0394 ms | 3.087x | 0.001953 |

The Qwen-compatible experiment reproduced Qwen's FP16 cast location, but it
did not reduce the measured absolute error and made the kernel slower.
It is rejected as an optimization candidate.

## Model-Level Prefill Result

The generic Triton replacement changed all 49 Qwen RMSNorm modules.

| Metric | Baseline | Triton Replacement |
|---|---:|---:|
| Prefill latency | 34.1563 ms | 30.3551 ms |
| End-to-end speedup | - | 1.125x |
| Maximum logit difference | - | 0.138672 |
| Mean logit difference | - | 0.005672 |
| Top-1 logit agreement | - | 0.996094 |

The kernel-level speedup does not translate linearly to the whole model.
Attention, GEMM, MLP, and LM Head remain significant prefill costs.

## Generation Parity Result

Configuration:

- Greedy decoding: `do_sample=False`
- Four fixed prompts
- `max_new_tokens=32`

Result:

- Exact prompt matches: 3/4
- Overall generated-token agreement: 0.750000
- One prompt diverged from the first generated token.

## Decision

The generic Triton RMSNorm kernel is retained as a performance experiment.

It must not replace Qwen RMSNorm in a production inference path because
generation behavior is not fully equivalent. The observed mismatch is
consistent with different FP32 reduction orders and FP16 rounding behavior
between Triton and PyTorch.

Any production adoption would require a larger regression corpus, task-level
quality evaluation, tolerance policy, rollout controls, and rollback support.