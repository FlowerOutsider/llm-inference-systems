# Environment Baseline

## Purpose

This document records the local development baseline for the LLM Inference Systems project.

## System

- OS: Ubuntu 22.04.5 LTS on WSL 2
- Kernel: 6.18.33.2-microsoft-standard-WSL2
- CPU available to WSL: 20 logical processors
- Memory available to WSL: 7.6 GiB
- WSL root filesystem available space: about 945 GiB

## GPU

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- VRAM: 6144 MiB
- Compute Capability: 8.6
- NVIDIA Driver: 566.14
- Driver CUDA compatibility: 12.7

## Development Tools

| Tool | Status |
| --- | --- |
| Python | 3.10.12 |
| g++ | 11.4.0 |
| CMake | 3.22.1 |
| Git | 2.34.1 |
| Docker | 29.2.1 |
| Docker Compose | v5.0.2 |
| CUDA Toolkit / nvcc | Not installed |
| Nsight Systems / nsys | Not installed |
| Nsight Compute / ncu | Not installed |

## Docker GPU Verification

Docker Desktop disk image location:

```text
D:\wsl-files\docker-desktop-data\DockerDesktopWSL
```

Verified command:


docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi



Result: the container detected the RTX 3060 Laptop GPU successfully.

## Local and Remote Compute Strategy


Local RTX 3060 is used for CUDA, Triton, profiling, lightweight PyTorch experiments, and small or quantized model inference.

AutoDL is used for multi-GPU inference, distributed communication, PD disaggregation, TensorRT-LLM, large-model serving, K8s/Ray, and RL infrastructure experiments.