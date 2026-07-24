cmake_minimum_required(VERSION 3.22)

project(wmma_tensor_core LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(wmma_gemm wmma_gemm.cu)