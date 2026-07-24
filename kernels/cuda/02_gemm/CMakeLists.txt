cmake_minimum_required(VERSION 3.22)

project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(gemm gemm.cu)