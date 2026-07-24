cmake_minimum_required(VERSION 3.22)

project(cublas_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_executable(cublas_gemm cublas_gemm.cu)

target_link_libraries(cublas_gemm PRIVATE CUDA::cublas)