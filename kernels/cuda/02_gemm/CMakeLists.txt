cmake_minimum_required(VERSION 3.22)

project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(GEMM_TILE_SIZE "16" CACHE STRING "Square tile dimension for tiled GEMM")

add_executable(gemm gemm.cu)

target_compile_definitions(
    gemm
    PRIVATE GEMM_TILE_SIZE=${GEMM_TILE_SIZE}
)