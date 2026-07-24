#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kMatrixSize = 1024;
constexpr int kTileSize = 16;
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 20;

void checkCuda(cudaError_t status, const char* expression,
               const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(status) +
            " at " + file + ":" + std::to_string(line) +
            " for " + expression);
    }
}

#define CHECK_CUDA(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

__global__ void naiveGemm(
    const float* a,
    const float* b,
    float* c,
    int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = sum;
}

__global__ void tiledGemm(
    const float* a,
    const float* b,
    float* c,
    int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (n + kTileSize - 1) / kTileSize; ++tile) {
        const int aCol = tile * kTileSize + threadIdx.x;
        const int bRow = tile * kTileSize + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

float measureNaiveGemm(
    const float* deviceA,
    const float* deviceB,
    float* deviceC,
    dim3 grid,
    dim3 block) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        naiveGemm<<<grid, block>>>(deviceA, deviceB, deviceC, kMatrixSize);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMilliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return totalMilliseconds / kBenchmarkIterations;
}

float measureTiledGemm(
    const float* deviceA,
    const float* deviceB,
    float* deviceC,
    dim3 grid,
    dim3 block) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        tiledGemm<<<grid, block>>>(deviceA, deviceB, deviceC, kMatrixSize);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMilliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return totalMilliseconds / kBenchmarkIterations;
}

float computeReferenceValue(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int row,
    int col,
    int n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    return sum;
}

int main() {
    const std::size_t elementCount =
        static_cast<std::size_t>(kMatrixSize) * kMatrixSize;
    const std::size_t bytes = elementCount * sizeof(float);

    std::vector<float> hostA(elementCount);
    std::vector<float> hostB(elementCount);
    std::vector<float> hostNaive(elementCount);
    std::vector<float> hostTiled(elementCount);

    for (std::size_t i = 0; i < elementCount; ++i) {
        hostA[i] = static_cast<float>((i * 17) % 101) / 101.0f;
        hostB[i] = static_cast<float>((i * 31) % 97) / 97.0f;
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceNaive = nullptr;
    float* deviceTiled = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA(cudaMalloc(&deviceNaive, bytes));
    CHECK_CUDA(cudaMalloc(&deviceTiled, bytes));

    CHECK_CUDA(cudaMemcpy(
        deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid(
        (kMatrixSize + kTileSize - 1) / kTileSize,
        (kMatrixSize + kTileSize - 1) / kTileSize);

    for (int i = 0; i < kWarmupIterations; ++i) {
        naiveGemm<<<grid, block>>>(
            deviceA, deviceB, deviceNaive, kMatrixSize);
        tiledGemm<<<grid, block>>>(
            deviceA, deviceB, deviceTiled, kMatrixSize);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const float naiveMilliseconds = measureNaiveGemm(
        deviceA, deviceB, deviceNaive, grid, block);
    const float tiledMilliseconds = measureTiledGemm(
        deviceA, deviceB, deviceTiled, grid, block);

    CHECK_CUDA(cudaMemcpy(
        hostNaive.data(), deviceNaive, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        hostTiled.data(), deviceTiled, bytes, cudaMemcpyDeviceToHost));

    float maxDifference = 0.0f;
    for (std::size_t i = 0; i < elementCount; ++i) {
        maxDifference = std::max(
            maxDifference,
            std::fabs(hostNaive[i] - hostTiled[i]));
    }

    float maxReferenceError = 0.0f;
    constexpr std::array<int, 4> sampleIndices{
        0, 17, kMatrixSize / 2, kMatrixSize - 1};

    for (const int row : sampleIndices) {
        for (const int col : sampleIndices) {
            const float expected = computeReferenceValue(
                hostA, hostB, row, col, kMatrixSize);
            const float actual = hostTiled[row * kMatrixSize + col];
            maxReferenceError = std::max(
                maxReferenceError,
                std::fabs(expected - actual));
        }
    }

    const double operations =
        2.0 * kMatrixSize * kMatrixSize * kMatrixSize;
    const double naiveGflops =
        operations / (naiveMilliseconds / 1000.0) / 1e9;
    const double tiledGflops =
        operations / (tiledMilliseconds / 1000.0) / 1e9;
    const double speedup = naiveMilliseconds / tiledMilliseconds;

    std::cout << std::fixed << std::setprecision(3)
              << "Matrix size: " << kMatrixSize << " x "
              << kMatrixSize << "\n"
              << "Launch config: " << grid.x << " x " << grid.y
              << " blocks, " << block.x << " x " << block.y
              << " threads\n"
              << "Naive GEMM latency (ms): " << naiveMilliseconds << "\n"
              << "Naive GEMM throughput (GFLOPS): " << naiveGflops << "\n"
              << "Tiled GEMM latency (ms): " << tiledMilliseconds << "\n"
              << "Tiled GEMM throughput (GFLOPS): " << tiledGflops << "\n"
              << "Tiled speedup: " << speedup << "x\n"
              << "Naive vs tiled max difference: " << maxDifference << "\n"
              << "Sampled reference max error: " << maxReferenceError << "\n";

    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceNaive));
    CHECK_CUDA(cudaFree(deviceTiled));

    return (maxDifference < 1e-3f && maxReferenceError < 1e-3f)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}