#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvcuda;

constexpr int kMatrixSize = 1024;
constexpr int kTileSize = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kWarmupIterations = 10;
constexpr int kBenchmarkIterations = 100;

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

struct ErrorMetrics {
    float maxAbsolute = 0.0f;
    float maxRelative = 0.0f;
};

__global__ void wmmaGemm(
    const __half* a,
    const __half* b,
    float* c,
    int n) {
    const int warpId = threadIdx.y;
    const int tileRow = blockIdx.y * kWarpsPerBlock + warpId;
    const int tileCol = blockIdx.x;

    const int row = tileRow * kTileSize;
    const int col = tileCol * kTileSize;

    if (row >= n || col >= n) {
        return;
    }

    wmma::fragment<
        wmma::matrix_a,
        kTileSize,
        kTileSize,
        kTileSize,
        __half,
        wmma::row_major> matrixA;

    wmma::fragment<
        wmma::matrix_b,
        kTileSize,
        kTileSize,
        kTileSize,
        __half,
        wmma::row_major> matrixB;

    wmma::fragment<
        wmma::accumulator,
        kTileSize,
        kTileSize,
        kTileSize,
        float> accumulator;

    wmma::fill_fragment(accumulator, 0.0f);

    for (int k = 0; k < n; k += kTileSize) {
        wmma::load_matrix_sync(matrixA, a + row * n + k, n);
        wmma::load_matrix_sync(matrixB, b + k * n + col, n);
        wmma::mma_sync(accumulator, matrixA, matrixB, accumulator);
    }

    wmma::store_matrix_sync(
        c + row * n + col,
        accumulator,
        n,
        wmma::mem_row_major);
}

float benchmarkWmma(
    const __half* deviceA,
    const __half* deviceB,
    float* deviceC,
    dim3 grid,
    dim3 block) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        wmmaGemm<<<grid, block>>>(
            deviceA, deviceB, deviceC, kMatrixSize);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        wmmaGemm<<<grid, block>>>(
            deviceA, deviceB, deviceC, kMatrixSize);
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

float referenceValue(
    const std::vector<__half>& a,
    const std::vector<__half>& b,
    int row,
    int col) {
    float sum = 0.0f;

    for (int k = 0; k < kMatrixSize; ++k) {
        sum += __half2float(a[row * kMatrixSize + k]) *
               __half2float(b[k * kMatrixSize + col]);
    }

    return sum;
}

ErrorMetrics sampleErrorMetrics(
    const std::vector<__half>& a,
    const std::vector<__half>& b,
    const std::vector<float>& actual) {
    constexpr std::array<int, 4> sampleIndices{
        0, 17, kMatrixSize / 2, kMatrixSize - 1};

    ErrorMetrics metrics{};

    for (const int row : sampleIndices) {
        for (const int col : sampleIndices) {
            const float expected = referenceValue(a, b, row, col);
            const float observed = actual[row * kMatrixSize + col];
            const float absoluteError = std::fabs(expected - observed);
            const float relativeError =
                absoluteError / std::max(std::fabs(expected), 1e-6f);

            metrics.maxAbsolute =
                std::max(metrics.maxAbsolute, absoluteError);
            metrics.maxRelative =
                std::max(metrics.maxRelative, relativeError);
        }
    }

    return metrics;
}

int main() {
    static_assert(kMatrixSize % kTileSize == 0);

    const std::size_t elementCount =
        static_cast<std::size_t>(kMatrixSize) * kMatrixSize;
    const std::size_t halfBytes = elementCount * sizeof(__half);
    const std::size_t floatBytes = elementCount * sizeof(float);

    std::vector<__half> hostA(elementCount);
    std::vector<__half> hostB(elementCount);
    std::vector<float> hostC(elementCount);

    for (int row = 0; row < kMatrixSize; ++row) {
        for (int col = 0; col < kMatrixSize; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kMatrixSize + col;

            const float valueA =
                static_cast<float>((row * 17 + col * 13) % 101) / 101.0f;
            const float valueB =
                static_cast<float>((row * 31 + col * 7) % 97) / 97.0f;

            hostA[index] = __float2half(valueA);
            hostB[index] = __float2half(valueB);
        }
    }

    __half* deviceA = nullptr;
    __half* deviceB = nullptr;
    float* deviceC = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceA, halfBytes));
    CHECK_CUDA(cudaMalloc(&deviceB, halfBytes));
    CHECK_CUDA(cudaMalloc(&deviceC, floatBytes));

    CHECK_CUDA(cudaMemcpy(
        deviceA, hostA.data(), halfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        deviceB, hostB.data(), halfBytes, cudaMemcpyHostToDevice));

    const dim3 block(32, kWarpsPerBlock);
    const dim3 grid(
        kMatrixSize / kTileSize,
        (kMatrixSize / kTileSize + kWarpsPerBlock - 1) /
            kWarpsPerBlock);

    const float milliseconds =
        benchmarkWmma(deviceA, deviceB, deviceC, grid, block);

    CHECK_CUDA(cudaMemcpy(
        hostC.data(), deviceC, floatBytes, cudaMemcpyDeviceToHost));

    const ErrorMetrics error = sampleErrorMetrics(hostA, hostB, hostC);

    const double operations =
        2.0 * kMatrixSize * kMatrixSize * kMatrixSize;
    const double gflops =
        operations / (milliseconds / 1000.0) / 1e9;

    std::cout << std::fixed << std::setprecision(3)
              << "Matrix size: " << kMatrixSize << " x "
              << kMatrixSize << "\n"
              << "Input type: FP16\n"
              << "Accumulator type: FP32\n"
              << "WMMA tile: 16 x 16 x 16\n"
              << "Launch config: " << grid.x << " x " << grid.y
              << " blocks, " << block.x << " x " << block.y
              << " threads\n"
              << "WMMA latency (ms): " << milliseconds << "\n"
              << "WMMA throughput (GFLOPS): " << gflops << "\n"
              << std::setprecision(6)
              << "Sampled max absolute error: "
              << error.maxAbsolute << "\n"
              << "Sampled max relative error: "
              << error.maxRelative << "\n";

    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceC));

    return EXIT_SUCCESS;
}