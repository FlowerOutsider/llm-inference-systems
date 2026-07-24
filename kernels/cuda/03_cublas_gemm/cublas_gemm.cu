#include <cublas_v2.h>
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

void checkCublas(cublasStatus_t status, const char* expression,
                 const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error code: ") +
            std::to_string(static_cast<int>(status)) +
            " at " + file + ":" + std::to_string(line) +
            " for " + expression);
    }
}

#define CHECK_CUDA(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

#define CHECK_CUBLAS(expression) \
    checkCublas((expression), #expression, __FILE__, __LINE__)

float benchmarkSgemm(
    cublasHandle_t handle,
    cublasMath_t mathMode,
    const float* deviceA,
    const float* deviceB,
    float* deviceC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSetMathMode(handle, mathMode));

    for (int i = 0; i < kWarmupIterations; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            kMatrixSize,
            kMatrixSize,
            kMatrixSize,
            &alpha,
            deviceA,
            kMatrixSize,
            deviceB,
            kMatrixSize,
            &beta,
            deviceC,
            kMatrixSize));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            kMatrixSize,
            kMatrixSize,
            kMatrixSize,
            &alpha,
            deviceA,
            kMatrixSize,
            deviceB,
            kMatrixSize,
            &beta,
            deviceC,
            kMatrixSize));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMilliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return totalMilliseconds / kBenchmarkIterations;
}

float referenceValue(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int row,
    int col) {
    float sum = 0.0f;

    // cuBLAS 默认使用 column-major layout。
    for (int k = 0; k < kMatrixSize; ++k) {
        sum += a[row + k * kMatrixSize] *
               b[k + col * kMatrixSize];
    }

    return sum;
}

struct ErrorMetrics {
    float maxAbsolute = 0.0f;
    float maxRelative = 0.0f;
};

ErrorMetrics sampleErrorMetrics(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& actual) {
    constexpr std::array<int, 4> sampleIndices{
        0, 17, kMatrixSize / 2, kMatrixSize - 1};

    ErrorMetrics metrics{};

    for (const int row : sampleIndices) {
        for (const int col : sampleIndices) {
            const float expected = referenceValue(a, b, row, col);
            const float observed = actual[row + col * kMatrixSize];
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

double gflopsFromMilliseconds(float milliseconds) {
    const double operations =
        2.0 * kMatrixSize * kMatrixSize * kMatrixSize;

    return operations / (milliseconds / 1000.0) / 1e9;
}

int main() {
    const std::size_t elementCount =
        static_cast<std::size_t>(kMatrixSize) * kMatrixSize;
    const std::size_t bytes = elementCount * sizeof(float);

    std::vector<float> hostA(elementCount);
    std::vector<float> hostB(elementCount);
    std::vector<float> hostFp32(elementCount);
    std::vector<float> hostTf32(elementCount);

    // 以 column-major 方式填充，和 cuBLAS 原生布局一致。
    for (int col = 0; col < kMatrixSize; ++col) {
        for (int row = 0; row < kMatrixSize; ++row) {
            const std::size_t index =
                static_cast<std::size_t>(row) +
                static_cast<std::size_t>(col) * kMatrixSize;

            hostA[index] =
                static_cast<float>((row * 17 + col * 13) % 101) / 101.0f;
            hostB[index] =
                static_cast<float>((row * 31 + col * 7) % 97) / 97.0f;
        }
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA(cudaMalloc(&deviceC, bytes));

    CHECK_CUDA(cudaMemcpy(
        deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));

    const float fp32Milliseconds = benchmarkSgemm(
        handle,
        CUBLAS_PEDANTIC_MATH,
        deviceA,
        deviceB,
        deviceC);

    CHECK_CUDA(cudaMemcpy(
        hostFp32.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    const float tf32Milliseconds = benchmarkSgemm(
        handle,
        CUBLAS_TF32_TENSOR_OP_MATH,
        deviceA,
        deviceB,
        deviceC);

    CHECK_CUDA(cudaMemcpy(
        hostTf32.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

const ErrorMetrics fp32Error =
    sampleErrorMetrics(hostA, hostB, hostFp32);
const ErrorMetrics tf32Error =
    sampleErrorMetrics(hostA, hostB, hostTf32);

    const double fp32Gflops = gflopsFromMilliseconds(fp32Milliseconds);
    const double tf32Gflops = gflopsFromMilliseconds(tf32Milliseconds);
    const double tf32Speedup = fp32Milliseconds / tf32Milliseconds;

std::cout << std::fixed << std::setprecision(3)
          << "Matrix size: " << kMatrixSize << " x "
          << kMatrixSize << "\n"
          << "Layout: column-major\n"
          << "cuBLAS FP32 latency (ms): " << fp32Milliseconds << "\n"
          << "cuBLAS FP32 throughput (GFLOPS): " << fp32Gflops << "\n"
          << std::setprecision(6)
          << "cuBLAS FP32 sampled max absolute error: "
          << fp32Error.maxAbsolute << "\n"
          << "cuBLAS FP32 sampled max relative error: "
          << fp32Error.maxRelative << "\n"
          << std::setprecision(3)
          << "cuBLAS TF32 latency (ms): " << tf32Milliseconds << "\n"
          << "cuBLAS TF32 throughput (GFLOPS): " << tf32Gflops << "\n"
          << std::setprecision(6)
          << "cuBLAS TF32 sampled max absolute error: "
          << tf32Error.maxAbsolute << "\n"
          << "cuBLAS TF32 sampled max relative error: "
          << tf32Error.maxRelative << "\n"
          << std::setprecision(3)
          << "TF32 speedup over FP32: " << tf32Speedup << "x\n";


          
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceC));

    return EXIT_SUCCESS;
}