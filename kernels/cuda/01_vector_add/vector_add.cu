#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

void checkCuda(cudaError_t error, const char* expression,
               const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error: " << cudaGetErrorString(error)
              << "\nExpression: " << expression
              << "\nLocation: " << file << ":" << line << std::endl;
    std::exit(EXIT_FAILURE);
}

#define CHECK_CUDA(call) checkCuda((call), #call, __FILE__, __LINE__)

__global__ void vectorAdd(const float* a, const float* b,
                          float* output, std::size_t size) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                  blockDim.x +
                              threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) *
                               gridDim.x;

    for (std::size_t i = index; i < size; i += stride) {
        output[i] = a[i] + b[i];
    }
}


int main() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

    std::cout << "CUDA device: " << properties.name << "\n"
              << "Compute capability: " << properties.major
              << "." << properties.minor << "\n"
              << "Global memory (MiB): "
              << properties.totalGlobalMem / (1024 * 1024) << "\n"
              << "SM count: " << properties.multiProcessorCount << "\n";

    constexpr std::size_t elementCount = 1ULL << 24;
    constexpr int threadsPerBlock = 256;
    constexpr int repeatCount = 100;

    const std::size_t bytes = elementCount * sizeof(float);
    const int requestedBlocks =
        static_cast<int>((elementCount + threadsPerBlock - 1) /
                         threadsPerBlock);
    const int blocks =
        std::min(requestedBlocks, properties.multiProcessorCount * 32);

    std::vector<float> hostA(elementCount);
    std::vector<float> hostB(elementCount);
    std::vector<float> hostOutput(elementCount);

    for (std::size_t i = 0; i < elementCount; ++i) {
        hostA[i] = static_cast<float>(i) * 0.001f;
        hostB[i] = 1.0f - static_cast<float>(i) * 0.0001f;
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOutput = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput, bytes));

    CHECK_CUDA(cudaMemcpy(
        deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        vectorAdd<<<blocks, threadsPerBlock>>>(
            deviceA, deviceB, deviceOutput, elementCount);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeatCount; ++i) {
        vectorAdd<<<blocks, threadsPerBlock>>>(
            deviceA, deviceB, deviceOutput, elementCount);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMilliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(
        hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (std::size_t i = 0; i < elementCount; ++i) {
        const float expected = hostA[i] + hostB[i];
        maxError = std::max(maxError, std::fabs(hostOutput[i] - expected));
    }

    const double averageMilliseconds = totalMilliseconds / repeatCount;
    const double bandwidthGBps =
        (3.0 * bytes) / (averageMilliseconds / 1000.0) / 1e9;

    std::cout << std::fixed << std::setprecision(3)
              << "Elements: " << elementCount << "\n"
              << "Launch config: " << blocks << " blocks x "
              << threadsPerBlock << " threads\n"
              << "Average kernel latency (ms): " << averageMilliseconds
              << "\n"
              << "Effective bandwidth (GB/s): " << bandwidthGBps << "\n"
              << "Maximum absolute error: " << maxError << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceOutput));

    return maxError == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}