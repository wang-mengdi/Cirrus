#include "HAGrid.h"


void CheckCudaError(const std::string& message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << fmt::format("CUDA error at {}: {}", message, cudaGetErrorString(err)) << std::endl;
    }
}