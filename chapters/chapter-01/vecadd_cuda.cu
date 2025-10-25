#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "cuda_utils.cuh"

__global__ void vecAddKernel(const float* A, const float* B, float* C, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    size_t n = 1<<24; // ~16M
    if (argc > 1) n = strtoull(argv[1], NULL, 10);
    size_t bytes = n * sizeof(float);

    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes);
    float *C_h = (float*)malloc(bytes);
    for (size_t i = 0; i < n; ++i) { A_h[i] = (float)i; B_h[i] = (float)(2*i); }

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&C_d, bytes));

    // Timers
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    int block = 256;
    dim3 grid = pmpp_grid_for((int)n, block);

    CUDA_CHECK(cudaEventRecord(start));
    vecAddKernel<<<grid, block>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // quick check
    int ok = 1;
    for (size_t i = 0; i < 10; ++i) {
        if (C_h[i] != A_h[i] + B_h[i]) { ok = 0; break; }
    }
    printf("[CUDA] n=%zu, kernel=%.3f ms, ok=%s\\n", n, ms, ok ? "yes" : "no");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A_h); free(B_h); free(C_h);
    return ok ? 0 : 1;
}