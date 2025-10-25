#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::cerr << "CUDA error " << cudaGetErrorString(err__)     \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(EXIT_FAILURE);                                     \
    }                                                               \
} while (0)

using namespace std;

__global__
void addVecKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void addVectorsCPU(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

void addVectorsGPU(float* A_h, float* B_h, float* C_h, int n, float &gpuTimeMs) {
    size_t size = n * sizeof(int);
    float *A_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc((void **)&A_d, size));
    CUDA_CHECK(cudaMalloc((void **)&B_d, size));
    CUDA_CHECK(cudaMalloc((void **)&C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice)); // Destination, Source, Size, Flag
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice)); // Dereferencing pointer with '->' to get underlying data

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    addVecKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n;

    cout << "Vector length for benchmark: ";
    cin >> n ;

    float* A = generateRandomVector(n);
    float* B = generateRandomVector(n);
    float C_cpu[n];
    float C_gpu[n];

    auto cpu_start = chrono::high_resolution_clock::now();
    addVectorsCPU(A, B, C_cpu, n);
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    float gpu_time = 0.0f;
    addVectorsGPU(A, B, C_gpu, n, gpu_time);

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (C_cpu[i] != C_gpu[i]) { ok = false; break; }
    }

    cout << "Results match: " << (ok ? "YES" : "NO") << endl;
    cout << "CPU time: " << cpu_time << " ms" << endl;
    cout << "GPU time: " << gpu_time << " ms" << endl;

    return 0;
}
