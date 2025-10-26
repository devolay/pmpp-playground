#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "cuda_utils.cuh"
#include "utils.h"

__global__ void matMulKernel(const float* A, const float* B, float* C, int width, int height) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float dotProduct = 0.0;
        for (int k = 0; k < width; k++){
            dotProduct += A[row*width + k]*B[width*k + col];
        }
        C[row*width + col] = dotProduct;
    }
}

__global__ void matMulPerRowKernel(const float* A, const float* B, float* C, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height) {
        for (int col = 0; col < width; col++) {
            float dotProduct = 0.0;
            for (int k = 0; k < width; k++){
                dotProduct += A[row*width + k]*B[width*k + col];
            }
            C[row*width + col] = dotProduct;
        }
    }
}

__global__ void matMulPerColKernel(const float* A, const float* B, float* C, int width, int height) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width) {
        for (int row = 0; row < height; row++){
            float dotProduct = 0.0;
            for (int k = 0; k < width; k++){
                dotProduct += A[row*width + k]*B[width*k + col];
            }
            C[row*width + col] = dotProduct;
        }
    }
}

static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            float acc = 0.0f;
            for (int k=0;k<n;k++) acc += A[i*n + k] * B[k*n + j];
            C[i*n + j] = acc;
        }
    }
}

static void stats_ms(const float* ms, int count, float* mean, float* minv, float* stddev){
    float s=0.0f, ss=0.0f, mn=ms[0];
    for (int i=0;i<count;i++){ s+=ms[i]; ss+=ms[i]*ms[i]; if (ms[i]<mn) mn=ms[i]; }
    *mean = s / count;
    *minv = mn;
    float var = fmaxf(0.0f, ss/count - (*mean)*(*mean));
    *stddev = sqrtf(var);
}


typedef void (*KernelLauncher)(const float* A, const float* B, float* C, int n);

static void benchmark_kernel(const char* name,
                             KernelLauncher launch,
                             const float* A_d, const float* B_d, float* C_d,
                             int n, int warmup, int iters) {
    for (int i=0;i<warmup;i++){
        launch(A_d,B_d,C_d,n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float *runs = (float*)malloc(iters*sizeof(float));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i=0;i<iters;i++){
        CUDA_CHECK(cudaEventRecord(start));
        launch(A_d,B_d,C_d,n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        runs[i]=ms;
    }

    float mean, minv, stddev;
    stats_ms(runs, iters, &mean, &minv, &stddev);

    double flops = 2.0 * (double)n * n * n;
    double gflops_mean = (flops / (mean/1000.0)) / 1e9;
    double gflops_min  = (flops / (minv/1000.0)) / 1e9;

    printf("%-18s n=%d  avg=%.3f ms  min=%.3f ms  std=%.3f ms  "
           "GFLOP/s(avg)=%.2f  GFLOP/s(best)=%.2f\n",
           name, n, mean, minv, stddev, gflops_mean, gflops_min);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(runs);
}

static void launch_matMul(const float* A, const float* B, float* C, int n) {
    dim3 block(16,16);
    dim3 grid( (n + block.x - 1)/block.x,
               (n + block.y - 1)/block.y );
    matMulKernel<<<grid, block>>>(A,B,C,n,n);
    CUDA_CHECK(cudaGetLastError());
}

static void launch_matMulPerRow(const float* A, const float* B, float* C, int n) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    matMulPerRowKernel<<<grid, block>>>(A,B,C,n,n);
    CUDA_CHECK(cudaGetLastError());
}

static void launch_matMulPerCol(const float* A, const float* B, float* C, int n) {
    dim3 block(1, 256);
    dim3 grid(1, (n + block.y - 1) / block.y);
    matMulPerColKernel<<<grid, block>>>(A,B,C,n,n);
    CUDA_CHECK(cudaGetLastError());
}


int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    const int WARMUP = 3;
    const int ITERS  = 10;

    srand(42);
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A_h = generateRandomSquareMatrix(N);
    float *B_h = generateRandomSquareMatrix(N);
    float *C_h = (float*)malloc(bytes);
    float *Ref = (float*)malloc(bytes);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&C_d, bytes));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    matmul_cpu(A_h, B_h, Ref, N);

    benchmark_kernel("matMulKernel", launch_matMul, A_d,B_d,C_d, N, WARMUP, ITERS);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    benchmark_kernel("perRowKernel", launch_matMulPerRow, A_d,B_d,C_d, N, WARMUP, ITERS);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    benchmark_kernel("perColKernel", launch_matMulPerCol, A_d,B_d,C_d, N, WARMUP, ITERS);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A_h); free(B_h); free(C_h); free(Ref);
    return 0;
}