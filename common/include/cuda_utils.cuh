#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>

// Compute 1D launch parameters
static inline dim3 pmpp_grid_for(int n, int block) {
    return dim3((n + block - 1) / block);
}
#endif