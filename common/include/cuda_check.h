#pragma once
#include <stdio.h>
#include <stdlib.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#else
#define CUDA_CHECK(x) x
#endif