#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "cuda_utils.cuh"
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__device__ const int CHANNELS = 3;

__global__
void colorToGrayConvertion(unsigned char *p_out, unsigned char *p_in, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row*width + col;
        int rgbOffset = grayOffset*CHANNELS;

        unsigned char r = p_in[rgbOffset];
        unsigned char g = p_in[rgbOffset + 1];
        unsigned char b = p_in[rgbOffset + 2];

        p_out[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main() {
    int width, height, channels;
    unsigned char* imgIn_h = stbi_load("/home/dawid.stachowiak/bookclub/pmpp-playground/bird.jpeg", &width, &height, &channels, 0);
    
    size_t bytes = (width * height* sizeof(unsigned char));
    unsigned char* imgOut_h = (unsigned char*)malloc(bytes);

    unsigned char* imgIn_d;
    unsigned char* imgOut_d;

    CUDA_CHECK(cudaMalloc((void**)&imgIn_d, bytes*channels));
    CUDA_CHECK(cudaMalloc((void**)&imgOut_d, bytes));

    CUDA_CHECK(cudaMemcpy(imgIn_d, imgIn_h, bytes*channels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(imgOut_d, imgOut_h, bytes, cudaMemcpyHostToDevice));
    
    dim3 block = dim3(4, 4);
    dim3 grid = dim3(ceil(width / block.y), ceil(height / block.x));

    colorToGrayConvertion<<<grid, block>>>(imgOut_d, imgIn_d, width, height);

    CUDA_CHECK(cudaMemcpy(imgOut_h, imgOut_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(imgOut_d));
    CUDA_CHECK(cudaFree(imgIn_d));

    if (stbi_write_png("output.png", width, height, 1, imgOut_h, width))
        printf("Saved output.png!\n");
    else
        printf("Failed to save image\n");

    free(imgOut_h); free(imgIn_h);

    return 0;
}