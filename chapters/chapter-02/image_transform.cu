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


__global__
void colorToGrayKernel(unsigned char *p_out, unsigned char *p_in, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row*width + col;
        int rgbOffset = grayOffset * 3;

        unsigned char r = p_in[rgbOffset];
        unsigned char g = p_in[rgbOffset + 1];
        unsigned char b = p_in[rgbOffset + 2];

        p_out[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__
void blurKernel(unsigned char *p_out, unsigned char *p_in, int width, int height, int blur_size) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixCount = 0;

        for (int blurCol=-blur_size; blurCol<blur_size+1; blurCol++) {
            for (int blurRow=-blur_size; blurRow<blur_size+1; blurRow++) {
                int currCol = col + blurCol;
                int currRow = row + blurRow;
                if (currCol > 0 && currCol < width && currRow > 0 && currRow < height) {
                    pixVal += p_in[currRow*width + currCol];
                    pixCount += 1;
                }
            }
        }

        p_out[row*width + col] = (unsigned char)(pixVal / pixCount);
    }
}

unsigned char* colorToGrey(unsigned char *imgIn_h, int width, int height, int channels) {
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
    colorToGrayKernel<<<grid, block>>>(imgOut_d, imgIn_d, width, height);

    CUDA_CHECK(cudaMemcpy(imgOut_h, imgOut_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(imgIn_d));
    CUDA_CHECK(cudaFree(imgOut_d));

    return imgOut_h;
}

unsigned char* blur(unsigned char *imgIn_h, int width, int height, int channels, int blur_size) {
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
    blurKernel<<<grid, block>>>(imgOut_d, imgIn_d, width, height, blur_size);

    CUDA_CHECK(cudaMemcpy(imgOut_h, imgOut_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(imgIn_d));
    CUDA_CHECK(cudaFree(imgOut_d));

    return imgOut_h;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_image> [output_image]\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = (argc > 2) ? argv[2] : "output.png";

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 0);
    img = colorToGrey(img, width, height, channels);
    img = blur(img, width, height, channels, 3);

    if (stbi_write_png(output_path, width, height, 1, img, width))
        printf("Saved output.png!\n");
    else
        printf("Failed to save image\n");

    free(img);

    return 0;
}