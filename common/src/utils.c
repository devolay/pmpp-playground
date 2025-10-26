#include "utils.h"
#include <stdio.h>
#include <stdlib.h>


float* generateRandomVector(size_t n) {
    float* vec = (float*)malloc(n * sizeof(int));
    if (!vec) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    for (size_t i = 0; i < n; ++i) vec[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    return vec;
}

float* generateRandomSquareMatrix(size_t n) {
    float* matrix = (float*)malloc(n * n * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    return matrix;
}
void freeSquareMatrix(int** matrix, size_t n) {
    if (!matrix) return;
    for (size_t i = 0; i < n; ++i) free(matrix[i]);
    free(matrix);
}
