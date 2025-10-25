#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <file.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

float*  generateRandomVector(size_t n);
float** generateRandomSquareMatrix(size_t n);
void  freeSquareMatrix(int** matrix, size_t n);

#ifdef __cplusplus
}
#endif

#endif