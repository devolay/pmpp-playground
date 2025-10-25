#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "timer.h"
#include "utils.h"

static void vec_add(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    size_t n = 1<<24; // ~16M
    size_t bytes = n * sizeof(float);

    float* A = generateRandomVector(n);
    float* B = generateRandomVector(n);
    float* C = (float*)malloc(bytes);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\\n"); return 1; }

    pmpp_timer_t t; pmpp_timer_start(&t);
    vec_add(A,B,C,n);
    pmpp_timer_stop(&t);

    int ok = 1;
    for (size_t i = 0; i < 10; ++i) {
        if (C[i] != A[i]+B[i]) { ok = 0; break; }
    }

    printf("[CPU] n=%zu, time=%.3f ms, ok=%s\\n", n, pmpp_timer_ms(&t), ok ? "yes" : "no");

    free(A); free(B); free(C);
    return ok ? 0 : 1;
}