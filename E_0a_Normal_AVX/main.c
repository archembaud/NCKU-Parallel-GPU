#include <stdio.h>
#include "memory.h"

void add_values_normal(float *a, float *b, float *c, int N) {
    // Compute c = a + b using normal code
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void add_values_simd(float * __restrict a,
                       float * __restrict b,
                       float * __restrict c,
                       int N) {
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}


int main() {
    int N = 64;
    float *a, *b, *c;				  // Normal floats in memory
    int i, error;
    size_t alignment = 32;

    // Allocate using alligned memory; return an error
    error = Allocate_Aligned_Memory(&a, &b, &c, N);
    if (error != 0) {
        // Failure during memory allocation; abort.
        return 0;
    }

    // Set up a and b
    for (i = 0; i < N; i++) {
        a[i] = (float)i;    b[i] = (float)2.0*i;
    }

    // Run one (or both) of the functions below.
    add_values_normal(a, b, c, N);
    add_values_simd(a, b, c, N);

    // Check our solution
    for (i = 0; i < N; i++) {
    printf("Result ==> c[%d] = %f\n", i, c[i]);
    }

    // Free the memory
    Free_Memory(&a, &b, &c);
    return 0;
}