#include <stdio.h>
#include <malloc.h>
#include <immintrin.h>
#include "memory.h"

int main() {
    int N = 64;
    float *a, *b, *c;				  // Normal floats in memory
    __m256 *AVX_a, *AVX_b, *AVX_c;    // Pointers for SSE registers 
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

    // Tell our SSE registers where to look for information
    AVX_a = (__m256*)a; AVX_b = (__m256*)b; AVX_c = (__m256*)c;     

    // Compute c = a + b using AVX
    // Since each AVX element performs operations on 8x floats,
    // our for loop is 8x smaller.
    for (i = 0; i < (N/8); i++) {
        AVX_c[i] = _mm256_add_ps(AVX_a[i] ,AVX_b[i] );
    }

    // Check our solution
    for (i = 0; i < N; i++) {
    printf("Result ==> c[%d] = %f\n", i, c[i]);
    }

    // Free the memory
    Free_Memory(&a, &b, &c);
    return 0;
}
