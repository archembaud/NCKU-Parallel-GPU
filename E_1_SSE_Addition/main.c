#include <stdio.h>
#include <malloc.h>
#include <xmmintrin.h>
#include "memory.h"

int main() {
    int N = 64;
    float *a, *b, *c;				  // Normal floats in memory
    __m128 *SSE_a, *SSE_b, *SSE_c;    // Pointers for SSE registers 
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
    SSE_a = (__m128*)a; SSE_b = (__m128*)b; SSE_c = (__m128*)c;     

    // Compute c = a + b using SSE
    // Since each SSE element performs operations on 4x floats,
    // our for loop is 4x smaller.
    for (i = 0; i < (N/4); i++) {
        SSE_c[i] = _mm_add_ps(SSE_a[i] ,SSE_b[i] );
    }

    // Check our solution
    for (i = 0; i < N; i++) {
    printf("Result ==> c[%d] = %f\n", i, c[i]);
    }

    // Free the memory
    Free_Memory(&a, &b, &c);
    return 0;
}
