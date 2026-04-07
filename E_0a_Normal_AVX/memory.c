#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int Allocate_Aligned_Memory(float **a, float **b, float **c, int N) {
    // Allocate using alligned memory; return an error
    int error;
    size_t alignment = 32;
    error = posix_memalign((void**)a, alignment, N*sizeof(float));
    if (error != 0) {
        printf("Aligned memory allocation failed (a) with error code %d\n", error);
        return 1;
    }
    error = posix_memalign((void**)b, alignment, N*sizeof(float));
    if (error != 0) {
        printf("Aligned memory allocation failed (b) with error code %d\n", error);
        return 1;
    }
    error = posix_memalign((void**)c, alignment, N*sizeof(float));
    if (error != 0) {
        printf("Aligned memory allocation failed (c) with error code %d\n", error);
        return 1;
    }
    return 0;
}

void Free_Memory(float **a, float **b, float **c) {
    // Free memory for a, b and c
    free(*a); free(*b); free(*c);
}