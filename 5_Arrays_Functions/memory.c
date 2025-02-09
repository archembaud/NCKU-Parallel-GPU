/*
memory.c

Contains function definitions for memory management
*/

#include <stdio.h>
#include <stdlib.h>

int Allocate_Memory(float **a, float **b, float **c, int N) {
    // Allocate memory for a, b and c
    *a = (float*)malloc(N*sizeof(float));
    *b = (float*)malloc(N*sizeof(float));
    *c = (float*)malloc(N*sizeof(float));
    // Let's return an error if we can't allocate a, b or c
    if ((*a == NULL) || (*b == NULL) || (*c == NULL)) {
        printf("Unable to allocate memory for a, b or c. Aborting\n");
        if (*a) free(*a); if (*b) free(*b); if (*c) free(*c);
        // Return 1 as we have an error
        return 1;
    }
    return 0;
}


void Free_Memory(float **a, float **b, float **c) {
    // Free memory for a, b and c
    free(*a); free(*b); free(*c);
}
