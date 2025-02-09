#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
Note - this example neglects the arguments provided with main;
Please see the example in 1_Hello_World for their application.
*/
int main(int argc, char *argv[]) {
    // Create some arrays on the heap
    const int N = 30;
    float *a, *b, *c;
    int i;

    // Allocate memory for a, b and c
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));

    // Let's return an error if we can't allocate a, b or c
    if ((a == NULL) || (b == NULL) || (c == NULL)) {
        printf("Unable to allocate memory for a, b or c. Aborting\n");
        if (a) free(a); if (b) free(b); if (c) free(c);
        // Return 1 as we have an error
        return 1;
    }

    // Set the value of a and b for each element based on position
    for (i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)2*i;
    }
    for (i = 0; i < N; i++) {
        c[i] = sqrtf(a[i] + b[i]);
        printf("Sqrt of (%g+%g) = %e\n", a[i], b[i], c[i]);
    }

    // Memory allocated on the heap must be freed.
    free(a); free(b); free(c);
    return 0;
}