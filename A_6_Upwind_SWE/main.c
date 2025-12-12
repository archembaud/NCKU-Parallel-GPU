#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
Note - this example neglects the arguments provided with main;
Please see the example in 1_Hello_World for their application.
*/
int main(int argc, char *argv[]) {
    // Create some arrays on the heap
    const int N = 200;
    float *U0, *U1;
    float *P0, *P1;
    float *F0, *F1;

    int i;

    // Allocate memory for a, b and c
    U0 = (float*)malloc(N*sizeof(float));
    U1 = (float*)malloc(N*sizeof(float));

    P0 = (float*)malloc(N*sizeof(float));
    P1 = (float*)malloc(N*sizeof(float));

    F0 = (float*)malloc(N*sizeof(float));
    F1 = (float*)malloc(N*sizeof(float));

    // Set the value of a and b for each element based on position
    for (i = 0; i < N; i++) {
        // Set P and U first
        if (i < 0.5*N) {
            // Primitive variables
            P0[i] = 10.0f; P1[i] = 0.0;
            // Conservative variables
            U0[i] = P0[i]; U1[i] = P0[i]*P1[i];
        } else {
            // Primitive variables
            P0[i] = 1.0f; P1[i] = 0.0;
            // Conservative variables
            U0[i] = P0[i]; U1[i] = P0[i]*P1[i];
        }
    }

    // Memory allocated on the heap must be freed.
    free(U0); free(U1);
    free(P0); free(P1);
    free(F0); free(F1);
    return 0;
}