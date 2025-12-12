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

    int NO_STEPS = 10;
    float DT = 0.001;
    float g = 9.81;

    float L = 1.0;
    float DX = L/(float)N;

    int i;

    // Allocate memory for a, b and c
    U0 = (float*)malloc(N*sizeof(float));
    U1 = (float*)malloc(N*sizeof(float));

    P0 = (float*)malloc(N*sizeof(float));
    P1 = (float*)malloc(N*sizeof(float));

    F0 = (float*)malloc((N+1)*sizeof(float));
    F1 = (float*)malloc((N+1)*sizeof(float));



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

    for (int step = 0; step < NO_STEPS; step++) {

        printf("Step %d\n", step);
        // Compute fluxes
        for (i = 0; i <= N; i++) {
            if (i == 0) {
                // This is the left most boundary, and is special
                // This approach is very wrong, but will do for now.
                F0[i] = P0[i]*P1[i];
                F1[i] = P0[i]*P1[i]*P1[i] + 0.5*g*P0[i]*P0[i];
            } else {
                // Consider this an ordinary upwind cell, assuming the speed is positive
                F0[i] = P0[i]*P1[i];
                F1[i] = P0[i]*P1[i]*P1[i] + 0.5*g*P0[i]*P0[i];
            }
        }

        // Update conservative variables
        for (i = 0; i < N; i++) {
            U0[i] = U0[i] - (DT/DX)*(F0[i+1] - F0[i]);
            U1[i] = U1[i] - (DT/DX)*(F1[i+1] - F1[i]);
        }

        // Update primitive variables
        for (i = 0; i < N; i++) {
            P0[i] = U0[i];
            P1[i] = U1[i]/U0[i];
        }
    }


    // Memory allocated on the heap must be freed.
    free(U0); free(U1);
    free(P0); free(P1);
    free(F0); free(F1);
    return 0;
}