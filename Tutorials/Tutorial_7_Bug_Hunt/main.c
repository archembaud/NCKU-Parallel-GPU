#include <stdio.h>
#include "gpu.h"

int main(int argc, char *argv[]) {

    float *h_a, *h_b;
    float *d_a;
    int N = 100;
    int i;

    // Allocate memory on both device and host
    Allocate_Memory(&h_a, &h_b, &d_a, N);

    // Initialise h_a, but not h_b
    for (i = 0; i < N; i++) {
        h_a[i] = (float)i;
    }

    // Take h_a and store it on the device in d_a
    Send_To_Device(&h_a, &d_a, N);

    // Copy d_a from the device into h_b on the host
    Get_From_Device(&d_a, &h_b, N);

    // Check the values of h_b; should be the same as h_a
    for (i = 0; i <= N; i++) {
        printf("Value of h_b[%d] = %g\n", i, h_b[i]);
    }    

    // Free memory
    Free_Memory(&h_a, &h_b, &d_a);

    return 0;
}