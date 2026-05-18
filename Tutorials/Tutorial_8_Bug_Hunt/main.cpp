// Simple 1D Heat Transfer code
// a = Temperature, b = New temperature
// but..there is a bug here.
#include <stdio.h>
#include "gpu.h"

int main(int argc, char *argv[]) {
    float *h_a, *h_b;  
    float *d_a, *d_b;
    int N = 20;
    float DX = 1.0/N;
    float DT = 0.001;
    float ALPHA = 1.0;
    int i;

    float CFL = DT*ALPHA/(DX*DX);
    printf("CFL = %g\n", CFL);
 
    // Allocate memory on both device and host
    Allocate_Memory(&h_a, &h_b, &d_a, &d_b, N);

    // Initialise h_a, but not h_b
    for (i = 0; i < N; i++) {
        h_a[i] = 0.0;
    }

    // Take h_a and store it on the device in d_a
    Send_To_Device(&h_a, &d_a, N);

    for (int step = 0; step < 100; step++) {
    	// Perform a computation - multiply d_a by a constant (2)
    	Compute_New_Temperature(d_a, d_b, CFL, N);
   	// This function copies
    	Device_To_Device(&d_a, &d_b, N);
    }

    // Copy d_a from the device into h_b on the host
    Get_From_Device(&d_b, &h_b, N);

    // Check the values of h_b; should be the same as h_a
    for (i = 0; i < N; i++) {
        printf("Value of h_b[%d] = %g\n", i, h_b[i]);
    }

    // Free memory
    Free_Memory(&h_a, &h_b, &d_a, &d_b);

    return 0;
}
