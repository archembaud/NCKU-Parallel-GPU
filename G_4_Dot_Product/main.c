#include <stdio.h>
#include "gpu.h"

#define MAX_BLOCK_SZ 256   	// Maximum number of threads per block

int main(int argc, char *argv[]) {

    float *h_x;	// Host variables
    float *h_y;
    float *h_z;
    float *d_x;	// Device variables
    float *d_y;
    float *d_z;
    float dot_sum = 0.0;
    const int N = 1000;	
    int blocksPerGrid = (N + MAX_BLOCK_SZ - 1) / MAX_BLOCK_SZ;		// Number of blocks
    int i;

    // Allocate memory on both device and host for x, y and z
    Allocate_Memory(&h_x, &d_x, "d_x", N);
    Allocate_Memory(&h_y, &d_y, "d_y", N);
    Allocate_Memory(&h_z, &d_z, "d_z", blocksPerGrid);

    // Initialise x and y; compute the dot product now to check the result
    for (i = 0; i < N; i++) {
        h_x[i] = 1.0;
        h_y[i] = 1.0;
        dot_sum += h_x[i]*h_y[i];
    }

    printf("Serial dot product = %g\n", dot_sum);
    dot_sum = 0.0; // Reset for GPU test

    // Send x and y to the device
    Send_To_Device(&h_x, &d_x, "d_x", N);
    Send_To_Device(&h_y, &d_y, "d_y", N);

    // Perform a computation - Partial Dot Product
    Partial_Dot_Product(&d_x, &d_y, &d_z, N);

    // Copy d_z from the device into h_z on the host
    Get_From_Device(&d_z, &h_z, "d_z", blocksPerGrid);

    // Check the values of h_z; sum these to get the dot product
    for (i = 0; i < blocksPerGrid; i++) {
        printf("Value of h_z[%d] = %g\n", i, h_z[i]);
        dot_sum += h_z[i];
    }

    printf("GPU Dot Product of h_x and h_y = %g\n", dot_sum);

    // Free memory
    Free_Memory(&h_x, &d_x);
    Free_Memory(&h_y, &d_y);
    Free_Memory(&h_z, &d_z);

    return 0;
}