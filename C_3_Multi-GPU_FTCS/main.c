#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "gpu.h"



int main(int argc, char *argv[]) {
    // Problem size
    int N = 100;
    int NO_STEPS = 200;
    // Create a global array
    float *h_temp_global = (float*)malloc(N*sizeof(float));
    // Create a tiny little array holding our swapped buffer regions
    float *h_swap = (float*)malloc(2*sizeof(float));

    // Initialise - 1D diffusion step with 1 and 0 initial condition
    for (int i = 0; i < N; i++) {
        if (i < 0.5*N) {
            h_temp_global[i] = 1.0;
        } else {
            h_temp_global[i] = 0.0;
        }
    }

    // Create parallel sections in OpenMP before time stepping
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        int M = (int)(0.5*N + 2);  // Number of cells, including buffer region, on each gpu

        // GPU_Set_Device(tid);

        // Allocate memory for each thread (1/2 of the domain in this case, plus 2 buffers)
        float *h_temp;
        float *d_temp, *d_temp_new;
        Allocate_Memory(&h_temp, &d_temp, &d_temp_new, M);

        // Copy the correct part of h_temp_global - not including buffer cells
        for (int i = 0; i < 0.5*N; i++) {
            // We only fill valid cells
            h_temp[i+1] = h_temp_global[tid*(N/2) + i]; // Could do this with a memcpy as well
        }

        // Fix our boundary conditions - including our outer-most buffer cells
        if (tid == 0) {
            h_temp[0] = 1.0;
            h_temp[M-1] = h_temp_global[(int)(0.5*N)];
        } else if (tid == 1) {
            h_temp[0] = h_temp_global[(int)((0.5*N)-1)];
            h_temp[M-1] = 0.0;
        }

        // Send to the GPU
        Send_To_Device(&h_temp, &d_temp, "Temperatures", M); // Send all cells, including buffer cells

        // Start time stepping
        for (int step = 0; step < NO_STEPS; step++) {
            // Compute new temperature (d_temp_new) using FTCS
            Compute_Tnew(d_temp, d_temp_new, 0.5*N);
            // Perform a memory copy on device to update d_temp based on d_temp_new
            Update_T(&d_temp, &d_temp_new, 0.5*N); 
            // Manage boundary conditions
            Collect_Boundaries(&d_temp, &h_swap, 0.5*N, tid);
            // Ensure all threads are up to speed before distributing bounds
            #pragma omp barrier
            Distribute_Boundaries(&d_temp, &h_swap, 0.5*N, tid);
        }

        // Copy this thread's result back to the host
        Get_From_Device(&d_temp, &h_temp, "Final temperatures", M);

        // Copy each h_temp into the correct position in the global temp
        memcpy(&h_temp_global[(int)(0.5*N*tid)], &h_temp[1], 0.5*N*sizeof(float));

        Free_Memory(&h_temp, &d_temp, &d_temp_new);
    }

    // Check the total result
    printf("-------- Final single result ----------\n");
    for (int i = 0; i < N; i++) {
        printf("h_temp_global[%d] = %g\n", i, h_temp_global[i]);
    }

    free(h_temp_global);
    free(h_swap);
    return 0;
}