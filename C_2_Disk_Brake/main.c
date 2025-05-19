#include <stdio.h>
#include "gpu.h"


// Set the problem up
void Init(float *h_T, float *h_Body) {
    FILE *fptr;
    int read_body;
    fptr = fopen("Brake.csv", "r");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int index = j + NY*i;
            float cx = (i+0.5)*DX;
            float cy = (j+0.5)*DY;
            // Assume this is a not a body
            h_Body[index] = 0.0;
            h_T[index] = 300.0;
            // Now I need to decide if this cell is a body or not.
            if (j == (NY-1)) {
                fscanf(fptr, "%d\n", &read_body);
            } else {
                fscanf(fptr, "%d,", &read_body);
            }
            if (read_body < 200) {              
                h_Body[index] = 1.0;
            }
        }
    }
    fclose(fptr);
}


// Save the result
void Save_Result(float *h_Body, float *h_T, const char *Filename) {
    // Open the file
    FILE *fptr;
    fptr = fopen(Filename, "w");
    for (int index = 0; index < N; index++) {
        // Change our 1D cell index into (xcell,ycell,zcell) for convenience in saving
		int xcell = (int)(index/NY);
		int ycell = (int)(index-xcell*NY);
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        fprintf(fptr, "%g\t%g\t%g\t%g\n", cx, cy, h_Body[index], h_T[index]);
    }
    // CLose the file
    fclose(fptr);
}



int main(int argc, char *argv[]) {

    float *h_Body, *d_Body;  // Body (1 = Heat Fin, 0 = Air (not simulated))
    float *h_T, *d_T;   // Temperature (host and GPU)
    float *d_Tnew;      // New Temperature (only exists on the GPU!)
    int i;

    // Allocate memory on both device and host, then initialize the problem.
    Allocate_Memory(&h_T, &h_Body, &d_T, &d_Tnew, &d_Body);
    Init(h_T, h_Body);
    // Take h_T and store it on the device in d_T
    Send_To_Device(&h_T, &d_T);
    // Do the same for the body.
    Send_To_Device(&h_Body, &d_Body);
    // Show critical data to the user before running the calculation
    // All of the ALPHA values are actually subject to stability restrictions and should
    // be less than 0.5 (worst case) - currently, the worst of these is around 0.25.
    printf("Running calculation with CFLs = %g, %g for total time = %g with %d steps\n", ALPHA_X, ALPHA_Y, NO_STEPS*DT, NO_STEPS);

    // Perform unsteady time stepping
    for (int step = 0; step < NO_STEPS; step++) {
        // Compute the updated Temperature after a single step on the GPU
        Compute_T_new(d_T, d_Tnew, d_Body, step);
        // Update T on the GPU
        Copy_on_Device(&d_Tnew, &d_T, step);
    }

    // Copy d_Tnew from the device into h_T on the host
    Get_From_Device(&d_T, &h_T);
    // Save data
    Save_Result(h_Body, h_T, "Results.txt");
    // Free memory on GPU device and host
    Free_Memory(&h_T, &h_Body, &d_T, &d_Tnew, &d_Body);

    return 0;
}