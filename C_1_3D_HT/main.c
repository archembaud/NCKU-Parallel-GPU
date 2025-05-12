#include <stdio.h>
#include "gpu.h"


// Set the problem up
void Init(float *h_T, float *h_Body) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int index = k + j*NZ + i*NY*NZ; // Compute 1D index from 3D for loop pattern
                float cx = (i+0.5)*DX;
                float cy = (j+0.5)*DY;
                float cz = (k+0.5)*DZ;
                // Assume this is a not a body
                h_Body[index] = 0.0;
                h_T[index] = 0.0;
                if (cz < 0.33*H) {
                    // This is the base of the heat sink
                    h_Body[index] = 1.0;
                    h_T[index] = 400.0; // Make this 400K
                } else {
                    // There may be a fin in this region. Check for each fin.
                    // Move our fins a little from the edge (no real engineering reason for this).
                    if ((cx > 0.05*L) && (cx < 0.95*L) && (cy > 0.05*W) && (cy < 0.95*W)) {
                        // Iterate over the fins to check if we are in one
                        for (int fin = 0; fin < 15; fin++) {
                            float fin_start = (0.05*L + fin*2.0*FIN_W);
                            float fin_end = fin_start + FIN_W;
                            if ((cx > fin_start) && (cx < fin_end)) {
                                // This means we are in a fin; set the body and T accordingly.
                                h_Body[index] = 1.0;
                                h_T[index] = 400.0;
                            }
                        }
                    }
                }
            }
        }
    }
}


// Save the result
void Save_Result(float *h_Body, float *h_T, const char *Filename) {
    // Open the file
    FILE *fptr;
    fptr = fopen(Filename, "w");
    fprintf(fptr, "x coord, y coord, z coord, body, temperature\n");
    for (int index = 0; index < N; index++) {
        // Change our 1D cell index into (xcell,ycell,zcell) for convenience in saving
		int xcell = (int)(index/(NY*NZ));
		int ycell = (int)((index-xcell*NY*NZ)/NZ);
		int zcell = index - xcell*NY*NZ - ycell*NZ;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        float cz = (zcell+0.5)*DZ;
        fprintf(fptr, "%g, %g, %g, %g, %g\n", cx, cy, cz, h_Body[index], h_T[index]);
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
    printf("Running calculation with CFLs = %g, %g, %g for total time = %g\n", ALPHA_X, ALPHA_Y, ALPHA_Z, NO_STEPS*DT);

    // Perform unsteady time stepping
    for (int step = 0; step < NO_STEPS; step++) {
        // Compute the updated Temperature after a single step on the GPU
        Compute_T_new(d_T, d_Tnew, d_Body);
        // Update T on the GPU
        Copy_on_Device(&d_Tnew, &d_T, step);
    }

    // Copy d_Tnew from the device into h_T on the host
    Get_From_Device(&d_T, &h_T);
    // Save data
    Save_Result(h_Body, h_T, "Results.csv");
    // Free memory on GPU device and host
    Free_Memory(&h_T, &h_Body, &d_T, &d_Tnew, &d_Body);

    return 0;
}