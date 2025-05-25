#include <stdio.h>
#include "gpu.h"

void Save_Results(float *h_T, int NX, int NY, float DX, float DY) {
    FILE *fptr;
    const int N = NX*NY;
    fptr = fopen("results.txt", "w");
    for (int i = 0; i < N; i++) {
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        fprintf(fptr, "%g\t%g\t%g\n", cx, cy, h_T[i]);
    }
    fclose(fptr);
}

void Init(float *h_T, int N) {
    // Initialise T
    for (int i = 0; i < N; i++) {
        h_T[i] = 300.0; // Set the initial temperature everywhere to 300
    }
}


int main(int argc, char *argv[]) {
    float *h_T, *h_Tnew, *d_T, *d_Tnew;
    int NX = 100;
    int NY = 100;
    int N = NX*NY;
    float L = 1.0;   // Length of region
    float H = 0.5;   // Height of region
    float W = 0.25;  // Hole size
    float DX = (L/NX);
    float DY = (H/NY);
    float DT = 0.02;
    int NO_STEPS = 400000;

    Set_GPU_Device(0);

    // Allocate memory on both device and host
    Allocate_Memory(&h_T, &h_Tnew, &d_T, &d_Tnew, N);

    // Set the types of material and initial temperature
    Init(h_T, N);

    // Take h_T store on the device
    Send_To_Device(&h_T, &d_T, N);

    // Take time steps
    for (int step = 0; step < NO_STEPS; step++) {
        Compute_GPU(d_T, d_Tnew, NX, NY, DX, DY, DT);
        Update_Temperature_GPU(&d_T, &d_Tnew, N);
    }

    // Copy d_T from the device into h_T on the host
    Get_From_Device(&d_T, &h_T, N);

    // Save the result
    Save_Results(h_T, NX, NY, DX, DY);

    // Free memory
    Free_Memory(&h_T, &h_Tnew, &d_T, &d_Tnew);

    return 0;
}