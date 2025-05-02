#include <stdio.h>
#include "gpu.h"

#define a0 18.8e-6
#define a1 165e-6

void Save_Results(float *h_T, int *h_Body, int NX, int NY, float DX, float DY) {
    FILE *fptr;
    const int N = NX*NY;
    fptr = fopen("results.txt", "w");
    // Write data to a file using tab delimiting
    for (int i = 0; i < N; i++) {
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        fprintf(fptr, "%g\t%g\t%d\t%g\n", cx, cy, h_Body[i], h_T[i]);
    }
    // Close the file
    fclose(fptr);
}


void Compute_New_Temperature(float *h_T, float *h_Tnew, int *h_Body, int NX, int NY, float DX, float DY, float DT) {
    // Iterate over all cells and then compute the new T
    const int N = NX*NY;
    for (int i = 0; i < N; i++) {
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float TC, TL, TR, TU, TD;
        float aC, aL, aR, aU, aD;
        // Set TC (Central T) to h_T[i]
        TC = h_T[i];
        aC = (h_Body[i]==0)*a0 + (h_Body[i]==1)*a1; // One line function to compute alpha based on h_Body
        // Check Left cell
        if (xcell == 0) {
            TL = 300.0;
            aL = (h_Body[i]==0)*a0 + (h_Body[i]==1)*a1;
        } else {
            TL = h_T[i-NY];
            aL = (h_Body[i-NY]==0)*a0 + (h_Body[i-NY]==1)*a1;
        }
        // Check Right cell
        if (xcell == (NX-1)) {
            TR = 1000.0;
            aR = (h_Body[i]==0)*a0 + (h_Body[i]==1)*a1;
        } else {
            TR = h_T[i+NY];
            aR = (h_Body[i+NY]==0)*a0 + (h_Body[i+NY]==1)*a1;
        }
        // Check Bottom cell
        if (ycell == 0) {
            TD = h_T[i];
            aD = (h_Body[i]==0)*a0 + (h_Body[i]==1)*a1;
        } else {
            TD = h_T[i-1];
            aD = (h_Body[i-1]==0)*a0 + (h_Body[i-1]==1)*a1;
        }
        // Check Top cell
        if (xcell == (NX-1)) {
            TU = h_T[i];
            aU = (h_Body[i]==0)*a0 + (h_Body[i]==1)*a1;
        } else {
            TU = h_T[i+1];
            aU = (h_Body[i+1]==0)*a0 + (h_Body[i+1]==1)*a1;
        }

        // Ready - Compute contribution from Right
        h_Tnew[i] = h_T[i] + (2.0*(1.0/( (1.0/aC) + (1.0/aR)))*DT/(DX*DX))*(TR-TC);
        // Left
        h_Tnew[i] = h_Tnew[i] - (2.0*(1.0/( (1.0/aC) + (1.0/aL)))*DT/(DX*DX))*(TC-TL);
        // Top
        h_Tnew[i] = h_Tnew[i] + (2.0*(1.0/( (1.0/aC) + (1.0/aU)))*DT/(DY*DY))*(TU-TC);
        // Bottom
        h_Tnew[i] = h_Tnew[i] - (2.0*(1.0/( (1.0/aC) + (1.0/aD)))*DT/(DY*DY))*(TC-TD);        
    }
}



int main(int argc, char *argv[]) {

    float *h_T, *h_Tnew, *d_T, *d_Tnew;
    int *h_Body, *d_Body;
    int NX = 100;
    int NY = 100;
    int N = NX*NY;
    float L = 1.0;   // Length of region
    float H = 0.5;   // Height of region
    float W = 0.25;  // Hole size
    float DX = (L/NX);
    float DY = (H/NY);
    const float DT = 0.02;
    const int NO_STEPS = 400000; 

    // Allocate memory on both device and host
    Allocate_Memory(&h_T, &h_Tnew, &h_Body, &d_T, &d_Tnew, &d_Body, N);

    // Initialise T and Body
    for (int i = 0; i < N; i++) {
        // We can use the index to compute the x, y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;

        h_T[i] = 300.0; // Set the initial temperature everywhere to 300
    
        // Set the body 
        h_Body[i] = 0;
        // First hole
        if ( (cx > 0.125) && (cx < (0.125 + W)) && (cy > 0.05) && (cy < 0.05 + W)) {
            h_Body[i] = 1;
        }
        // Second hole
        if ( (cx > (L-0.125-W)) && (cx < (L-0.125)) && (cy > (H-0.05-W)) && (cy < (H-0.05))) {
            h_Body[i] = 1;
        }
    }

    // Take h_T and h_Body and store both on the device
    //Send_To_Device(&h_T, &d_T, &h_Body, &d_Body, N);

    // Set up a time loop
    for (int step = 0; step < NO_STEPS; step++) {
        printf("Computing step %d of %d\n", step, NO_STEPS);
        Compute_New_Temperature(h_T, h_Tnew, h_Body, NX, NY, DX, DY, DT);
        // Update the temp
        for (int i = 0; i < N; i++) {
            h_T[i] = h_Tnew[i];
        }
    }

    // Copy d_a from the device into h_b on the host
    // Get_From_Device(&d_T, &h_T, N);

    // Save the result
    Save_Results(h_T, h_Body, NX, NY, DX, DY);

    // Free memory
    Free_Memory(&h_T, &h_Tnew, &h_Body, &d_T, &d_Tnew, &d_Body);

    return 0;
}