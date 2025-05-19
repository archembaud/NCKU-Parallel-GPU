#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
// Device Functions
__device__ float Compute_Heat_Flux(float Temp, float cx, float cy, float time) {
    // Depending on the time, we either have:
    // i) Heating happening due to the presence of a hot brake pad, or
    // ii) Cooling happening due to convective cooling
    // Here we have a convective heat flux owing to natural convection
    // Compute the convective heat flux on the device in a device function
    // Calculate angular position of rotating brake
    // Rotational speed (rad/s) = 10.472 
    float brake_angle = 10.472*time;
    // Compute the central location of the pad
    float CX = 0.08*cos(brake_angle);
    float CY = 0.08*sin(brake_angle);
    // Check if this cell is inside
    float RADIUS = sqrtf( (cx-CX)*(cx-CX) + (cy-CY)*(cy-CY));
    if (RADIUS < 0.005) {
        return 4.3238; // This is how much heat each cell of the pad will add.
    } else {
        float radius = sqrtf( cx*cx + cy*cy);
        float Re = AIR_RHO*2.0*3.14159*radius*(10.472*radius)/AIR_VIS;
        // Compute the convective heat transfer coefficient h and then heat flux using
        // Newton's Law of Cooling.
        float h = (0.664*sqrtf(Re)*0.8879*AIR_K)/(2.0*3.14159*radius);
        // Make this artificially larger (because its cool)
        return -h*DX*DY*(Temp - 300.0)*100.0;
    }
}

// Modified from the GPU_Compute_New_Temp function that was created for the 3D heat sink calculation
__global__ void GPU_Compute_New_Temp(float *d_T, float *d_Tnew, float *d_Body, float time) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        if (d_Body[index] == 1.0f) {
            // Center, Left, Right (X), Forward, Backward (Y) Temperatures
            float TC, TL, TR, TF, TB;
            TC = d_T[index]; // Temperature of this cell
		    int xcell = (int)(index/NY);
		    int ycell = (int)(index-xcell*NY);
            float cx = (xcell+0.5)*DX - 0.5*L;
            float cy = (ycell+0.5)*DY - 0.5*W;
            float Heat_Flux = Compute_Heat_Flux(d_T[index], cx, cy, time);
            /*
            Now we need to check for boundaries
            If we have a convective surface, we compute the heat transfer via conduction as 0 
            (by using an insulated condition where both T values are identical) and then adding
            the heat loss across that surface to the cumulative sum for that cell.
            */
            // Check left
            if (xcell == 0) {
                TL = TC; // No conductive heat transfer; all convective.
            } else {
                // Check if the cell to the left is air or a solid
                if (d_Body[index-NY] == 0) {
                    TL = TC;
                } else {
                    // Heat transfer is via conduction
                    TL = d_T[index - NY];
                }
            }
            // Right.
            if (xcell == (NX-1)) {
                TR = TC;
            } else {
                // Check if the cell to the right is air or a solid
                if (d_Body[index+NY] == 0) {
                    TR = TC;
                } else {
                    TR = d_T[index + NY];
                }
            }

            // Backward
            if (ycell == 0) {
                TB = TC;
            } else {
                if (d_Body[index-1] == 0) {
                    TB = TC;
                } else {
                    TB = d_T[index - 1];
                }
            }
            // Forward
            if (ycell == (NY-1)) {
                TF = TC;
            } else {
                if (d_Body[index+1] == 0) {
                    TF = TC;
                } else {
                    TF = d_T[index +1];
                }
            }
            // Update T_new from the X, Y and Z directions respectively
            d_Tnew[index] = d_T[index] + ALPHA_X*(TL + TR - 2.0*TC) + ALPHA_Y*(TB + TF - 2.0*TC);
            // Add the Heat Loss (or gain, depending)
            d_Tnew[index] = d_Tnew[index] + Heat_Flux*(DT/(DX*DY*DZ*RHO*CP)); 
        } else {
            // This is an air cell
            // (ignored for this demo)
            d_Tnew[index] = d_T[index];
        }
    }
}

// Host Functions
void Allocate_Memory(float **h_T, float **h_Body, float **d_T, float **d_Tnew, float **d_Body) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Host memory
    *h_T = (float*)malloc(size); 
    *h_Body = (float*)malloc(size); 
    // Device memory
    Error = cudaMalloc((void**)d_T, size); 
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Tnew, size); 
    printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Body, size); 
    printf("CUDA error (malloc d_Body) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_T, float **h_Body, float **d_T,  float **d_Tnew, float **d_Body) {
    free(*h_T);
    free(*h_Body);
    cudaFree(*d_T);
    cudaFree(*d_Tnew);
    cudaFree(*d_Body);
}

void Send_To_Device(float **h_a, float **d_a) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Send A to the GPU
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
}

// Each time step we need to update the temperature with the new temperature
// This function does this with a cudaMemcpy (Device to Device)
void Copy_on_Device(float **d_source, float **d_destination, int step) {
    size_t size = N*sizeof(float);
    cudaMemcpy(*d_destination, *d_source, size, cudaMemcpyDeviceToDevice); 
}

void Get_From_Device(float **d_a, float **h_b) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_a -> h_b) = %s\n", cudaGetErrorString(Error));
}

// Wrapper function to call the GPU work from the CPU
void Compute_T_new(float *d_T, float *d_Tnew, float *d_Body, int time_step) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float time = time_step*DT;
    GPU_Compute_New_Temp<<<blocksPerGrid, threadsPerBlock>>>(d_T, d_Tnew, d_Body, time);
}