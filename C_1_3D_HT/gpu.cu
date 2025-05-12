#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
// Device Functions
__device__ float Compute_Heat_Flux(float Temp, float Area, float height) {
    // Here we have a convective heat flux owing to natural convection
    // Compute the convective heat flux on the device in a device function
    // The air temp is fixed.
    float Beta = 1.0/Temp;      // Temperature must be in K; true for an ideal gas only.
    // Compute the dimensionless Grashof number based on the temp in that cell
    float Gr = height*height*height*AIR_RHO*AIR_RHO*G*Beta*(Temp-300.0)/(AIR_VIS*AIR_VIS);
    // Compute the Nusselt number (assuming laminar flow)
    float Nu = __powf(Gr*0.25, 0.25)*GPR; // Note the approximate power function!
    // Compute the convective heat transfer coefficient h and then heat flux using
    // Newton's Law of Cooling.
    float h = Nu*AIR_K/height; 
    return h*Area*(Temp - 300.0);
}

// Modify the GPU_Vector_Times_Constant function to be our 3D heat transfer function
__global__ void GPU_Compute_New_Temp(float *d_T, float *d_Tnew, float *d_Body) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        if (d_Body[index] == 1.0f) {
            // Center, Up, Down, Left, Right, Forward, Backward Temperatures
            float TC, TU, TD, TL, TR, TF, TB;
            float Heat_Loss = 0.0; // We will cumulatively add to this as some cells have multiple convective surfaces.
            TC = d_T[index]; // Temperature of this cell
            int xcell = (int)(index/(NY*NZ));
            int ycell = (int)((index-xcell*NY*NZ)/NZ);
            int zcell = index - xcell*NY*NZ - ycell*NZ;

            /*
            Now we need to check for boundaries
            If we have a convective surface, we compute the heat transfer via conduction as 0 
            (by using an insulated condition where both T values are identical) and then adding
            the heat loss across that surface to the cumulative sum for that cell.
            */
            // Check left
            if (xcell == 0) {
                // Convection around edges of domain
                Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DY*DZ, (zcell+0.5)*DZ);
                TL = TC; // No conductive heat transfer; all convective.
            } else {
                // Check if the cell to the left is air or a solid
                if (d_Body[index-NY*NZ] == 0) {
                    // Convective heat transfer only
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DY*DZ, (zcell+0.5)*DZ);
                    TL = TC;
                } else {
                    // Heat transfer is via conduction
                    TL = d_T[index - NY*NZ];
                }
            }
            // Right.
            if (xcell == (NX-1)) {
                Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DY*DZ, (zcell+0.5)*DZ);
                TR = TC;
            } else {
                // Check if the cell to the right is air or a solid
                if (d_Body[index+NY*NZ] == 0) {
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DY*DZ, (zcell+0.5)*DZ);
                    TR = TC;
                } else {
                    TR = d_T[index + NY*NZ];
                }
            }

            // Down
            if (zcell == 0) {
                TD = 400.0; // Constant temperature heat source - this is the CPU our heat sink sits on.
            } else {
                if (d_Body[index-1] == 0) {
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DY, (zcell+0.5)*DZ);
                    TD = TC;
                } else {
                    TD = d_T[index - 1];
                }
            }
            // Up
            if (zcell == (NZ-1)) {
                Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DY, (zcell+0.5)*DZ);
                TU = TC;
            } else {
                if (d_Body[index+1] == 0) {
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DY, (zcell+0.5)*DZ);
                    TU = TC;
                } else {
                    TU = d_T[index + 1];
                }
            }
            
            // Backward
            if (ycell == 0) {
                Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DZ, (zcell+0.5)*DZ);
                TB = TC;
            } else {
                if (d_Body[index-NZ] == 0) {
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DY, (zcell+0.5)*DZ);
                    TB = TC;
                } else {
                    TB = d_T[index - NZ];
                }
            }
            // Forward
            if (ycell == (NY-1)) {
                Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DZ, (zcell+0.5)*DZ);
                TF = TC;
            } else {
                if (d_Body[index+NZ] == 0) {
                    Heat_Loss = Heat_Loss + Compute_Heat_Flux(TC, DX*DY, (zcell+0.5)*DZ);
                    TF = TC;
                } else {
                    TF = d_T[index + NZ];
                }
            }
            // Update T_new from the X, Y and Z directions respectively
            d_Tnew[index] = d_T[index] + ALPHA_X*(TL + TR - 2.0*TC)
                           + ALPHA_Y*(TB + TF - 2.0*TC) + ALPHA_Z*(TU + TD - 2.0*TC);
            // Update T due to lost heat - don't forget the source (heat loss) is volumetric (divide by cell volume)
            // as well as RHO and CP (as we are solving for dT/dt directly).
            d_Tnew[index] = d_Tnew[index] - Heat_Loss*(DT/(DX*DY*DZ*RHO*CP)); 
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
void Compute_T_new(float *d_T, float *d_Tnew, float *d_Body) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GPU_Compute_New_Temp<<<blocksPerGrid, threadsPerBlock>>>(d_T, d_Tnew, d_Body);
}