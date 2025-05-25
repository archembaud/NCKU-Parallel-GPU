#include <stdio.h>
#include <stdlib.h>

#define k_steel 165e-6

void Set_GPU_Device(int device) {
    cudaSetDevice(device);
}

__global__  void Compute_New_Temperature_GPU(float *d_T, float *d_Tnew, int NX, int NY, float DX, float DY, float DT) {

    // Goal is to calculate h_Tnew
    int xcell = blockDim.x * blockIdx.x + threadIdx.x;
    int ycell = blockDim.y * blockIdx.y + threadIdx.y;
    // We still need to compute i, the cell index
    // But we know how to do this!
    int i = xcell*NY + ycell;

    // Since we are using 2D threads and 2D blocks, it's possible that we have overlap
    // in both x and y - so I need to check both xcell and ycell
    if ((xcell < NX) && (ycell < NY)) {

        float TC = d_T[i];
        float TL, TR, TD, TU;

        // Check right
        if (xcell == (NX-1)) {
            // We are on the right edge, its damn hot here.
            TR = 1000.0;
        } else {
            // We can check the cell to the right
            TR = d_T[i+NY];
        }
        // Check left
        if (xcell == 0) {
            // We are on the left edge, its cool here.
            TL = 300.0;
        } else {
            // We can check the cell to the right
            TL = d_T[i-NY];
        }
        // Vertical direction now
        // Check up (U)
        if (ycell == (NY-1)) {
            // We are on the top edge, no heat flow.
            TU = d_T[i];
        } else {
            // We can check the cell above
            TU = d_T[i+1];
        }
        // Check down (D)
        if (ycell == 0) {
            // We are on the bottom edge, set to 300
            TD = 300.0;
        } else {
            // We can check the cell to bottom
            TD = d_T[i-1];
        }
        // Update T based on contributions from each direction (X and Y)
        d_Tnew[i] = d_T[i] + k_steel*(DT/(DX*DX))*(TR + TL - 2.0*TC);
        d_Tnew[i] = d_Tnew[i] + k_steel*(DT/(DY*DY))*(TU + TD - 2.0*TC);
    }
}


void Compute_GPU(float *d_T, float *d_Tnew, int NX, int NY, float DX, float DY, float DT) {
    int no_threads_x = 16;   // No. threads in the X direction
	int no_threads_y = 16;   // No. threads in the Y direction
    // Compute the number of blocks required in each direction
    int no_blocks_x = (int)((NX + no_threads_x - 1)/no_threads_x);
    int no_blocks_y = (int)((NY + no_threads_y - 1)/no_threads_y);
	dim3 threads(no_threads_x, no_threads_y, 1); // 2D Threads in each block (ignore Z for now)
	dim3 grid(no_blocks_x, no_blocks_y);
    Compute_New_Temperature_GPU<<<grid,threads>>>(d_T, d_Tnew, NX, NY, DX, DY, DT);
}

void Allocate_Memory(float **h_T, float **h_Tnew,  
                     float **d_T, float **d_Tnew, int N) {
    cudaError_t Error;
    // Host memory
    *h_T = (float*)malloc(N*sizeof(float));
    *h_Tnew = (float*)malloc(N*sizeof(float)); 
    // Device memory
    Error = cudaMalloc((void**)d_T, N*sizeof(float));
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Tnew, N*sizeof(float));
    printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_T, float **h_Tnew, 
                 float **d_T, float **d_Tnew) {
    if (*h_T) free(*h_T);
    if (*h_Tnew) free(*h_Tnew);
    if (*d_T) cudaFree(*d_T);
    if (*d_Tnew) cudaFree(*d_Tnew);
}

void Send_To_Device(float **h_T, float **d_T, int N) {
    // Grab an error type
    cudaError_t Error;
    // Send T to the GPU
    Error = cudaMemcpy(*d_T, *h_T, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_T -> d_T) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_T, float **h_T, int N) {
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_T, *d_T, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_T -> h_T) = %s\n", cudaGetErrorString(Error));
}

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N) {
    cudaError_t Error;
    // Send d_Tnew into d_T
    Error = cudaMemcpy(*d_T, *d_Tnew, N*sizeof(float), cudaMemcpyDeviceToDevice);
}