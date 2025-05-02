#include <stdio.h>
#include <stdlib.h>
void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                     float **d_T, float **d_Tnew, int **d_Body, int N) {
    cudaError_t Error;
    // Host memory
    *h_T = (float*)malloc(N*sizeof(float));
    *h_Tnew = (float*)malloc(N*sizeof(float)); 
    *h_Body = (int*)malloc(N*sizeof(int)); 
    // Device memory
    Error = cudaMalloc((void**)d_T, N*sizeof(float));
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Tnew, N*sizeof(float));
    printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Body, N*sizeof(int));
    printf("CUDA error (malloc d_Body) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                 float **d_T, float **d_Tnew, int **d_Body) {
    if (*h_T) free(*h_T);
    if (*h_Tnew) free(*h_Tnew);
    if (*h_Body) free(*h_Body);
    if (*d_T) cudaFree(*d_T);
    if (*d_Tnew) cudaFree(*d_Tnew);
    if (*d_Body) cudaFree(*d_Body);
}

void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N) {
    // Grab an error type
    cudaError_t Error;
    // Send T to the GPU
    Error = cudaMemcpy(*d_T, *h_T, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_T -> d_T) = %s\n", cudaGetErrorString(Error));
    // Send Body to the GPU
    Error = cudaMemcpy(*d_Body, *h_Body, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_Body -> d_Body) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_T, float **h_T, int N) {
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_T, *d_T, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_T -> h_T) = %s\n", cudaGetErrorString(Error));
}