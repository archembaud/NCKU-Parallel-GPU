#include <stdio.h>
#include <stdlib.h>

// Device Functions

__global__ void GPU_Vector_Times_Constant(float *a, float C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		a[i] = C*a[i];
	}
}

// Host Functions

void Allocate_Memory(float **h_a, float **h_b, float **d_a, int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Host memory
    *h_a = (float*)malloc(size); 
    *h_b = (float*)malloc(size); 
    // Device memory
    Error = cudaMalloc((void**)d_a, size); 
    printf("CUDA error (malloc d_a) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_a, float **h_b, float **d_a) {
    if (*h_a) free(*h_a);
    if (*h_b) free(*h_b);
    if (*d_a) cudaFree(*d_a);
}

void Send_To_Device(float **h_a, float **d_a, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;

    // Send A to the GPU
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_a -> h_b) = %s\n", cudaGetErrorString(Error));
}

void Vector_Times_Constant(float **d_a, float C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GPU_Vector_Times_Constant<<<blocksPerGrid, threadsPerBlock>>>(*d_a, C, N);
}
