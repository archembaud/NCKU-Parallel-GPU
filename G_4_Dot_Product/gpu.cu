#include <stdio.h>
#include <stdlib.h>

#define MAX_BLOCK_SZ 256   	// Maximum number of threads per block

// Device Functions

__global__ void GPU_Partial_Dot_Product(float *x, float *y, float *z, const int n) {
    __shared__ float temp[MAX_BLOCK_SZ];	
    int i = blockDim.x*blockIdx.x + threadIdx.x;	// Global index
    int I = threadIdx.x;							// Local thread index
    // First, we work out the multiplication phase of the dot product
    // We will store the result in our "temp" variable in shared memory
    if (i < n) temp[I] = x[i]*y[i];	

    // Now, we need to syncronize the threads.
    __syncthreads();

    // Now, we can sum these within this block
    // We will use a tree structure to perform these additions
    for (int stride = blockDim.x/2; stride > 0; stride = stride/2) {
        if (I < stride) {
            temp[I] += temp[I + stride];
        }
        // Force another thread syncronization
        __syncthreads();
    }

    // Store the result in z
    // Each block will store one number
    if (I == 0) z[blockIdx.x] = temp[0];
}

// Host Functions

void Allocate_Memory(float **h_a, float **d_a, const char *variable, const int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Host memory
    *h_a = (float*)malloc(size);
    // Device memory
    Error = cudaMalloc((void**)d_a, size); 
    printf("CUDA error (malloc %s) = %s\n", variable, cudaGetErrorString(Error));
}

void Free_Memory(float **h_a, float **d_a) {
    if (*h_a) free(*h_a);
    if (*d_a) cudaFree(*d_a);
}

void Send_To_Device(float **h_a, float **d_a, const char *variable, const int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send A to the GPU
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy host -> %s) = %s\n", variable, cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, const char *variable, const int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy %s -> host) = %s\n", variable, cudaGetErrorString(Error));
}

void Partial_Dot_Product(float **d_a, float **d_b, float **d_c, const int N) {
    int threadsPerBlock = MAX_BLOCK_SZ;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GPU_Partial_Dot_Product<<<blocksPerGrid, MAX_BLOCK_SZ>>>(*d_a, *d_b, *d_c, N);
}
