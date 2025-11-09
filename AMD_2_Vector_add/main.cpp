/*
   Simple Vector Add demonstration using ROCm with hipcc
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>

int N = 10000;
float *d_A, *d_B, *d_C; // Device (GPU) pointers
float *h_A, *h_B, *h_C; // Host (CPU)  pointers


// Check the error variable returned from a ROCm / HIP operation
void Check_Error(hipError_t error, const char* event_string) {
    if (error != hipSuccess) {
        printf("Error Detected (%s)->(%s).\n", event_string, hipGetErrorString(error));
    } else {
        printf("Success (%s)\n", event_string);
    }
}


// Allocate memory on the GPU and CPU
void Allocate_Memory() {
    hipError_t err; // HIP error type
    // Allocate host (CPU) memory
    h_A = (float*)malloc(N*sizeof(float));
    h_B = (float*)malloc(N*sizeof(float));
    h_C = (float*)malloc(N*sizeof(float));
    // Allocate device memory
    err = hipMalloc(&d_A, N * sizeof(float)); Check_Error(err, "d_A Allocation");
    err = hipMalloc(&d_B, N * sizeof(float)); Check_Error(err, "d_B Allocation");
    err = hipMalloc(&d_C, N * sizeof(float)); Check_Error(err, "d_C Allocation");
}


void Free_Memory() {
    hipError_t err;
    // Free host (CPU) memory
    free(h_A);
    free(h_B);
    free(h_C);
    // Free device memory
    err = hipFree(d_A); Check_Error(err, "d_A Free");
    err = hipFree(d_B); Check_Error(err, "d_B Free");
    err = hipFree(d_C); Check_Error(err, "d_C Free");
}


void Copy_To_Device() {
    hipError_t err;
    // Copy data from host to device
    err = hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    Check_Error(err, "h_A->d_A Copy");
    err = hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);
    Check_Error(err, "h_B->d_B Copy");
}


void Copy_From_Device() {
    hipError_t err;
    err = hipMemcpy(h_C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);
    Check_Error(err, "d_C->h_C Copy");
}


void Init() {
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)1.0*i;
        h_B[i] = (float)2.0*i;
    }
}


// Kernel function to be executed on the GPU
__global__ void Vector_Add(const float* A, const float* B, float* C, int n) {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = blockId * blockDim.x + threadId;
    if (globalId < n) {
        C[globalId] = A[globalId] + B[globalId];
    }
}


int main() {
    Allocate_Memory();
    Init();
    Copy_To_Device();
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(Vector_Add, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);
    // Fetch contents of C from device to host
    Copy_From_Device();
    // Show the contents of (some of) C
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    // Free memory on CPU and GPU
    Free_Memory();
    return 0;
}
