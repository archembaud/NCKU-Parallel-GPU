#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void GPU_Set_Device(int device) {
    cudaSetDevice(device);
}

void Allocate_Memory(float **h_temp, float **d_temp, float **d_temp_new, int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    *h_temp = (float*)malloc(size);
    Error = cudaMalloc((void**)d_temp, size); 
    printf("CUDA error (malloc d_temp) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_temp_new, size); 
    printf("CUDA error (malloc d_temp_new) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_temp, float **d_temp, float **d_temp_new) {
    if (*h_temp) free(*h_temp);
    if (*d_temp) cudaFree(*d_temp);
    if (*d_temp_new) cudaFree(*d_temp_new);
}

void Send_To_Device(float **h_a, float **d_a, const char *name, int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy %s) = %s\n", name, cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, const char *name, int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy %s) = %s\n", name, cudaGetErrorString(Error));
}

void Update_T(float **d_temp, float **d_temp_new, int N) {
    // This is strictly copying only the properly computed 0.5N cells
    // This is important as we treat our far left and right bounds as constant
    // This may change if we have an insulated boundary condition.
    size_t size = N*sizeof(float);
    cudaError_t Error;
    Error = cudaMemcpy(*d_temp+1, *d_temp_new+1, size, cudaMemcpyDeviceToDevice); 
    printf("CUDA error (d_temp_new -> d_temp) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPU_Compute_Tnew(float *d_temp, float *d_temp_new, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int index = i+1; // Because we have N proper cells, and 2 buffer cells (left and right)
    if (i < N) {
        // dt*alpha/(DX*DX) = 0.25
        // Note we don't need to handle the boundaries - they are managed through
        // the buffer cells.
        d_temp_new[index] = d_temp[index] + 0.25*(d_temp[index-1] + d_temp[index+1] - 2.0*d_temp[index]); 
    }
}

void Compute_Tnew(float *d_temp, float *d_temp_new, int N) {
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GPU_Compute_Tnew<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_temp_new, N);
}

void Collect_Boundaries(float **d_temp, float **swap, int N, int tid) {
    size_t size = sizeof(float);
    float buffer_value;
    if (tid == 0) {
        // Collect the right hand side of the 0'th domain
        cudaMemcpy(&buffer_value, &(*d_temp)[N], size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize(); // Wait for the memory copy to complete
        (*swap)[0] = buffer_value;
    } else if (tid == 1) {
        // Collect the left hand side of the 1st domain
        cudaMemcpy(&buffer_value, &(*d_temp)[1], size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize(); // Wait for the memory copy to complete
        (*swap)[1] = buffer_value;
    }
}

void Distribute_Boundaries(float **d_temp, float **swap, int N, int tid) {
    size_t size = sizeof(float);
    float buffer_value;
    // Now we send the buffer cell values to the neighbour domains
    // We only have 2x domains, so this is easy.
    if (tid == 0) {        
        buffer_value = (*swap)[1];
        cudaMemcpy(&(*d_temp)[N+1], &buffer_value, size, cudaMemcpyHostToDevice);
    } else if (tid == 1) {
        buffer_value = (*swap)[0];
        cudaMemcpy(&(*d_temp)[0], &buffer_value, size, cudaMemcpyHostToDevice);
    }
}