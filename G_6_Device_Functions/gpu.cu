#include <stdio.h>
#include <stdlib.h>

// Device Functions
__device__ float Compute_Reynolds_Number(float V, float density, float length) {
    const float viscosity = 1.81e-5;  // Air viscosity
    return (density*length*V/viscosity);
}

/*
    Computes the Reynolds number for an array holding N mass flow rates and areas.
    Each thread computes Re for one of these elements, needing at least N threads total.
*/
__global__ void GPU_Compute_Re(float *mass_flow_rate, float *area, float *Re, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    const float density = 1.25;       // Air density
    if (i < N) {
        float velocity = mass_flow_rate[i]/(area[i]*density);
        float radius = sqrtf(area[i]/3.14159);
        Re[i] = Compute_Reynolds_Number(velocity, density, 2.0*radius);
    }
}

// Host Functions

void Allocate_Memory(float **h_mass_flow_rate, float **h_area, float **h_Re, float **d_mass_flow_rate, float **d_area, float **d_Re, int N) {
    // Allocate arrays holding N values of mass flow rate area and Re on both CPU and GPU.
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Host memory
    *h_mass_flow_rate = (float*)malloc(size); 
    *h_area = (float*)malloc(size);
    *h_Re = (float*)malloc(size); 
    // Device memory
    Error = cudaMalloc((void**)d_mass_flow_rate, size); 
    printf("CUDA error (malloc d_mass_flow_rate) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_area, size); 
    printf("CUDA error (malloc d_area) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Re, size); 
    printf("CUDA error (malloc d_Re) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_mass_flow_rate, float **h_area, float **h_Re, float **d_mass_flow_rate, float **d_area, float **d_Re) {
    if (*h_mass_flow_rate) free(*h_mass_flow_rate);
    if (*h_area) free(*h_area);
    if (*h_Re) free(*h_Re);
    if (*d_mass_flow_rate) cudaFree(*d_mass_flow_rate);
    if (*d_area) cudaFree(*d_area);
    if (*d_Re) cudaFree(*d_Re);
}

void Send_To_Device(float **h_a, float **d_a, const char *name, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send A to the GPU
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy %s) = %s\n", name, cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, const char *name, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy %s) = %s\n", name, cudaGetErrorString(Error));
}

void Compute_Re(float *d_mass_flow_rate, float *d_area, float *d_Re, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GPU_Compute_Re<<<blocksPerGrid, threadsPerBlock>>>(d_mass_flow_rate, d_area, d_Re, N);
}
