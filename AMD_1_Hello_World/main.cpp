#include <hip/hip_runtime.h>
#include <iostream>

// Kernel function to be executed on the GPU
__global__ void hello_kernel() {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = blockId * blockDim.x + threadId;
    printf("Hello from the GPU (Thread ID = %d (local thread %d, block %d)\n", globalId, threadId, blockId);
}

int main() {
    // Launch the kernel on the GPU using 2 threads in 1 block
    hello_kernel<<<1, 2>>>();

    // Synchronize the device to ensure the kernel completes before the host program exits
    hipDeviceSynchronize();

    std::cout << "Hello from the CPU!" << std::endl;

    return 0;
}
